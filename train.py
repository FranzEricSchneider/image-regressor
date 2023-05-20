import cv2
import datetime
import gc
import numpy
from pathlib import Path
import time
from tqdm import tqdm
import torch
from torch import nn
import wandb

from image_regressor.scheduler import get_scheduler
from image_regressor.vis import save_debug_images, vis_model


numpy.set_printoptions(suppress=True, precision=3)


def get_tools(loader, model, config, num_cpus):

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )

    scheduler = get_scheduler(config, optimizer, loader)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    return (criterion, optimizer, scheduler, scaler)


def sanity_check(criterion, loader, model, device):

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):

            x = x.to(device)
            y = y.to(device)
            print("x.shape", x.shape)
            print("y.shape", y.shape)

            model.eval()
            with torch.inference_mode():
                out = model(x)
                loss = criterion(out, y)

            print("out.shape", out.shape)
            print(f"Mean squared prediction error: {loss}")

            break


def train_step(
    loader,
    model,
    optimizer,
    criterion,
    scaler,
    config,
    device,
    scheduler=None,
    log_loss=False,
    log_images=False,
):

    model.train()
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )
    train_loss = 0

    for i, (x, y) in enumerate(loader):

        if log_images:
            save_debug_images(x, Path("/tmp/"))

        # Zero gradients (necessary to call explicitly in case you have split
        # training up across multiple devices)
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()  # A.k.a. loss.backward()
        scaler.step(optimizer)  # A.k.a. optimizer.step()
        scaler.update()

        train_loss += float(loss.detach().cpu())
        batch_bar.set_postfix(
            loss=f"{train_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()

        if log_loss and config["wandb"]:
            if i % config["train_report_iter"] == 0:
                wandb.log({"batch-loss": loss.item()})
                wandb.log({"batch-lr": float(scheduler.get_last_lr()[0])})
        if scheduler is not None:
            scheduler.step()

        del x, y, out, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Return the average loss over all batches
    train_loss /= len(loader)

    print(
        f"{str(datetime.datetime.now())}"
        f"    Avg Train Loss: {train_loss:.4f}"
        f"    LR: {float(optimizer.param_groups[0]['lr']):.1E}"
    )

    return train_loss


def evaluate(criterion, loader, models, device):

    # TODO: Handle the ensemble case
    assert len(models) == 1
    model = models[0]
    model.eval()

    loss = 0
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.inference_mode():
            out = model(x)
            this_loss = criterion(out, y)
        loss += this_loss
        batch_bar.set_postfix(avg_loss=f"{loss/(i+1):.4f}")
        batch_bar.update()

        del x, y, out
        torch.cuda.empty_cache()

    batch_bar.close()

    # Get the average loss across the epoch
    loss /= len(loader)

    return loss


def run_train(
    train_loader, val_loader, model, config, num_cpus, device, run, debug=False
):

    print(f"Starting training {str(datetime.datetime.now())}")
    torch.cuda.empty_cache()
    gc.collect()

    (criterion, optimizer, scheduler, scaler) = get_tools(
        train_loader, model, config, num_cpus
    )
    step_kwargs = {
        "loader": train_loader,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "scaler": scaler,
        "config": config,
        "device": device,
        "log_images": config["log_images"],
    }
    if debug:
        sanity_check(criterion, train_loader, model, device)

    best_val_loss = 1e6

    for epoch in range(config["epochs"]):
        print("Epoch", epoch + 1)

        if config["scheduler"] == "StepLR":
            train_loss = train_step(**step_kwargs)
            scheduler.step()
        elif config["scheduler"] in ["CosMulti", "OneCycleLR", "constant"]:
            train_loss = train_step(**step_kwargs, scheduler=scheduler)
        elif config["scheduler"] == "LRTest":
            train_loss = train_step(**step_kwargs, scheduler=scheduler, log_loss=True)
        else:
            raise NotImplementedError()

        log_values = {
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if epoch % config["eval_report_iter"] == 0 or epoch == config["epochs"] - 1:
            val_loss = evaluate(criterion, val_loader, [model], device)
            print(
                f"{str(datetime.datetime.now())}" f"    Validation Loss: {val_loss:.4f}"
            )
            log_values.update({"val_loss": val_loss})
        if config["wandb"]:
            wandb.log(log_values)

        if val_loss < best_val_loss:
            print("Saving model")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "epoch": epoch,
                },
                "./checkpoint.pth",
            )
            if config["wandb"]:
                wandb.save("checkpoint.pth")
                if not config["is_autoencoder"]:
                    vis_model(
                        [model],
                        config,
                        (val_loader, train_loader),
                        device,
                        prefixes=("train-val", "train-train"),
                    )
            best_val_loss = val_loss

    if run is not None:
        run.finish()

    return val_loss
