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


def value_too_high(values, epoch, max_epoch, scale=4):
    if epoch < max_epoch // 2:
        return DEFAULT
    if values[-1] > (scale * values[0]):
        return (
            True,
            f"Value {values[-1]} was more than {scale}x that of the starting"
            f" value, {values[0]}",
        )
    else:
        return DEFAULT


def too_many_nans(values, epoch, max_epoch, number=2):
    count = numpy.isnan(values).sum()
    if count >= number:
        return True, f"Found {count} NaN values in {values}"
    else:
        return DEFAULT


def too_much_growth(values, epoch, max_epoch, width=4, eps=0.01):
    if epoch < max_epoch // 2:
        return DEFAULT
    if numpy.all(numpy.diff(values)[-width:] > eps):
        return True, f"Found {width} consecutive values growing more that {eps}"
    else:
        return DEFAULT


DEFAULT = (False, None)
ENDERS = {
    "too_high": value_too_high,
    "nans": too_many_nans,
    "growth": too_much_growth,
}


def get_tools(loader, model, config, num_cpus):

    if model.is_autoencoder:
        # NOTE - if using SSIM, have to do 1 - SSIM(im1, im2)
        # criterion = SSIM(window_size=11, size_average=True)
        criterion = nn.MSELoss()
        per_input_criterion = nn.MSELoss(reduction="none")
    else:
        criterion = nn.MSELoss()
        per_input_criterion = nn.MSELoss(reduction="none")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )

    scheduler = get_scheduler(config, optimizer, loader)

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    return (criterion, per_input_criterion, optimizer, scheduler, scaler)


def sanity_check(criterion, loader, model, device):

    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):

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
    per_input_criterion,
    scaler,
    config,
    device,
    scheduler=None,
    log_loss=False,
    log_training_images=False,
    train_augmentation_path=None,
):

    model.train()
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )
    train_loss = 0
    result = {"impaths": [], "losses": [], "outputs": []}

    for i, (x, y, paths) in enumerate(loader):

        if log_training_images:
            debug_impaths = save_debug_images(
                paths,
                Path("/tmp/"),
                from_torch=x,
                prefix=f"imgvis_{Path(train_augmentation_path).stem}_",
            )
            print(f"Saved debug images: {debug_impaths}")

        # Zero gradients (necessary to call explicitly in case you have split
        # training up across multiple devices)
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)

        with torch.cuda.amp.autocast():
            # Sample the embeddings the first batch
            if i == 0:
                out, embeddings = model(x, return_embedding=True)
            else:
                out = model(x)
            loss = criterion(out, y)
            per_input_loss = per_input_criterion(out, y)

        # Do some bookkeeping, save these for later use
        result["impaths"].extend(paths)
        result["outputs"].extend([float(o) for o in out.detach().cpu()])
        result["losses"].extend([float(pil) for pil in per_input_loss.detach().cpu()])

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

        del x, y, out, loss, per_input_loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Return the average loss over all batches
    train_loss /= len(loader)

    print(
        f"{str(datetime.datetime.now())}"
        f"    Avg Train Loss: {train_loss:.4f}"
        f"    LR: {float(optimizer.param_groups[0]['lr']):.1E}"
    )

    return train_loss, result, embeddings


def evaluate(criterion, per_input_criterion, loader, models, device):

    # TODO: Handle the ensemble case
    assert len(models) == 1
    model = models[0]
    model.eval()

    val_loss = 0
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )

    result = {"impaths": [], "losses": [], "outputs": []}

    for i, (x, y, paths) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        with torch.inference_mode():
            # Sample the embeddings the first batch
            if i == 0:
                out, embeddings = model(x, return_embedding=True)
            else:
                out = model(x)
            loss = criterion(out, y)
            per_input_loss = per_input_criterion(out, y)
        val_loss += float(loss.detach().cpu())
        batch_bar.set_postfix(avg_loss=f"{val_loss/(i+1):.4f}")
        batch_bar.update()

        # Do some bookkeeping, save these for later use
        result["impaths"].extend(paths)
        result["outputs"].extend([float(o) for o in out.detach().cpu()])
        result["losses"].extend([float(pil) for pil in per_input_loss.detach().cpu()])

        del x, y, out, loss, per_input_loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Get the average val_loss across the epoch
    val_loss /= len(loader)

    return val_loss, result, embeddings


def run_train(
    train_loader, val_loader, model, config, num_cpus, device, run, debug=False
):

    print(f"Starting training {str(datetime.datetime.now())}")
    torch.cuda.empty_cache()
    gc.collect()

    (criterion, per_input_criterion, optimizer, scheduler, scaler) = get_tools(
        train_loader, model, config, num_cpus
    )
    step_kwargs = {
        "loader": train_loader,
        "model": model,
        "optimizer": optimizer,
        "criterion": criterion,
        "per_input_criterion": per_input_criterion,
        "scaler": scaler,
        "config": config,
        "device": device,
        "log_training_images": config["log_training_images"],
        "train_augmentation_path": config["train_augmentation_path"],
    }
    if config["scheduler"] in ["constant", "CosMulti", "LRTest", "OneCycleLR"]:
        step_kwargs["scheduler"] = scheduler
        if config["scheduler"] == "LRTest":
            step_kwargs["log_loss"] = True

    if debug:
        sanity_check(criterion, train_loader, model, device)

    best_val_loss = 1e6

    losses = {"train": [], "val": []}
    sampled_paths = {}

    for epoch in range(config["epochs"]):
        print("Epoch", epoch + 1)

        train_loss, train_result, train_embeddings = train_step(**step_kwargs)
        if config["scheduler"] == "StepLR":
            scheduler.step()
        losses["train"].append(train_loss)

        log_values = {
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if (
            epoch % config["eval_report_iter"] == 0
            or epoch == config["epochs"] - 1
            or config["scheduler"] == "ReduceLROnPlateau"
        ):
            val_loss, val_result, val_embeddings = evaluate(
                criterion, per_input_criterion, val_loader, [model], device
            )
            print(f"{str(datetime.datetime.now())}    Validation Loss: {val_loss:.4f}")
            log_values.update({"val_loss": val_loss})
            losses["val"].append(val_loss)
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)

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
                sampled_paths = vis_model(
                    models=[model],
                    config=config,
                    loaders=(train_loader, val_loader),
                    device=device,
                    prefixes=("train-train", "train-val"),
                    results=(train_result, val_result),
                    embeddings=(train_embeddings, val_embeddings),
                    sampled_paths=sampled_paths,
                )
            best_val_loss = val_loss

        # End early in some circumstances
        end = False
        for name, kwargs in config.get("end_early", {}).items():
            end, message = ENDERS[name](
                losses["val"], epoch, config["epochs"], **kwargs
            )
            if end is True:
                print("*" * 80)
                print(f"ENDING EARLY\n{message}")
                print("*" * 80)
                break
        if end is True:
            break

    if run is not None:
        run.finish()

    return val_loss
