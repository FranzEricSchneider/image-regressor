import numpy as np
import torch


def parse_lrtest_kwargs(optimizer, loader, min_per_epoch, runtime_min,
                        start=1e-6, end=1, step_size=50):
    """
    This should work so that StepLR will increase the learning rate from
    start to end LR in runtime_min minutes.
    """

    # Reset the optimizer learning rate
    for group in optimizer.param_groups:
        group["lr"] = start

    desired_scale = end / start
    num_steps = (runtime_min / min_per_epoch) * len(loader) / step_size
    # gamma^num_steps = desired_scale
    gamma = np.power(desired_scale, 1 / num_steps)

    return {"step_size": step_size, "gamma": gamma}


def parse_onecycle_kwargs(loader, epochs, max_lr, min_lr):
    return {
        "max_lr": max_lr,
        "final_div_factor": max_lr / min_lr,
        "steps_per_epoch": len(loader),
        "epochs": epochs,
    }


def parse_cosmulti_kwargs(loader, epoch_per_cycle, eta_min):
    return {
        "T_max": epoch_per_cycle * len(loader),
        "eta_min": eta_min,
    }


def get_scheduler(config, optimizer, loader):
    if config["scheduler"] == "LRTest":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            **parse_lrtest_kwargs(optimizer, loader, **config["LRTest_kwargs"]),
        )
    elif config["scheduler"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            **parse_onecycle_kwargs(loader, config["epochs"], **config["OneCycleLR_kwargs"])
          )
    # elif config["scheduler"] == "CyclicLR":
    #     scheduler = torch.optim.lr_scheduler.CyclicLR(
    #         optimizer,
    #         base_lr=config["cyclic-baselr"],
    #         max_lr=config["1cycle-maxlr"],
    #         step_size_up=int(0.5 * len(loader)),
    #         cycle_momentum=False,
    #     )
    elif config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **config["StepLR_kwargs"],
        )
    elif config["scheduler"] == "CosMulti":
        # This will be called every batch, hence the T_max=number of batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **parse_cosmulti_kwargs(loader, **config["CosMulti_kwargs"])
        )
    # elif config["scheduler"] == "Cos":
    #     # This will be called every epoch
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer,
    #         T_max=config["epochs"],
    #         eta_min=config["1cycle-minlr"],
    #     )
    # elif config["scheduler"] == "ReduceLROnPlateau":
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode="max",
    #         **config["plateau_kwargs"],
    #         # patience=config["plateau-patience"],
    #         # factor=config["plateau-scale"],
    #         # min_lr=config["plateau-min"],
    #     )
    elif config["scheduler"] == "constant":
        scheduler = None
    else:
        raise NotImplementedError()

    return scheduler
