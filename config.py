CONFIG = {
    "data_dir": None,
    "extension": "jpg",
    "regression_key": "value",
    "starting_channels": 1,

    "wandb": True,
    "wandb_print": ["batch_size",
                    "lr",
                    "cnn_depth",
                    "cnn_kernel",
                    "cnn_width",
                    "cnn_outdim",
                    "cnn_downsample",
                    "pool",
                    "lin_depth",
                    "lin_width"],
    "keyfile": "/hdd/wandb.json",
    "train": True,

    "models": [],
    # "models": ["checkpoint.pth"],
    # "models": [{"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/3q34k58v", "replace": True}],
    # "models": [{"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/fqsx3zdk", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/37z196qx", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/phan45yu", "replace": True}],
    # "models": ["checkpoint.pth",
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw2p3-ablations/sk5209ak", "replace": True}],

    "lr": 5e-2,
    "scheduler": "constant",
    "StepLR_kwargs": {"step_size": 5, "gamma": 0.2},
    "LRTest_kwargs": {"min_per_epoch": 5, "runtime_min": 20, "start": 1e-6, "end": 0.5},
    "OneCycleLR_kwargs": {"max_lr": 2.5e-3, "min_lr": 2.5e-6},
    "CosMulti_kwargs": {"epoch_per_cycle": 20, "eta_min": 1.5e-6},

    "batch_size": 1024,  # Increase if you can handle it, generally
    "epochs": 20,
    "wd": 0.01,
    "cnn_depth": 3,
    "cnn_kernel": 3,
    "cnn_width": 256,
    "cnn_outdim": 128,
    "cnn_downsample": 4,
    "pool": "avg",
    "lin_depth": 3,
    "lin_width": 256,

    # How many epochs between validation checks (can take a while)
    "eval_report_iter": 1,
    # How many batches between wandb logs if we are logging batch stats
    "train_report_iter": 10,
}