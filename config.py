CONFIG = {
    "data_dir": None,
    "extension": "jpg",
    "regression_key": "value",

    "wandb": True,
    "wandb_print": ["batch_size", "cnn_width", "lin_depth", "lin_width"],
    "train": True,
    "submit": False,
    "submit_override_eval": 4.0,

    "models": [],
    # "models": ["checkpoint.pth"],
    # "models": [{"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/3q34k58v", "replace": True}],
    # "models": [{"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/fqsx3zdk", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/37z196qx", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw3p2-ablations/phan45yu", "replace": True}],
    # "models": ["checkpoint.pth",
    #            {"name": "checkpoint.pth", "run_path": "diplernerz/hw2p3-ablations/sk5209ak", "replace": True}],

    "lr": 5e-2,
    "scheduler": "OneCycleLR",
    "StepLR_kwargs": {"step_size": 5, "gamma": 0.2},
    "LRTest_kwargs": {"min_per_epoch": 5, "runtime_min": 20, "start": 1e-6, "end": 0.5},
    "OneCycleLR_kwargs": {"max_lr": 2.5e-3, "min_lr": 2.5e-6},
    "CosMulti_kwargs": {"epoch_per_cycle": 20, "eta_min": 1.5e-6},

    "batch_size": 128,  # Increase if you can handle it
    "epochs": 20,
    "wd": 0.1,
    "cnn_depth": 5,
    "cnn_kernel": 5,
    "cnn_width": 256,
    "cnn_outdim": 128,
    "cnn_downsample": 4,
    "lin_depth": 3,
    "lin_width": 256,
    "beam_width": 5,
    "test_beam_width": 30,

    # How many epochs between validation checks (can take a while)
    "eval_report_iter": 1,
    # How many batches between wandb logs if we are logging batch stats
    "train_report_iter": 50,
}