CONFIG = {
    "data_dir": None,
    "extension": "png",
    "starting_channels": 3,
    "regression_key": "value",
    "is_autoencoder": False,

    "wandb": True,
    "wandb_print":
    [
        "PW",
        "is_autoencoder",
        "pretrained_embedding",
        "frozen_embedding",
    ],
    "keyfile": "/hdd/wandb.json",
    "train": True,
    # Option to log images during training to the /tmp/ dir for visualization.
    # This is likely a slowdown so should be used for debugging.
    "log_images": False,
    "num_vis_images": 1,
    # Choose between "all" or a list of layer indices
    "idx_vis_layers": [3],

    "models": [],
    # "models": ["checkpoint.pth"],
    # "models": [{"name": "checkpoint.pth", "run_path": "image-regression/3q34k58v", "replace": True}],
    # "models": [{"name": "checkpoint.pth", "run_path": "image-regression/fqsx3zdk", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "image-regression/37z196qx", "replace": True},
    #            {"name": "checkpoint.pth", "run_path": "image-regression/phan45yu", "replace": True}],
    # "models": ["checkpoint.pth",
    #            {"name": "checkpoint.pth", "run_path": "image-regression/sk5209ak", "replace": True}],

    # If we define a model using a path to a .pth file, this needs to be a list
    # to the corresponding wandb config.yaml file
    "config_paths": None,

    # "pretrained_embedding": None,
    "pretrained_embedding": {"name": "checkpoint.pth", "run_path": "image-regression/8o91vqqo", "replace": True},
    "frozen_embedding": False,

    "lr": 5e-3,
    "scheduler": "constant",
    "StepLR_kwargs": {"step_size": 5, "gamma": 0.2},
    "LRTest_kwargs": {"min_per_epoch": 2, "runtime_min": 20, "start": 1e-6, "end": 0.5},
    "OneCycleLR_kwargs": {"max_lr": 2.5e-3, "min_lr": 2.5e-6},
    "CosMulti_kwargs": {"epoch_per_cycle": 20, "eta_min": 1.5e-6},

    # Augmentations are stored in a json file as (name, kwargs). They are
    # applied in the loader phase. The way to experiment with augmentations is
    # to make a copy of the file, set those that you want, and then select the
    # files one-by-one as a command-line argument.
    "train_augmentation_path": "./train_augmentations.json",
    "test_augmentation_path": "./test_augmentations.json",

    # Increase if you can handle it, generally
    # "batch_size": 500,  # MNIST original size, regressor
    # "batch_size": 32,  # MNIST scaled up
    # "batch_size": 24,  # Beets
    # "batch_size": 100,  # Outdoors @ //4,//4
    "batch_size": 52,  # Vines @ //4,//4
    # "epochs": 8,  # Beets
    # "epochs": 40,  # Outdoors
    "epochs": 15,  # Vines
    "wd": 0.01,

    # TODO: Try out all of these models on bigger input pictures
    # None
    # resnet18, resnet34, resnet50, resnet101, resnet152,
    # vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
    # DON'T USE FOR NOW: inception_v3
    # densenet121, densenet161, densenet169, densenet201
    # mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
    # efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    # efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    # efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
    "use_existing": None,
    "pretrained": True,

    # Used only when not using an existing encoder
    "cnn_depth": 3,
    "cnn_kernel": 3,
    "cnn_width": 256,
    "cnn_outdim": 128,
    "cnn_downsample": 4,
    "cnn_batchnorm": True,
    "cnn_dropout": None,
    "pool": "max", # "avg",
    "lin_depth": 3,
    "lin_width": 256,
    "lin_batchnorm": True,
    "lin_dropout": None,
    # Choices are None or a number. If a number is given, we will squash the
    # final values with a sigmoid and scale it so it's from 0-limit. If we want
    # to set the lower limit in the future we can expand this.
    "output_limit": None,

    # How many epochs between validation checks (can take a while)
    "eval_report_iter": 1,
    # How many batches between wandb logs if we are logging batch stats
    "train_report_iter": 1,
}