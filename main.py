"""
TODO
"""

import argparse

# from datetime import datetime
# import gc
# import glob
# import math
# import numpy as np
# import os
# import pandas as pd
from pathlib import Path

# import PIL.Image
# from shutil import rmtree
# from sklearn.metrics import accuracy_score
# import time
# from timm.models.layers import DropPath
# from tqdm import tqdm
import torch

# from torch import nn
# from torchsummary import summary
# import torchvision
# import wandb

from image_regressor.loader import get_loaders
from image_regressor.model import get_models
from image_regressor.train import run_train
from image_regressor.utils import load_config, login_wandb, system_check, wandb_run
from image_regressor.vis import vis_model


def main():
    num_cpus, device = system_check()
    config = load_config()
    train_loader, test_loader = get_loaders(config, debug=True)
    run = connect_wandb(config) if config["wandb"] else None
    # Login before getting models so we can modify the config
    if config["wandb"]:
        login_wandb(config)
    models = get_models(config, test_loader, device, debug=True)
    if config["wandb"] and not config["is_autoencoder"]:
        run = connect_wandb(config)
        vis_model(models, config, (test_loader,), device, prefixes=("load-test",))
    else:
        run = None

    if config["train"]:
        assert len(models) == 1
        model = models[0]
        run_train(train_loader, test_loader, model, config, num_cpus, device, run)


if __name__ == "__main__":
    main()
