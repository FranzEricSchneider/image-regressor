'''
TODO
'''

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

from loader import get_loaders
from model import get_models
from train import run_train
from utils import (connect_wandb, load_config, system_check)


def main():
    num_cpus, device = system_check()
    config = load_config()
    train_loader, test_loader = get_loaders(config, debug=True)
    models = get_models(config, test_loader, device, debug=True)

    if config["train"]:
        assert len(models) == 1
        model = models[0]
        run = connect_wandb(config) if config["wandb"] else None
        run_train(train_loader, test_loader, model, config, num_cpus, device, run)


if __name__ == "__main__":
    main()
