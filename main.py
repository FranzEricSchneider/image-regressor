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
from utils import (load_config, system_check)


def main():
    num_cpus, device = system_check()
    config = load_config()
    train_loader, val_loader, test_loader = get_loaders(config, debug=True)


if __name__ == "__main__":
    main()
