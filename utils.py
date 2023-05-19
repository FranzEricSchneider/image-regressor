import argparse
import copy
import json
import multiprocessing
from pathlib import Path
from psutil import virtual_memory
import subprocess
import torch
import wandb

from config import CONFIG


def login_wandb(config):

    keyfile = Path(config["keyfile"])
    assert keyfile.is_file(), \
           f"Need to populate {keyfile} with json containing wandb key"
    wandb.login(key=json.load(keyfile.open("r"))["key"])


def wandb_run(config):

    name = "-".join([
        f"{key}:{config[key]}" if (key in config and not isinstance(config[key], dict)) else key
        for key in config["wandb_print"]
    ])

    run = wandb.init(
        # Wandb creates random run names if you skip this field
        name=name,
        # Allows reinitalizing runs
        reinit=True,
        # Insert specific run id here if you want to resume a previous run
        # run_id=
        # You need this to resume previous runs, but comment out reinit=True when using this
        # resume="must"
        # Project should be created in your wandb account
        project="image-regression",
        config=config,
    )
    return run


def load_config():
    config = copy.copy(CONFIG)

    parser = argparse.ArgumentParser(description="Set config via command line")
    parser.add_argument("data_dir",
                        type=Path,
                        help="Path to dir with all images and labels")
    parser.add_argument("--wandb-print", nargs="+", default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=None)
    parser.add_argument("-e", "--epochs", type=int, default=None)
    parser.add_argument("-l", "--lr", type=float, default=None)
    parser.add_argument("-w", "--wd", type=float, default=None)
    parser.add_argument("-g", "--train-augmentation-path", default=None)
    parser.add_argument("--use-existing", default=None)
    parser.add_argument("--pretrained", action="store_true", default=None)
    parser.add_argument("--frozen-embedding", action="store_true", default=None)
    parser.add_argument("-d", "--cnn-depth", type=int, default=None)
    parser.add_argument("-k", "--cnn-kernel", type=int, default=None)
    parser.add_argument("-t", "--cnn-width", type=int, default=None)
    parser.add_argument("-o", "--cnn-outdim", type=int, default=None)
    parser.add_argument("-s", "--cnn-downsample", type=int, default=None)
    parser.add_argument("-r", "--cnn-dropout", type=float, default=None)
    parser.add_argument("-a", "--cnn-batchnorm", action="store_true", default=None)
    parser.add_argument("-p", "--pool", default=None)
    parser.add_argument("-D", "--lin-depth", type=int, default=None)
    parser.add_argument("-W", "--lin-width", type=int, default=None)
    parser.add_argument("-R", "--lin-dropout", type=float, default=None)
    parser.add_argument("-A", "--lin-batchnorm", action="store_true", default=None)
    parser.add_argument("--run-paths", nargs="+", default=None)
    args = parser.parse_args()

    # Blindly fill arguments into the config
    for key in config.keys():
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                config[key] = value

    # Handle these cases special
    if args.run_paths is not None:
        config["models"] = [
            {"run_path": run_path, "name": "checkpoint.pth", "replace": True}
            for run_path in args.run_paths
        ]

    print("\n" + "=" * 36 + " CONFIG " + "=" * 36)
    for k, v in config.items():
        print(f"\t{k}: {v}")
    print("=" * 80)

    return config


def system_check():

    print("=" * 80)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        subprocess.call(["nvidia-smi"])

    # Check RAM
    ram_gb = virtual_memory().total / 1e9
    print("Your runtime has {:.1f} gigabytes of available RAM\n".format(ram_gb))
    if ram_gb < 20:
        print("Not using a high-RAM runtime")
    else:
        print("You are using a high-RAM runtime!")

    # CPUs
    num_cpus = multiprocessing.cpu_count()
    print("Number of CPUs:", num_cpus)

    print("DEVICE", device)

    print("=" * 80)

    return num_cpus, device
