import os
import torch
from torch import nn
from torchsummaryX import summary
import yaml
import wandb


class Network(nn.Module):

    def __init__(self,
                 starting_channels,
                 cnn_depth,
                 cnn_kernel,
                 cnn_width,
                 cnn_outdim,
                 cnn_downsample,
                 cnn_batchnorm,
                 cnn_dropout,
                 pool,
                 lin_depth,
                 lin_width,
                 lin_batchnorm,
                 lin_dropout):
        super(Network, self).__init__()

        self.embedding = None
        layers = []
        downsampled = 1
        for i in range(cnn_depth):

            in_channels = cnn_width
            out_channels = cnn_width
            if i == 0:
                in_channels = starting_channels
            if i == cnn_depth - 1:
                out_channels = cnn_outdim

            if downsampled < cnn_downsample:
                stride = 2
                downsampled *= 2
            else:
                stride = 1
            layers.append(nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size=cnn_kernel,
                                    stride=stride))
            if i < cnn_depth - 1:
                if cnn_batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU())
                if cnn_dropout is not None:
                    layers.append(nn.Dropout(p=cnn_dropout))
        self.embedding = nn.Sequential(*layers)

        # Pool all features into one across the image
        if pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == "max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError(f"Unknown pool {pool}")

        layers = []
        for i in range(lin_depth):
            in_features = lin_width
            out_features = lin_width
            if i == 0:
                in_features = cnn_outdim
            if i == lin_depth - 1:
                out_features = 1
            layers.append(nn.Linear(in_features, out_features))
            if i == lin_depth - 1:
                pass
            else:
                if lin_batchnorm:
                    layers.append(nn.BatchNorm1d(out_features))
                layers.append(nn.ReLU())
                if lin_dropout is not None:
                    layers.append(nn.Dropout(p=lin_dropout))
        self.classification = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pool(x)
        x = torch.squeeze(x)
        x = self.classification(x)
        return x


def network_kwargs(config):
    return {
        "starting_channels": config["starting_channels"],
        "cnn_depth": config["cnn_depth"],
        "cnn_kernel": config["cnn_kernel"],
        "cnn_width": config["cnn_width"],
        "cnn_outdim": config["cnn_outdim"],
        "cnn_downsample": config["cnn_downsample"],
        "cnn_batchnorm": config["cnn_batchnorm"],
        "cnn_dropout": config["cnn_dropout"],
        "pool": config["pool"],
        "lin_depth": config.get("lin_depth", 1),
        "lin_width": config.get("lin_width", 256),
        "lin_batchnorm": config["lin_batchnorm"],
        "lin_dropout": config["lin_dropout"],
    }


def load_wandb_config(run_path):
    """
    The wandb config splits config values into desc (description) and value
    (the stuff we want). Undo that.
    """
    wandb.restore(name="config.yaml", run_path=run_path, replace=True)
    wandb_dict = yaml.safe_load(open("config.yaml", "r"))
    loaded_config = {}
    for key, value in wandb_dict.items():
        if isinstance(value, dict) and "value" in value:
            loaded_config[key] = value["value"]
    return loaded_config


def get_models(config, loader, device, debug=False):

    torch.cuda.empty_cache()

    if len(config["models"]) == 0:
        models = [
            Network(**network_kwargs(config)).to(device)
        ]
    else:
        models = []
        for settings in config["models"]:
            if isinstance(settings, str):
                load_file = settings
                kwargs = network_kwargs(config)
            elif isinstance(settings, dict):
                wandb.restore(**settings)
                load_file = f"{settings['run_path'].replace('/', '_')}.pth"
                os.rename(settings["name"], load_file)
                try:
                    kwargs = network_kwargs(load_wandb_config(settings["run_path"]))
                except ValueError:
                    # No file found in wandb
                    kwargs = network_kwargs(config)
            else:
                raise NotImplementedError(f"Unknown setting type for {config['models']}")
            model = Network(**kwargs).to(device)
            model.load_state_dict(torch.load(load_file, map_location=torch.device(device))["model_state_dict"])
            models.append(model)

    if debug:
        for x, y in loader:
            x = x.to(device)
            for i, model in enumerate(models):
                print(f"NUMBER {i+1}")
                summary(model, x.to(device))
            break

    return models
