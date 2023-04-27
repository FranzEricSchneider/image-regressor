import os
import torch
from torch import nn
import torchvision
from torchsummaryX import summary
import yaml
import wandb


def flattener(sequential):
    assert isinstance(sequential, nn.Sequential)
    for item in sequential:
        if isinstance(item, nn.Sequential):
            for recurse_item in flattener(item):
                yield recurse_item
        else:
            yield item


def existing_as_embedder(name, in_channels, pretrained):
    model_function = getattr(torchvision.models, name)
    model = model_function(pretrained=pretrained)

    modules = [m for m in model.children()]

    # Potentially replace the first layer if the starting channel # is wrong
    # Then replace the pooling/linear layers
    # Then get the out channels
    if name.startswith("resnet"):
        modules[0] = replace_conv2d_if_needed(modules[0], in_channels)

        assert isinstance(modules[-2], nn.modules.pooling.AdaptiveAvgPool2d)
        assert isinstance(modules[-1], nn.Linear)
        modules = modules[:-2]

        if name.endswith("18") or name.endswith("34"):
            norm = [m for m in modules[-1][-1].children()][-1]
        else:
            norm = [m for m in modules[-1][-1].children()][-2]
        assert isinstance(norm, nn.BatchNorm2d)
        out_channels = norm.num_features

    elif name.startswith("vgg"):
        modules[0][0] = replace_conv2d_if_needed(modules[0][0], in_channels)

        assert isinstance(modules[-2], nn.modules.pooling.AdaptiveAvgPool2d)
        assert isinstance(modules[-1], nn.Sequential)
        assert isinstance(modules[-1][0], nn.Linear)
        modules = modules[:-2]

        if name.endswith("bn"):
            norm = [m for m in modules[-1].children()][-3]
            assert isinstance(norm, nn.BatchNorm2d)
            out_channels = norm.num_features
        else:
            conv2d = [m for m in modules[-1].children()][-3]
            assert isinstance(conv2d, nn.Conv2d)
            out_channels = conv2d.out_channels

    elif name.startswith("inception"):
        raise NotImplementedError(
            "Something weird is happening here with inception where just"
            " replacing the first Conv2d is insufficient. Would need to read"
            " the paper and potentially replace other layers as well to get it"
            " to work. I suspect there are parallel pathways."
        )
        modules[0].conv = replace_conv2d_if_needed(modules[0].conv, in_channels)

        assert isinstance(modules[-3], nn.modules.pooling.AdaptiveAvgPool2d)
        assert isinstance(modules[-2], nn.Dropout)
        assert isinstance(modules[-1], nn.Linear)
        modules = modules[:-3]

        norm = modules[-1].branch_pool.bn
        assert isinstance(norm, nn.BatchNorm2d)
        out_channels = norm.num_features

    elif name.startswith("densenet"):
        modules[0][0] = replace_conv2d_if_needed(modules[0][0], in_channels)

        assert isinstance(modules[-1], nn.Linear)
        modules = modules[:-1]

        norm = modules[-1][-1]
        assert isinstance(norm, nn.BatchNorm2d)
        out_channels = norm.num_features

    elif name.startswith("mobilenet"):
        modules[0][0][0] = replace_conv2d_if_needed(modules[0][0][0], in_channels)

        if name.endswith("v2"):
            assert isinstance(modules[-2], nn.Sequential)
            assert isinstance(modules[-1][0], nn.Dropout)
            assert isinstance(modules[-1][1], nn.Linear)
            assert len(modules[-1]) == 2
            modules = modules[:-1]
        else:
            assert isinstance(modules[-2], nn.AdaptiveAvgPool2d)
            assert isinstance(modules[-1][0], nn.Linear)
            assert isinstance(modules[-1][1], nn.Hardswish)
            assert isinstance(modules[-1][2], nn.Dropout)
            assert isinstance(modules[-1][3], nn.Linear)
            assert len(modules[-1]) == 4
            modules = modules[:-2]

        norm = modules[-1][-1][1]
        assert isinstance(norm, nn.BatchNorm2d)
        out_channels = norm.num_features

    elif name.startswith("efficientnet"):
        modules[0][0][0] = replace_conv2d_if_needed(modules[0][0][0], in_channels)

        assert isinstance(modules[-2], nn.AdaptiveAvgPool2d)
        assert isinstance(modules[-1][0], nn.Dropout)
        assert isinstance(modules[-1][1], nn.Linear)
        assert len(modules[-1]) == 2
        modules = modules[:-2]

        norm = modules[-1][-1][-2]
        assert isinstance(norm, nn.BatchNorm2d)
        out_channels = norm.num_features

    else:
        raise NotImplementedError()

    return nn.Sequential(*modules), out_channels


def replace_conv2d_if_needed(conv2d, in_channels):
    '''
    Replaces a Conv2d layer with a similar layer but with the right number of
    input channels.
    '''
    assert isinstance(conv2d, nn.Conv2d)
    if conv2d.in_channels != in_channels:
        if isinstance(conv2d.bias, bool):
            bias = conv2d.bias
        else:
            if conv2d.bias is None:
                bias = False
            else:
                bias = True
        new_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            bias=bias,
        )
        return new_conv2d
    else:
        return conv2d


class Network(nn.Module):

    def __init__(self,
                 use_existing,
                 pretrained,
                 frozen_embedding,
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

        if use_existing is not None:
            self.embedding, cnn_outdim = existing_as_embedder(
                use_existing,
                starting_channels,
                pretrained,
            )
            if frozen_embedding:
                for param in self.embedding.parameters():
                    param.requires_grad = False
        else:
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
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pool(x)
        x = torch.squeeze(x, dim=(2, 3))
        x = self.classifier(x)
        return x


def network_kwargs(config):
    return {
        "use_existing": config["use_existing"],
        "pretrained": config["pretrained"],
        "frozen_embedding": config["frozen_embedding"],
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


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# TODO: This is set up for use in TRAINING, can it be simplified for execution
# and inference?
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
                kwargs = network_kwargs(load_wandb_config(settings["run_path"]))
            else:
                raise NotImplementedError(f"Unknown setting type for {config['models']}")
            model = Network(**kwargs).to(device)
            model.load_state_dict(
                torch.load(
                    load_file,
                    map_location=torch.device(device)
                )["model_state_dict"]
            )
            models.append(model)

    if debug:
        for x, y in loader:
            x = x.to(device)
            for i, model in enumerate(models):
                print(f"NUMBER {i+1}")
                summary(model, x.to(device))
                print("FLATTENED EMBEDDING")
                for i, element in enumerate(flattener(model.embedding)):
                    print(f"IDX: {i}\n{element})")
            break

    # Save the model size in the config
    config["model_sizes"] = {
        f"model-{i}": count_parameters(model)
        for i, model in enumerate(models)
    }

    return models
