import cv2
import gc
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, folder


# Inspired by
# https://towardsdatascience.com/using-shap-to-debug-a-pytorch-image-regression-model-4b562ddef30d
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform, key, extension, channels):

        self.transform = transform
        self.paths = paths
        self.key = key
        self.extension = extension
        self.channels = channels

    def __getitem__(self, idx):
        """Get image and target value"""
        # Read image
        path = self.paths[idx]
        if self.channels == 3:
            image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR),
                                 cv2.COLOR_BGR2RGB)
        elif self.channels == 1:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError(f"Channels={self.channels} not supported")
        image = Image.fromarray(image)
        # Transform image
        image = self.transform(image)
        # Get target
        target = torch.Tensor(self.get_target(path))
        return image, target

    def get_target(self, path):
        name = path.name.replace(self.extension, "json")
        metadata = json.load(path.parent.joinpath(name).open("r"))
        return [metadata[self.key]]

    def __len__(self):
        return len(self.paths)


def build_loader(data_path, batch_size, shuffle, key, extension, channels):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = ImageDataset(
        paths=sorted(data_path.glob(f"*{extension}")),
        transform=transform,
        key=key,
        extension=extension,
        channels=channels,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


def get_loaders(config, debug=False):

    print("Loading data...")
    gc.collect()

    # TODO: Consider making a separate validation loader
    train_loader = None
    if config["train"]:
        train_loader = build_loader(
            data_path=config["data_dir"].joinpath("train"),
            batch_size=config["batch_size"],
            shuffle=True,
            key=config["regression_key"],
            extension=config["extension"],
            channels=config["starting_channels"],
        )
    test_loader = build_loader(
        data_path=config["data_dir"].joinpath("test"),
        batch_size=config["batch_size"],
        shuffle=False,
        key=config["regression_key"],
        extension=config["extension"],
        channels=config["starting_channels"],
    )

    if debug:
        print()
        print("=" * 80)
        print("Batch size: ", config["batch_size"])
        print(f"Train batches = {len(train_loader)}")
        print(f"Test batches = {len(test_loader)}")
        for name, loader in (("TRAIN", train_loader),
                             ("TEST", test_loader)):
            for x, y in loader:
                print(f"{name}: x.shape: {x.shape}, y.shape: {y.shape}, y[:4]: {y[:4].flatten()}")
                break
        print("=" * 80)

    return train_loader, test_loader
