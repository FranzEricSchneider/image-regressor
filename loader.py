from collections import namedtuple
import cv2
import gc
import json
import numpy
from pathlib import Path
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


def build_loader(data_path, batch_size, augpath, shuffle, key, extension,
                 channels):

    augpath = Path(augpath)
    assert augpath.is_file()
    transform = transforms.Compose([
        (
            # First looks for functions in torchvision
            getattr(transforms, name)(**kwargs)
            if hasattr(transforms, name) else
            # Then looks for locally defined functions
            globals()[name](**kwargs)
        )
        for name, kwargs in json.load(augpath.open("r"))
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
            augpath=config["train_augmentation_path"],
            shuffle=True,
            key=config["regression_key"],
            extension=config["extension"],
            channels=config["starting_channels"],
        )
    test_loader = build_loader(
        data_path=config["data_dir"].joinpath("test"),
        batch_size=config["batch_size"],
        augpath=config["test_augmentation_path"],
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


class CutoutBoxes(object):
    """
    Custom Cutout augmentation (modified from below to remove bbox stuff)
    Note: (only supports square cutout regions)
    https://www.kaggle.com/code/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup
    """

    def __init__(
        self,
        fill_value=0,
        min_cut_ratio=0.1,
        max_cut_ratio=0.5,
        num_cutouts=5,
        p=0.5,
    ):
        """
        Class constructor
        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param min_cut_ratio: minimum size of cutout (192 x 192)
        :param max_cut_ratio: maximum size of cutout (512 x 512)
        """
        self.fill_value = fill_value
        self.min_cut_ratio = min_cut_ratio
        self.max_cut_ratio = max_cut_ratio
        self.num_cutouts = num_cutouts
        self.p = p

    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple

        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            numpy.random.randint(0, img_width - cutout_size + 1),
            numpy.random.randint(0, img_height - cutout_size + 1)
        )

    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image

        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        min_side = min(img_height, img_width)
        cutout_size = numpy.random.randint(
            int(min_side * self.min_cut_ratio),
            int(min_side * self.max_cut_ratio),
        )
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return (cutout_size, cutout_position)

    def __call__(self, image):
        """
        Applies the cutout augmentation on the given image

        :param image: The image to be augmented
        :returns augmented image
        """

        if torch.rand(1) > self.p:
            return image

        for _ in range(numpy.random.randint(1, self.num_cutouts + 1)):
            cutout_size, cutout_pos = self._get_cutout(*image.shape[1:])
            image[:,
                  cutout_pos.y:cutout_pos.y+cutout_size,
                  cutout_pos.x:cutout_size+cutout_pos.x] = self.fill_value

        return image
