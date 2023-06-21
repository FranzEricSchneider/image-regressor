import argparse
from collections import namedtuple
import cv2
import gc
import imageio
import json
import numpy
from pathlib import Path
from PIL import Image
from scipy import signal
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, folder


# Inspired by
# https://towardsdatascience.com/using-shap-to-debug-a-pytorch-image-regression-model-4b562ddef30d
class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, paths, transform, key, channels, is_autoencoder=False, include_path=False
    ):

        self.transform = transform
        self.paths = paths
        self.key = key
        self.channels = channels
        self.is_autoencoder = is_autoencoder
        self.include_path = include_path

    def __getitem__(self, idx):
        """Get image and target value"""

        # Read image
        path = self.paths[idx]

        def has_extension(extensions):
            return any([path.name.endswith(e) for e in extensions])

        if self.channels == 3:
            assert has_extension([".png", ".jpg"])
            image = cv2.cvtColor(
                cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        elif self.channels == 1:
            assert has_extension([".png", ".jpg"])
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            assert has_extension([".npy"])
            image = numpy.load(path)

        image = Image.fromarray(image)
        # Transform image
        image = self.transform(image)

        # Get target
        if self.is_autoencoder is True:
            target = image.clone()
        else:
            target = torch.Tensor(self.get_target(path))

        if self.include_path:
            return image, target, str(path)
        else:
            return image, target

    def get_target(self, path):
        name = path.with_suffix(".json").name
        metadata = json.load(path.parent.joinpath(name).open("r"))
        return [metadata[self.key]]

    def __len__(self):
        return len(self.paths)


def build_transform(augpath):
    assert augpath.is_file()
    return transforms.Compose(
        [
            (
                # First looks for functions in torchvision
                getattr(transforms, name)(**kwargs)
                if hasattr(transforms, name)
                else
                # Then looks for locally defined functions
                globals()[name](**kwargs)
            )
            for name, kwargs in json.load(augpath.open("r"))
        ]
    )


def build_loader(
    data_path,
    batch_size,
    augpath,
    shuffle,
    key,
    extension,
    channels,
    is_autoencoder=False,
    include_path=False,
):

    transform = build_transform(Path(augpath))

    dataset = ImageDataset(
        paths=sorted(data_path.glob(f"*{extension}")),
        transform=transform,
        key=key,
        channels=channels,
        is_autoencoder=is_autoencoder,
        include_path=include_path,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, pin_memory=True, batch_size=batch_size, shuffle=shuffle
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
            is_autoencoder=config["is_autoencoder"],
        )
    test_loader = build_loader(
        data_path=config["data_dir"].joinpath("test"),
        batch_size=config["batch_size"],
        augpath=config["test_augmentation_path"],
        shuffle=False,
        key=config["regression_key"],
        extension=config["extension"],
        channels=config["starting_channels"],
        is_autoencoder=config["is_autoencoder"],
    )

    if debug:
        print()
        print("=" * 80)
        print("Batch size: ", config["batch_size"])
        print(f"Train batches = {len(train_loader)}")
        print(f"Test batches = {len(test_loader)}")
        for name, loader in (("TRAIN", train_loader), ("TEST", test_loader)):
            for x, y in loader:
                print(
                    f"{name}: x.shape: {x.shape}, y.shape: {y.shape}, y[:4]: {y[:4].flatten()}"
                )
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
        self, fill_value=0, min_cut_ratio=0.1, max_cut_ratio=0.5, num_cutouts=5, p=0.5
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
        position = namedtuple("Point", "x y")
        return position(
            numpy.random.randint(0, img_width - cutout_size + 1),
            numpy.random.randint(0, img_height - cutout_size + 1),
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
            int(min_side * self.min_cut_ratio), int(min_side * self.max_cut_ratio)
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
            image[
                :,
                cutout_pos.y : cutout_pos.y + cutout_size,
                cutout_pos.x : cutout_size + cutout_pos.x,
            ] = self.fill_value

        return image


class SunGlare(object):
    """
    Custom brightness augmentation.
    """

    def __init__(
        self,
        max_added_brightness=(0.2, 0.5),
        std_ratio=(0.03, 0.08),
        w_to_h_ratio=(0.75, 1.25),
        p=0.5,
    ):
        """
        TODO
        """

        self.max_added_brightness = max_added_brightness
        self.std_ratio = std_ratio
        self.w_to_h_ratio = w_to_h_ratio
        self.p = p

    def _gaussian_kernel(self, sidelen, std):
        """Returns a 2D Gaussian kernel array."""
        kernel_1d = signal.gaussian(sidelen, std=std).reshape(sidelen, 1)
        return numpy.outer(kernel_1d, kernel_1d)

    def _get_fill(self, img_height, img_width):
        min_side = min(img_height, img_width)
        radius = numpy.random.randint(
            int(min_side * self.std_ratio[0]), int(min_side * self.std_ratio[1])
        )
        kernel = self._gaussian_kernel(sidelen=8 * radius, std=radius)

        # Cap value
        maxval = numpy.random.uniform(*self.max_added_brightness)
        fill = kernel * maxval / kernel.max()

        # Deform
        w_to_h = numpy.random.uniform(*self.w_to_h_ratio)
        new_height = int(kernel.shape[1] * w_to_h)
        fill = cv2.resize(src=fill, dsize=(kernel.shape[1], new_height))

        return fill

    def _get_slices(self, img_height, img_width, fill):
        """
        TODO: Explain (complicated)
        """

        position = namedtuple("Point", "x y")
        start = position(
            numpy.random.randint(-fill.shape[1] // 2, img_width // 2 + 1),
            numpy.random.randint(-fill.shape[0] // 2, img_height // 2 + 1),
        )

        return (
            (
                slice(max(0, start.y), min(img_height, start.y + fill.shape[0])),
                slice(max(0, start.x), min(img_width, start.x + fill.shape[1])),
            ),
            (
                slice(max(-start.y, 0), min(img_height - start.y, fill.shape[0])),
                slice(max(-start.x, 0), min(img_width - start.x, fill.shape[1])),
            ),
        )

    def __call__(self, image):

        if torch.rand(1) > self.p:
            return image

        fill = self._get_fill(*image.shape[1:])
        im_slices, fill_slices = self._get_slices(*image.shape[1:], fill)
        image[:, im_slices[0], im_slices[1]] = (
            image[:, im_slices[0], im_slices[1]] + fill[fill_slices[0], fill_slices[1]]
        )

        return torch.clamp(image, 0, 1)


class ShadowBar(object):
    """
    Custom shadowed augmentation.
    """

    def __init__(
        self,
        max_shade=(0.2, 0.5),
        sides_available=(0, 1, 2, 3),
        width_ratio=(0.15, 0.5),
        p=0.5,
    ):
        """
        TODO
            sides_available: I'm just making this up, but let's make the sides
                available 0 (top), 1 (bottom), 2 (left), and 3 (right)
        """

        self.max_shade = max_shade
        self.sides_available = sides_available
        self.width_ratio = width_ratio
        self.p = p

    def _get_mask(self, img_height, img_width):

        # Make these mappings based on which number means which side
        side_lens = {0: img_width, 1: img_width, 2: img_height, 3: img_height}

        # Get the start and end point
        pts = []
        for side in numpy.random.choice(self.sides_available, size=2, replace=False):
            randlen = numpy.random.randint(0, side_lens[side])
            if side == 0:
                pts.append([randlen, 0])
            elif side == 1:
                pts.append([randlen, img_height])
            elif side == 2:
                pts.append([0, randlen])
            else:
                pts.append([img_width, randlen])

        # Get the width
        min_side = min(img_height, img_width)
        width = int(numpy.random.uniform(*self.width_ratio) * min_side)

        mask = numpy.zeros((img_height, img_width), dtype=float)
        mask = cv2.line(img=mask, pt1=pts[0], pt2=pts[1], color=1, thickness=width)
        return mask.astype(bool)

    def __call__(self, image):

        if torch.rand(1) > self.p:
            return image

        shade_val = numpy.random.uniform(*self.max_shade)
        mask = self._get_mask(*image.shape[1:])

        shade_fill = mask.astype(numpy.float32) * shade_val
        image = image - shade_fill

        return torch.clamp(image, 0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate image stats useful for loading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--image-directory",
        help="Directory with all images we want to examine",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-e",
        "--extension",
        help="Image extension WITH period (e.g. '.png')",
        default=".jpg",
    )
    parser.add_argument(
        "-n",
        "--number",
        help="Number of images to sample (limited by memory constraints)",
        type=int,
        default=200,
    )
    args = parser.parse_args()
    assert args.image_directory.is_dir()

    files = args.image_directory.glob(f"*{args.extension}")
    imgs = [imageio.imread(x) for x, _ in zip(files, range(args.number))]
    # Tile and scale to 0-1
    imgs = numpy.concatenate(imgs, axis=0) / 255
    mean = numpy.mean(imgs, axis=(0, 1))
    std = numpy.std(imgs, axis=(0, 1))
    print(f"mean: {mean}, stdev: {std}")
