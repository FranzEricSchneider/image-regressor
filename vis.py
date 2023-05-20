import matplotlib

matplotlib.use("Agg")

import argparse
import cv2
from datetime import datetime
import json
from matplotlib import pyplot
import numpy
from pathlib import Path
from PIL import Image
import time
import torch
from torch import nn
import torch.nn.functional as F
import wandb

from image_regressor.loader import build_loader
from image_regressor.model import flattener, get_models
from image_regressor.utils import login_wandb, system_check


def vis_model(models, config, loaders, device, prefixes):

    impaths = []
    for i, model in enumerate(models):
        for loader, prefix in zip(loaders, prefixes):
            for x, y in loader:
                if config["idx_vis_layers"] == "all":
                    indices = range(len([_ for _ in flattener(model.embedding)]))
                else:
                    indices = config["idx_vis_layers"]

                for target in indices:
                    cam = ScoreCam(model, target_layer=target)
                    for j in range(config["num_vis_images"]):
                        save_path = Path(
                            f"/tmp/{prefix}_model{i}_testim{j}_layer{target}.jpg"
                        )
                        cam.generate_cam(
                            input_image=x[j : j + 1].to(device),
                            target_class=0,
                            save_path=save_path,
                        )
                        impaths.append(save_path)
                break

    wandb.log({impath.name: wandb.Image(str(impath)) for impath in impaths})


def scale_0_1(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def save_autoencoder_images(x, savedir, images, prefix="debug"):

    timestamp = str(int(time.time() * 1e6))

    for label, tensor in (("original", x), ("decoded", images)):
        for i, torch_img in enumerate(tensor):
            name = f"{prefix}_{timestamp}_{i}_{label}.jpg"
            uint8_image = torch_img_to_array(torch_img)
            cv2.imwrite(str(savedir.joinpath(name)), uint8_image)


def save_debug_images(x, savedir, labels=None, prefix="debug"):

    if labels is None:
        labels = [None] * len(x)

    timestamp = str(int(time.time() * 1e6))

    for i, (torch_img, label) in enumerate(zip(x, labels)):
        name = f"{prefix}_{timestamp}_{i}.jpg"
        uint8_image = torch_img_to_array(torch_img)
        if label is not None:
            white = 255
            black = 0
            if len(uint8_image.shape) == 3 and uint8_image.shape[2] == 3:
                white = (255, 255, 255)
                black = (0, 0, 0)
            uint8_image[:50, :100] = white
            cv2.putText(
                img=uint8_image,
                text=f"{label:.1f}",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=black,
                thickness=2,
            )
        cv2.imwrite(str(savedir.joinpath(name)), uint8_image)

    print(f"Saved images as {savedir.joinpath(prefix)}_{timestamp}_0-{len(x) - 1}.jpg")


def torch_img_to_array(torch_img):
    float_image = torch_img.movedim(0, -1).detach().cpu().numpy()
    uint8_image = (float_image * 255).astype(numpy.uint8)
    if uint8_image.shape[2] == 3:
        uint8_image = cv2.cvtColor(uint8_image, cv2.COLOR_RGB2BGR)
    return uint8_image


def visually_label_images(
    imdir,
    savedir,
    run_path,
    wandb_paths,
    augmentation,
    extension,
    number,
    key,
    shuffle,
    keyfile,
):

    # Needed before we can restore the models
    login_wandb({"keyfile": keyfile})

    num_cpus, device = system_check()
    if run_path is not None:
        models = {"name": "checkpoint.pth", "run_path": run_path, "replace": True}
        config_paths = None
    elif wandb_paths is not None:
        # Should be given as a tuple, where each is a Path() to a .pth or .yaml
        # file we want to load
        models, config_path = wandb_paths
        config_paths = [config_path]
    else:
        raise ValueError(f"Invalid paths given: {run_path}, {config_paths}")

    models = get_models(
        config={
            "models": [models],
            "config_paths": config_paths,
            "pretrained_embedding": None,
        },
        loader=None,
        device=device,
        debug=False,
    )
    assert len(models) == 1
    model = models[0]
    model.to(device)
    model.eval()

    loader = build_loader(
        data_path=imdir,
        batch_size=1,
        augpath=augmentation,
        shuffle=shuffle,
        key=key,
        extension=extension,
        # TODO: Expand in the future as necessary
        channels=3,
    )

    print(f"Started visualizing at {datetime.now()}")
    count = 0
    for x, _ in loader:
        x = x.to(device)
        output = model(x)
        if model.is_autoencoder:
            save_autoencoder_images(x, savedir, images=output)
        else:
            labels = output.detach().cpu().numpy().flatten()
            save_debug_images(x, savedir, labels=labels)
        count += len(output)
        if count >= number:
            break
    print(f"Finished visualizing {count} images at {datetime.now()}")


"""
Code adapted for regression from:
https://github.com/utkuozbulak/pytorch-cnn-visualizations

MIT License

Copyright (c) 2017 Utku Ozbulak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""


class CamExtractor:
    """Extracts cam features from the model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """Does forward pass on convolutions, hooks function at given layer."""
        conv_output = None
        # for module_pos, module in self.model.embedding._modules.items():
        for module_pos, module in enumerate(flattener(self.model.embedding)):
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """Does a full forward pass on the model, returns layer output."""
        # Forward pass on the convolutions
        conv_output, _ = self.forward_pass_on_convolutions(x)
        return conv_output, self.model(x)


class ScoreCam:
    """Saves class activation map."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class, save_path):

        # Get this for later resizing
        input_size = (input_image.shape[2], input_image.shape[3])

        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model
        conv_output, model_output = self.extractor.forward_pass(input_image)

        # Get convolution outputs
        target = conv_output[0]

        # Create empty numpy array for cam
        cam = numpy.ones(target.shape[1:], dtype=numpy.float32)

        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):

            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :], 0), 0)

            # Upsampling to input size
            saliency_map = F.interpolate(
                saliency_map, size=input_size, mode="bilinear", align_corners=False
            )
            if saliency_map.max() == saliency_map.min():
                continue
            norm_saliency_map = scale_0_1(saliency_map)

            # Get the target score
            w = F.softmax(
                self.extractor.forward_pass(input_image * norm_saliency_map)[1], dim=1
            )[0][target_class]
            cam += (
                w.data.detach().cpu().numpy()
                * target[i, :, :].data.detach().cpu().numpy()
            )

        cam = numpy.maximum(cam, 0)
        cam = scale_0_1(cam)
        cam = Image.fromarray(cam).resize(input_size[::-1], Image.ANTIALIAS)
        cam = numpy.uint8(numpy.array(cam) * 255)
        if input_image[0].shape[0] != 1:
            cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2RGB)

        original = (input_image[0].permute(1, 2, 0)).detach().cpu().numpy()
        if input_image[0].shape[0] == 1:
            original = numpy.squeeze(original)
        elif input_image[0].shape[0] != 3:
            original = original[:, :, :3]
        original = numpy.uint8(original * 255)
        if numpy.all(original < 10):
            original *= 25

        figure, axis = pyplot.subplots(figsize=(7, 7))
        axis.imshow(numpy.vstack([original, cam]))
        axis.axis("off")
        axis.set_title(f"Score-CAM layer {self.target_layer}")
        pyplot.tight_layout()
        pyplot.savefig(save_path, dpi=100)
        pyplot.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label images based on the results of a model for human"
        " visualization.",
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
        "-o",
        "--output-directory",
        help="Directory for where to store labeled/visualized images",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-k",
        "--wandb-keyfile",
        help="Wandb login info in a json dictionary, has the element key",
        type=Path,
        default=Path("/home/wandb.json"),
    )
    parser.add_argument(
        "-a",
        "--augmentation",
        help="Image extension WITH period (e.g. '.png')",
        type=Path,
        default=Path("./test_augmentations.json"),
    )
    parser.add_argument(
        "-e",
        "--extension",
        help="Image extension WITH period (e.g. '.png')",
        default=".png",
    )
    parser.add_argument(
        "-n", "--number", help="Number of images to label", type=int, default=10
    )
    parser.add_argument(
        "-r",
        "--regression-key",
        help="Value in the json file associated with the measurement",
        default="value",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        help="Shuffle images in the folder before selecting <number>",
        action="store_true",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-w",
        "--wandb-run-path",
        help="Wandb run (e.g. 'image-regression/wnevejfn' in config)",
    )
    group.add_argument(
        "-c",
        "--config-paths",
        help="Path to (.pth, .yaml) for the (model, config) files (space separated)",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()
    assert args.image_directory.is_dir()
    assert args.output_directory.is_dir()
    assert args.wandb_keyfile.is_file()

    visually_label_images(
        imdir=args.image_directory,
        savedir=args.output_directory,
        run_path=args.wandb_run_path,
        wandb_paths=args.config_paths,
        augmentation=args.augmentation,
        extension=args.extension,
        number=args.number,
        key=args.regression_key,
        shuffle=args.shuffle,
        keyfile=args.wandb_keyfile,
    )
