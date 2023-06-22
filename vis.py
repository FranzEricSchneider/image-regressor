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
from tqdm import tqdm
import wandb

from image_regressor.loader import build_loader
from image_regressor.model import flattener, get_models
from image_regressor.utils import login_wandb, system_check


def read_rgb(impath):
    return cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)


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


def save_autoencoder_images(x, savedir, images, prefix="debug", loss=None):

    timestamp = str(int(time.time() * 1e6))

    for label, tensor, imloss in (("original", x, None), ("decoded", images, loss)):
        for i, torch_img in enumerate(tensor):
            name = f"{prefix}_{timestamp}_{i}_{label}.jpg"
            uint8_image = torch_img_to_array(torch_img)
            if imloss is not None:
                imloss = f"{imloss:.2f}"
                print(f"Loss: {imloss}")
                highlight_text(uint8_image, imloss)
                name.replace(".jpg", f"_{imloss}.jpg")
            cv2.imwrite(str(savedir.joinpath(name)), uint8_image)


def save_debug_images(impaths, savedir, labels=None, prefix="debug", from_torch=None, metakeys=None):

    if labels is None:
        labels = [None] * len(x)

    for i, (impath, label) in enumerate(zip(impaths, labels)):

        if from_torch is None:
            uint8_image = cv2.imread(str(impath))
        else:
            uint8_image = torch_img_to_array(from_torch[i])

        if label is not None:
            highlight_text(uint8_image, f"{label:.2f}")
        if metakeys is not None:
            data = json.load(impath.with_suffix(".json").open("r"))
            for j, key in enumerate(metakeys):
                highlight_text(uint8_image, f"{data[key]:.2f}", level=1 + j)

        new_path = savedir.joinpath(impath.name)
        cv2.imwrite(str(new_path), uint8_image)
        print(f"Saved {new_path}")


def highlight_text(image, text, level=0):
    vstep = 50
    white = 255
    black = 0
    if len(image.shape) == 3 and image.shape[2] == 3:
        white = (255, 255, 255)
        black = (0, 0, 0)
    image[vstep * level : vstep * (level + 1), :100] = white
    cv2.putText(
        img=image,
        text=text,
        org=(10 + vstep * level, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=black,
        thickness=2,
    )


def torch_img_to_array(torch_img, sigma=3):
    # Convert torch to numpy
    float_image = torch_img.movedim(0, -1).detach().cpu().numpy()
    # If torch images are normalized, then we can
    # 1) scale that down to get a certain number of sigmas into -0.5 - 0.5
    # 2) add 0.5 so we're from 0 - 1
    # 3) clip so the high-sigma values are maxed at 0 and 1 instead of clipping
    if float_image.min() < 0:
        float_image = numpy.clip((float_image / (2 * sigma)) + 0.5, 0, 1)
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
    vis_torch_images=False,
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
        include_path=True,
    )

    print(f"Started visualizing at {datetime.now()}")
    count = 0
    for x, _, paths in loader:
        paths = map(Path, paths)
        x = x.to(device)
        output = model(x)
        if model.is_autoencoder:
            save_autoencoder_images(
                x=x,
                savedir=savedir,
                images=output,
                loss=nn.functional.mse_loss(x, output),
            )
        else:
            labels = output.detach().cpu().numpy().flatten()
            if vis_torch_images:
                save_debug_images(paths, savedir, labels=labels, from_torch=x)
            else:
                save_debug_images(paths, savedir, labels=labels)
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


def compilation_video(impaths):
    height, width, _ = cv2.imread(str(impaths[0])).shape
    size = (width, height)
    vid_path = impaths[0].parent.joinpath("compilation.mp4")
    out = cv2.VideoWriter(filename=str(vid_path), fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=1, frameSize=size)
    for impath in tqdm(impaths):
        out.write(cv2.imread(str(impath)))
    out.release()
    return vid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label images based on the results of a model for human"
        " visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Decide where we want to get our coverage values from",
        choices=["from_file", "from_model"],
        default="from_model",
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
        "-e",
        "--extension",
        help="Image extension WITH period (e.g. '.png')",
        default=".png",
    )
    parser.add_argument(
        "-r",
        "--regression-key",
        help="Value in the json file associated with the measurement",
        default="value",
    )
    parser.add_argument(
        "-v",
        "--video",
        help="Whether to turn output images into a compilation video",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--augmentation",
        help="[Only used w/ from_model] Path to image augmentation json file",
        type=Path,
        default=Path("./test_augmentations.json"),
    )
    parser.add_argument(
        "-k",
        "--wandb-keyfile",
        help="[Only used w/ from_model] Wandb login info in json dictionary, has the element key",
        type=Path,
        default=Path("/home/wandb.json"),
    )
    parser.add_argument(
        "-n",
        "--number",
        help="[Only used w/ from_model] Number of images to process",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-S",
        "--shuffle",
        help="[Only used w/ from_model] Shuffle images before selecting <number>",
        action="store_true",
    )
    parser.add_argument(
        "-V",
        "--vis-torch-images",
        help="[Only used w/ from_model] Label augmented images from torch loader, not originals",
        action="store_true",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-w",
        "--wandb-run-path",
        help="[Only used w/ from_model] [Mutually exclusive (1)] Wandb run"
        " (e.g. 'image-regression/wnevejfn' in config)",
    )
    group.add_argument(
        "-c",
        "--config-paths",
        help="[Only used w/ from_model] [Mutually exclusive (2)] Path to"
        " (.pth, .yaml) for the (model, config) files (space separated)",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()
    assert args.image_directory.is_dir()
    assert args.output_directory.is_dir()

    if args.source == "from_file":
        impaths = sorted(args.image_directory.glob(f"*{args.extension}"))
        labels = numpy.array(
            [
                json.load(impath.with_suffix(".json").open("r"))[args.regression_key]
                for impath in impaths
            ]
        )
        save_debug_images(
            impaths,
            savedir=args.output_directory,
            labels=labels,
        )
    elif args.source == "from_model":
        assert args.wandb_keyfile.is_file()
        # XOR
        # TODO: Remove this? Test that this is irrelevant with argparse
        assert bool(args.wandb_run_path is not None) ^ bool(
            args.config_paths is not None
        )
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
            vis_torch_images=args.vis_torch_images,
        )
    else:
        raise NotImplementedError(f"Source {args.source} not handled")

    if args.video:
        impaths = sorted(args.output_directory.glob(f"*{args.extension}"))
        print(f"Creating a video out of all images ({len(impaths)}) in {args.output_directory}")
        path = compilation_video(impaths=impaths)
        print(f"Saved {path}")
