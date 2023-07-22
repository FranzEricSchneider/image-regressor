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


numpy.set_printoptions(suppress=True, precision=4)


def read_rgb(impath):
    return cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2RGB)


def vis_model(models, config, loaders, device, prefixes, results):
    """
    Call a series of visualizations.

    Arguments:
        models: list of models, which will be called individually
        config: the keys (idx_vis_layers, idx_vis_layers, num_scorecam_images,
            is_autoencoder) are used to determine how many layers to visualize
        loaders: list of image loaders, e.g. the train and test loaders
        device: pytorch requirement, "cpu" or "cuda"
        prefixes: string prefix that should match the number of loaders, will
            be included in the filename for human readability
        results: list of dictionaries, each containing ("impaths", "outputs",
            and "losses") terms for each item in the loader. The list should
            correspond with loaders and prefixes. For example, it might be
            [{train loader results}, {test loader results}]. If an empty list
            is given, won't do this visualization.
    """

    impaths = []

    # Do ScoreCam visualization on certain layers
    if not config["is_autoencoder"]:
        impaths.extend(scorecam_vis(models, config, loaders, device, prefixes))

    # Visualize output results from worst to best performing
    impaths.extend(
        sorted_vis(
            results=results,
            prefixes=prefixes,
            key=config["regression_key"],
            num_sample=config["num_in_out_images"],
        )
    )

    wandb.log({impath.name: wandb.Image(str(impath)) for impath in impaths})


def scorecam_vis(models, config, loaders, device, prefixes):
    """
    Calls ScoreCam on various model layers, then returns those images.

    Arguments:
        models: list of models, which will be called individually
        config: the keys (idx_vis_layers, idx_vis_layers, num_scorecam_images,
            is_autoencoder) are used to determine how many layers to visualize
        loaders: list of image loaders, e.g. the train and test loaders
        device: pytorch requirement, "cpu" or "cuda"
        prefixes: string prefix that should match the number of loaders, will
            be included in the filename for human readability
    """
    impaths = []
    for i, model in enumerate(models):
        for loader, prefix in zip(loaders, prefixes):
            for x, y, _ in loader:
                if config["idx_vis_layers"] == "all":
                    indices = range(len([_ for _ in flattener(model.embedding)]))
                else:
                    indices = config["idx_vis_layers"]

                for target in indices:
                    cam = ScoreCam(model, target_layer=target)
                    for j in range(config["num_scorecam_images"]):
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
    return impaths


def sorted_vis(results, prefixes, key, num_sample):
    """
    TODO
    """

    impaths = []
    for result, prefix in zip(results, prefixes):
        if result is None:  # REMOVE
            continue        # REMOVE
        # Sample the desired indices (high to low loss)
        import ipdb; ipdb.set_trace()
        indices = numpy.argsort(result["losses"])

        # Then save the images
        impaths.extend(
            save_debug_images(
                impaths=[Path(result["impaths"][i] for i in indices)],
                savedir=Path("/tmp/"),
                labels=[result["outputs"][i] for i in indices],
                prefix=f"{prefix}_sorted-vis_",
                from_torch=None,
                metakeys=[key],
            )
        )

    return impaths


def scale_0_1(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


def save_autoencoder_images(x, savedir, images, prefix="debug", loss=None):

    impaths = []
    timestamp = str(int(time.time() * 1e6))

    for label, tensor, imloss in (("original", x, None), ("decoded", images, loss)):
        for i, torch_img in enumerate(tensor):
            name = f"{prefix}{timestamp}_{i}_{label}.jpg"
            uint8_image = torch_img_to_array(torch_img)
            if imloss is not None:
                imloss = f"{imloss:.2f}"
                print(f"Loss: {imloss}")
                highlight_text(uint8_image, imloss)
                name.replace(".jpg", f"_{imloss}.jpg")
            impaths.append(savedir.joinpath(name))
            cv2.imwrite(str(impaths[-1]), uint8_image)

    return impaths


def save_debug_images(
    impaths,
    savedir,
    labels=None,
    prefix="debug",
    from_torch=None,
    metakeys=None,
    sortkey=None,
):

    if labels is None:
        labels = [None] * len(x)

    if sortkey is not None:
        order = numpy.argsort(
            [
                json.load(impath.with_suffix(".json").open("r")).get(sortkey, 0)
                for impath in impaths
            ]
        )
    else:
        order = list(range(len(impaths)))

    new_impaths = []
    for i, (impath, label) in enumerate(zip(impaths, labels)):

        uint8_image = cv2.imread(str(impath))
        if from_torch is not None:
            uint8_image = numpy.hstack((uint8_image, torch_img_to_array(from_torch[i])))

        if label is not None:
            highlight_text(uint8_image, f"{label:.2f}")
        if metakeys is not None:
            data = json.load(impath.with_suffix(".json").open("r"))
            for j, key in enumerate(metakeys):
                highlight_text(
                    uint8_image, f"{key}: {data.get(key, 'None')}", level=1 + j
                )

        if sortkey is None:
            name = prefix + impath.name
        else:
            name = f"{prefix}{numpy.where(order == i)[0][0]:04}_{impath.name}"
        new_impaths.append(savedir.joinpath(name))
        cv2.imwrite(str(new_impaths[-1]), uint8_image)

    return new_impaths


def highlight_text(image, text, level=0):
    vstep = 50
    white = 255
    black = 0
    if len(image.shape) == 3 and image.shape[2] == 3:
        white = (255, 255, 255)
        black = (0, 0, 0)
    image[vstep * level : vstep * (level + 1), : 18 * len(text) + 18] = white
    cv2.putText(
        img=image,
        text=text,
        org=(10, 30 + vstep * level),
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
    metakeys,
):
    """
    1) Load model according to arguments
    2) Load the image loaders according to arguments
    3) Save visualized images, according to the model type (autoencoder or
       regression)

    Arguments:
        imdir: pathlib.Path where we load images to process from
        savedir: pathlib.Path where output images are saved
        run_path: string for a wandb run, e.g. "image-regression/3q34k58v"
        wandb_paths: two-element list of a .pth and .yaml file
        augmentation: pathlib.Path for augmentation file for the loader
        extension: suffix like ".png" or ".jpg" for the loader
        number: how many images to process (for speed reasons)
        key: (string) the key in the json files for the GT value
        shuffle: (bool) whether to shuffle the loader
        keyfile: pathlib.Path for our wandb login file
        metakeys: other values in the json file we want to include in the
            visualization (e.g. writing on the image)
    """

    # Needed before we can restore the models
    login_wandb({"keyfile": keyfile})

    num_cpus, device = system_check()
    if run_path is not None:
        model = {"name": "checkpoint.pth", "run_path": run_path, "replace": True}
        config_paths = None
    elif wandb_paths is not None:
        # Should be given as a tuple, where each is a Path() to a .pth or .yaml
        # file we want to load
        model, config_path = wandb_paths
        config_paths = [config_path]
    else:
        raise ValueError(f"Invalid paths given: {run_path}, {config_paths}")

    models = get_models(
        config={
            "models": [model],
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
    for x, value, paths in loader:
        # Enforce a rule of thumb size limit (expand if we want later)
        assert all(
            numpy.array(x.shape[2:]) < 1500
        ), f"Images too large ({x.shape[2:]}), were they downsampled?"

        paths = [Path(p) for p in paths]
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
            if metakeys is None:
                metakeys = [key]
            else:
                metakeys = sorted(set([key] + metakeys))
            save_debug_images(
                impaths=paths,
                savedir=savedir,
                labels=labels,
                metakeys=metakeys,
                from_torch=x,
            )
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
    out = cv2.VideoWriter(
        filename=str(vid_path),
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=1,
        frameSize=size,
    )
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
        "-E",
        "--extra-labels",
        help="Extra fields in the json file we want to write on the images",
        nargs="+",
    )
    parser.add_argument(
        "-r",
        "--regression-key",
        help="Value in the json file associated with the measurement",
        default="value",
    )
    parser.add_argument(
        "-t",
        "--sort-key",
        help="json key we want to sort images by - DOESN'T CURRENTLY WORK FOR FROM_MODEL",
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
            metakeys=args.extra_labels,
            sortkey=args.sort_key,
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
            metakeys=args.extra_labels,
        )
    else:
        raise NotImplementedError(f"Source {args.source} not handled")

    if args.video:
        impaths = sorted(args.output_directory.glob(f"*{args.extension}"))
        print(
            f"Creating a video out of all images ({len(impaths)}) in {args.output_directory}"
        )
        path = compilation_video(impaths=impaths)
        print(f"Saved {path}")
