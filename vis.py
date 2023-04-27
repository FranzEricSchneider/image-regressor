import matplotlib
matplotlib.use("Agg")

import cv2
from matplotlib import pyplot
import numpy
from pathlib import Path
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import wandb

from model import flattener


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
                        save_path = Path(f"/tmp/{prefix}_model{i}_testim{j}_layer{target}.jpg")
                        cam.generate_cam(input_image=x[j:j+1].to(device),
                                         target_class=0,
                                         save_path=save_path)
                        impaths.append(save_path)
                break

    wandb.log({impath.name: wandb.Image(str(impath)) for impath in impaths})


def scale_0_1(matrix):
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())


'''
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
'''
class CamExtractor():
    '''Extracts cam features from the model.'''
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        '''Does forward pass on convolutions, hooks function at given layer.'''
        conv_output = None
        # for module_pos, module in self.model.embedding._modules.items():
        for module_pos, module in enumerate(flattener(self.model.embedding)):
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        '''Does a full forward pass on the model, returns layer output.'''
        # Forward pass on the convolutions
        conv_output, _ = self.forward_pass_on_convolutions(x)
        return conv_output, self.model(x)


class ScoreCam():
    '''Saves class activation map.'''
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
            saliency_map = F.interpolate(saliency_map,
                                         size=input_size,
                                         mode="bilinear",
                                         align_corners=False)
            if saliency_map.max() == saliency_map.min():
                print(f"CONTINUING, {self.target_layer}")
                continue
            norm_saliency_map = scale_0_1(saliency_map)

            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image * norm_saliency_map)[1],
                          dim=1)[0][target_class]
            cam += w.data.detach().cpu().numpy() * \
                   target[i, :, :].data.detach().cpu().numpy()

        cam = numpy.maximum(cam, 0)
        cam = scale_0_1(cam)
        cam = Image.fromarray(cam).resize(input_size[::-1], Image.ANTIALIAS)
        cam = cv2.cvtColor(numpy.uint8(numpy.array(cam) * 255), cv2.COLOR_GRAY2RGB)

        original = (input_image[0].permute(1, 2, 0)).detach().cpu().numpy()
        original = numpy.uint8(original * 255)

        figure, axis = pyplot.subplots(figsize=(7, 7))
        axis.imshow(numpy.vstack([original, cam]))
        axis.axis("off")
        axis.set_title(f"Score-CAM layer {self.target_layer}")
        pyplot.tight_layout()
        pyplot.savefig(save_path, dpi=100)
        pyplot.close(figure)
