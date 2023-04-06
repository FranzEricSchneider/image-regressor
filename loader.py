import os
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, folder


def build_loader(data_path, batch_size, shuffle, extension="jpg"):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    def target_transform(target_path):
        with target_path.open("r") as file:
            return torch.Tensor([json.load(file)["coverage"]])

    dataset = DatasetFolder(
        str(data_path),
        loader=folder.default_loader,
        extensions=(extension, "json"),
        transform=transform,
        target_transform=target_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader
