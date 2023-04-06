import gc
import json
import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, folder


def build_loader(data_path, batch_size, shuffle, key, extension="jpg"):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    def target_transform(target_path):
        with target_path.open("r") as file:
            return torch.Tensor([json.load(file)[key]])

    dataset = DatasetFolder(
        str(data_path),
        loader=folder.default_loader,
        extensions=(extension, "json"),
        transform=transform,
        target_transform=target_transform,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader


def get_loaders(config, debug=False):

    print("Loading data...")
    gc.collect()

    train_loader = val_loader = None
    if config["train"]:
        train_loader = build_loader(
            data_path=config["data_dir"].joinpath("train"),
            batch_size=config["batch_size"],
            shuffle=True,
            key=config["regression_key"],
            extension=config["extension"],
        )
    test_loader = build_loader(
        data_path=config["data_dir"].joinpath("test"),
        batch_size=config["batch_size"],
        shuffle=False,
        key=config["regression_key"],
        extension=config["extension"],
    )

    if debug:
        print()
        print("=" * 80)
        print("Batch size: ", config["batch_size"])
        print(f"Train batches = {len(train_loader)}")
        print(f"Test batches = {len(test_loader)}")
        import ipdb; ipdb.set_trace()
        for name, loader in (("TRAIN", train_loader),
                             ("TEST", test_loader)):
            for x, y, lx, ly in loader:
                print(name, x.shape, y.shape, lx.shape, ly.shape)
                break
        print("=" * 80)

    return train_loader, val_loader, test_loader
