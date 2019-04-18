import pathlib

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size: int) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        "data/mnist", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def load_icons(path: pathlib.Path, batch_size: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data = datasets.ImageFolder(str(path), transform=transform)
    loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=4)

    return loader
