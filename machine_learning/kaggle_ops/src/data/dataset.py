"""Dataset module."""
# default packages
import pathlib
import typing as t

# third party packages
import pytorch_lightning as pl
import torch.utils.data as torch_data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Mnist(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.path_raw = pathlib.Path("data/raw")
        self.batch_size = 64

    def train_dataloader(self, *args, **kwargs) -> torch_data.DataLoader:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist = datasets.MNIST(
            self.path_raw, train=True, download=True, transform=transform
        )
        return torch_data.DataLoader(mnist, batch_size=self.batch_size)

    def val_dataloader(
        self,
        *args,
        **kwargs,
    ) -> t.Union[torch_data.DataLoader, t.List[torch_data.DataLoader]]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        mnist = datasets.MNIST(
            self.path_raw, train=False, download=True, transform=transform
        )
        return torch_data.DataLoader(mnist, batch_size=self.batch_size)
