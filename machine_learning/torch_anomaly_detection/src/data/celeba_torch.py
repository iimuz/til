"""Celeba の PyTorch Dataset."""
# default packages
import logging
import typing as t

# third party packages
import numpy as np
import PIL.Image as Image
import torch.utils.data as td
import torchvision.transforms as tvt

# my packages
import src.data.dataset_torch as ds
import src.data.celeba as celeba

# logger
_logger = logging.getLogger(__name__)


class DatasetAE(td.Dataset):
    """Autoencoder 系用データセット."""

    def __init__(
        self, transforms: t.Optional[tvt.Compose] = None, mode: ds.Mode = ds.Mode.TRAIN
    ) -> None:
        self.dataset = celeba.Celeba().load()
        self.transforms = transforms

        self.mode = mode
        self.type = self.mode.value

        if self.mode == ds.Mode.TRAIN:
            self.datalist = self.dataset.train
        elif self.mode == ds.Mode.VALID:
            self.datalist = self.dataset.valid
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        filepath = self.dataset.path.joinpath(self.datalist["filepath"].iloc[idx])
        img = Image.open(filepath)

        if self.transforms is None:
            img = np.array(img).transpose(0, 3, 1, 2)
        else:
            img = self.transforms(img)

        return img

    def __len__(self) -> int:
        return self.datalist.shape[0]
