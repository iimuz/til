"""MVTecAd のデータセット."""
# default packages
import enum
import logging
import pathlib
import typing as t

# third party packages
import numpy as np
import PIL.Image as Image
import torch.utils.data as torch_data

# logger
logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    """データセットの出力モード."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Dataset(torch_data.Dataset):
    """MVTecAd  用データセット"""

    def __init__(
        self,
        filelist: t.List[pathlib.Path],
        transform: t.Optional[t.Any],
        mode: Mode = Mode.TRAIN,
    ) -> None:
        self.filelist = filelist
        self.transform = transform
        self.mode = mode

        self.type = self.mode.value

    def __getitem__(self, idx):
        img = Image.open(self.filelist[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img).transpose(0, 3, 1, 2)

        return img

    def __len__(self) -> int:
        return len(self.filelist)
