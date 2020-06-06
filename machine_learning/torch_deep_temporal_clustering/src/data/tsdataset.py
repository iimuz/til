"""時系列データセット."""
# defualt packages
import logging
import pathlib

# third party packages
import numpy as np
import torch
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset

# logger
logger = logging.getLogger(__name__)


class TSDataset(Dataset):
    """時系列データセットクラス."""

    def __init__(self, filepath: pathlib.Path, transform=None) -> None:
        self.transform = transform
        self.x, self.y = load_from_tsfile_to_dataframe(str(filepath))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx):
        """データを取得する.
        Notes:
            データの次元: [Sequence, feature, 1]
        """
        index = idx.tolist() if torch.is_tensor(idx) else idx

        x = self.x.iloc[index, 0].to_numpy().astype(np.float32).reshape((1, 1, -1))
        y = np.float32(self.y[index])

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def _main() -> None:
    logging.basicConfig(level=logging.INFO)

    target = pathlib.Path("_data/raw/CBF/CBF_TRAIN.ts")
    dataset = TSDataset(target)
    logger.info(f"dataset length: {len(dataset)}")
    logger.info(f"value shape: {dataset[0][0].shape}, {dataset[0][1].shape}")


if __name__ == "__main__":
    _main()
