"""vwap データセット."""
# default pcakges
import pathlib
from logging import getLogger

# thrid party packages
import pandas as pd
import torch
from torch.utils.data import Dataset

# logger
logger = getLogger(__name__)


class VwapDataset(Dataset):
    """vwapデータセットクラス."""

    def __init__(self, filepath: pathlib.Path, seq_len: int, transform=None):
        self.dataset = pd.read_pickle(filepath)
        self.sequence_length = seq_len
        self.transform = transform if transform is not None else None

    def __len__(self) -> int:
        return self.dataset.shape[0] - self.sequence_length

    def __getitem__(self, idx):
        index = idx.tolist() if torch.is_tensor(idx) else idx

        data = (
            self.dataset.iloc[index : index + self.sequence_length, :]
            .to_numpy()
            .reshape((1, -1, 1))
        )
        if self.transform is not None:
            data = self.transform(data)

        return data


def _main() -> None:
    """vwapデータセットの簡易確認用スクリプト."""
    import logging

    logging.basicConfig(level=logging.INFO)

    dataset = VwapDataset(
        filepath=pathlib.Path("_data/interim/dataset/train.pkl"), seq_len=64
    )
    for i in range(5):
        logger.info(f"value: {dataset[i]}")


if __name__ == "__main__":
    _main()
