import pathlib

import pandas as pd
import torch
from torch.utils.data import Dataset


class VwapDataset(Dataset):
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
