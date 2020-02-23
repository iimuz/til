import argparse
import math
import pathlib
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from vwapdataset import VwapDataset
from minmax_scaler import MinMaxScaler


class AETrainer(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        batch_size: int,
        scaler_path: str,
        train_path: str,
        validation_path: str,
        learning_rate: float,
        num_workers: int,
        hparams: Dict,
    ) -> None:
        super(AETrainer, self).__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sgd_momentum = 0.9
        self.num_workers = num_workers
        self.hparams = argparse.Namespace(
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            sgd_momentum=self.sgd_momentum,
            num_workers=self.num_workers,
            **hparams,
        )

        self.scaler_path = scaler_path
        self.train_path = train_path
        self.validation_path = validation_path

        self.model = model
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        output = self.forward(batch)
        loss = self.criterion(output, batch)

        if self.global_step % self.trainer.row_log_interval == 0:
            self.logger.experiment.add_figure(
                tag="train/overlap",
                figure=_create_graph(batch, output),
                global_step=self.global_step,
            )
            plt.cla()
            plt.clf()
            plt.close()

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.criterion(output, batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return [
            optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum
            )
        ]

    @pl.data_loader
    def train_dataloader(self):
        return _create_loader(
            scaler_path=self.scaler_path,
            dataset_path=self.train_path,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            shuffle=True,
            workers=self.num_workers,
        )

    @pl.data_loader
    def val_dataloader(self):
        return _create_loader(
            scaler_path=self.scaler_path,
            dataset_path=self.validation_path,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            shuffle=False,
            workers=self.num_workers,
        )


def _create_graph(data, output):
    data_cpu = data.detach().cpu().numpy().reshape((data.shape[0], -1))
    output_cpu = output.detach().cpu().numpy().reshape((data.shape[0], -1))

    fig = plt.figure(figsize=(30, 6))
    rows, cols = 1, math.ceil(math.sqrt(data_cpu.shape[0]))

    for idx in range(cols):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(data_cpu[idx, :], label="input")
        plt.plot(output_cpu[idx, :], label="reconstruct")

    return fig


def _create_loader(
    scaler_path: str,
    dataset_path: str,
    sequence_length: int,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> torch.utils.data.DataLoader:
    with open(scaler_path, "rb") as f:
        sc = pickle.load(f)
    transform = transforms.Compose(
        [transforms.ToTensor(), MinMaxScaler(sc.data_min_, sc.data_max_)]
    )
    trainset = VwapDataset(
        pathlib.Path(dataset_path), sequence_length, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=workers
    )

    return loader
