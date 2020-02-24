"""学習用処理."""
# default packages
import argparse
import logging
import math
import pathlib

# third party packags
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# my packages
import src.models.deep_temporal_clustering as dtc
from src.data.tsdataset import TSDataset

# logger
logger = logging.getLogger(__name__)


class DTCTrainer(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        train_path: pathlib.Path,
        valid_path: pathlib.Path,
        batch_size: int,
        workers: int,
    ) -> None:
        super(DTCTrainer, self).__init__()
        self.learning_rate = 1e-3
        self.sgd_momentum = 0.9

        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.workers = workers

        self.hparams = argparse.Namespace(
            learning_rate=self.learning_rate,
            sgd_momentum=self.sgd_momentum,
            train_path=str(self.train_path),
            valid_path=str(self.valid_path),
            batch_size=self.batch_size,
            workers=self.workers,
        )

        self.network = network
        self.criterion_ae = nn.MSELoss()
        self.criterion_cl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb, optimizer_idx):
        x, y = batch
        decode, q = self.forward(x)

        if optimizer_idx == 0:
            loss = self.criterion_ae(x, decode)
        else:
            p = dtc.target_distribution(q)
            loss = self.criterion_cl(q, p)

        if self.global_step % self.trainer.row_log_interval == 0:
            self.logger.experiment.add_figure(
                tag="train/overlap",
                figure=_create_graph(x, decode),
                global_step=self.global_step,
            )
            plt.cla()
            plt.clf()
            plt.close()

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        decode, p = self.forward(x)
        loss = self.criterion_ae(x, decode)

        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optim_ae = optim.SGD(
            self.network.autoencoder.parameters(),
            lr=self.learning_rate,
            momentum=self.sgd_momentum,
        )
        optim_cl = optim.SGD(
            self.network.clustering.parameters(),
            lr=self.learning_rate,
            momentum=self.sgd_momentum,
        )

        return [optim_ae, optim_cl]

    @pl.data_loader
    def train_dataloader(self):
        return _create_loader(self.train_path, self.batch_size, True, self.workers)

    @pl.data_loader
    def val_dataloader(self):
        return _create_loader(self.valid_path, self.batch_size, False, self.workers)


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
    filepath: pathlib.Path, batch_size: int, shuffle: bool, workers: int
) -> torch.utils.data.DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = TSDataset(filepath, transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers
    )

    return loader


def _main() -> None:
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    _main()
