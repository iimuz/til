import argparse
import math
import pathlib
import pickle

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from torchvision.utils import make_grid

from simplecnnautoencoder import SimpleCNNAutoencoder
from vwapdataset import VwapDataset


class PyTMinMaxScalerVectorized:
    def __init__(self, min_val, max_val):
        self.min_val = torch.from_numpy(min_val)
        self.scale = torch.from_numpy(1.0 / (max_val - min_val))

    def __call__(self, tensor):
        result = (tensor - self.min_val) * self.scale
        result = result.float()
        return result


class Cnn3AEPl(pl.LightningModule):
    def __init__(self) -> None:
        super(Cnn3AEPl, self).__init__()
        self.batch_size = 64
        self.hparams = argparse.Namespace(channels=1, batch_size=self.batch_size)

        self.model = SimpleCNNAutoencoder(1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        output = self.forward(batch)
        loss = self.criterion(output, batch)

        if self.global_step % self.trainer.row_log_interval == 0:
            nrow = math.ceil(math.sqrt(self.batch_size))
            self.logger.experiment.add_image(
                tag="train/input",
                img_tensor=make_grid(batch, nrow=nrow, padding=0),
                global_step=self.global_step,
            )
            self.logger.experiment.add_image(
                tag="train/output",
                img_tensor=make_grid(output, nrow=nrow, padding=0),
                global_step=self.global_step,
            )

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
        return [optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)]

    @pl.data_loader
    def train_dataloader(self):
        with open("_data/interim/scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                PyTMinMaxScalerVectorized(sc.data_min_, sc.data_max_),
            ]
        )
        trainset = VwapDataset(
            pathlib.Path("_data/interim/train.pkl"), 64, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        with open("_data/interim/scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                PyTMinMaxScalerVectorized(sc.data_min_, sc.data_max_),
            ]
        )
        trainset = VwapDataset(
            pathlib.Path("_data/interim/train.pkl"), 64, transform=transform
        )
        loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        return loader


def _main():
    model = Cnn3AEPl()
    trainer = Trainer(
        early_stop_callback=True,
        default_save_path="_models/cnn3aepl",
        fast_dev_run=False,
        min_epochs=1,
        max_epochs=10,
        gpus=[0],
    )
    trainer.fit(model)


if __name__ == "__main__":
    _main()
