"""Autoencoder 系の学習用スクリプト."""
# default
import argparse
import dataclassses as dc
import logging
import traceback
import typing as t

# third party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as torch_data
import torchvision.transforms as tv_transforms

# my packages
import src.data.dataset as dataset
import src.data.directories as directories
import src.data.log_utils as lu

# logger
logger = logging.getLogger(__name__)


class AETrainer(pl.LightningModule):
    """Autoencoder 用の学習クラス."""

    def __init__(self, network: nn.Module, hparams: t.Dict = dict()) -> None:
        super(AETrainer, self).__init__()

        self.network = network
        self.hparams = argparse.Namespace(**hparams)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-2
        self.sgd_momentum = 0.9

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):
        decode = self.forward(batch)
        loss = self.criterion(decode, batch)

        tensorboard_logs = {"loss": loss}

        num_display = 4
        img_logs = {
            "batch": batch[:num_display].detach().cpu().numpy(),
            "decode": decode[:num_display].detach().cpu().numpy(),
            "global_step": self.global_step,
        }

        return {"loss": loss, "log": tensorboard_logs, "img": img_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        img_logs = outputs[0]["img"]
        self.logger.experiment.add_figure(
            tag="train/overlap",
            figure=_create_graph(img_logs["batch"], img_logs["decode"]),
            global_step=img_logs["global_step"],
        )
        plt.cla()
        plt.clf()
        plt.close()

        tensorboard_logs = {"train_loss": avg_loss}
        return {"train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        decode = self.forward(batch)
        loss = self.criterion(decode, batch)

        num_display = 4
        img_logs = {
            "batch": batch[:num_display].detach().cpu().numpy(),
            "decode": decode[:num_display].detach().cpu().numpy(),
            "global_step": self.global_step,
        }

        return {"val_loss": loss, "img": img_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        img_logs = outputs[0]["img"]
        self.logger.experiment.add_figure(
            tag="valid/overlap",
            figure=_create_graph(img_logs["batch"], img_logs["decode"]),
            global_step=img_logs["global_step"],
        )
        plt.cla()
        plt.clf()
        plt.close()

        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader_train

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader_valid

    def set_dataloader(
        self, train: torch_data.DataLoader, valid: torch_data.DataLoader
    ) -> None:
        self.dataloader_train = train
        self.dataloader_valid = valid


def _create_graph(batch: np.ndarray, decode: np.ndarray) -> None:
    batch = batch.transpose(0, 2, 3, 1)
    decode = decode.transpose(0, 2, 3, 1)

    params_imshow = dict()
    if batch.shape[3] == 1:
        batch = batch.reshape(batch.shape[:3])
        decode = decode.reshape(decode.shape[:3])
        params_imshow["cmap"] = "gray"

    rows, cols = 2, batch.shape[0]
    figsize = (4 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for idx in range(cols):
        ax = axes[0, idx]
        ax.imshow(batch[idx], **params_imshow)

        ax = axes[1, idx]
        ax.imshow(decode[idx], **params_imshow)

    return fig


@dc.dataclass
class Config:
    train_list_path: str = "path/to/train/list.csv"
    valid_list_path: str = "path/to/valid/list.csv"

    image_size: t.Tuple[int, int] = (64, 64)


def get_transforms(image_size: t.Tuple[int, int]) -> tv_transforms.Compose:
    """データ変換コンポーネントの取得.

    Args:
        image_size (Tuple[int, int]): 最終的に取得する画像サイズ.

    Returns:
        torchvision.transforms.Compose: データ変換コンポーネント
    """
    transforms = tv_transforms.Compose(
        [
            tv_transforms.RandomHorizontalFlip(),
            tv_transforms.CenterCrop(148),
            tv_transforms.Resize(image_size),
            tv_transforms.ToTensor(),
        ]
    )

    return transforms


def train(config: Config):
    """学習処理の実行スクリプト."""
    transforms = get_transforms(config.image_size)
    dataset_train = dataset.ImageDataset(
        config.train_list_path, transforms, dataset.Mode.TRAIN
    )
    dataset_valid = dataset.ImageDataset(
        config.valid_list_path, transforms, dataset.Mode.VALID
    )


if __name__ == "__main__":
    try:
        lu.init_root_logger()
        train()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
