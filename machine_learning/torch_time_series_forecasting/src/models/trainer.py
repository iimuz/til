"""PyTorch Lightning を利用した学習用クラス."""
# default
import argparse
import logging
import math
import traceback

# third party
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data

# logger
logger = logging.getLogger(__name__)


class ForecastTrainer(pl.LightningModule):
    """予測モデルの学習を実行."""

    def __init__(self, network: nn.Module,) -> None:
        super(ForecastTrainer, self).__init__()

        self.network = network

        self.hparams = argparse.Namespace()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        prediction = self.forward(x)
        loss = self.criterion(prediction, y)

        if self.global_step % self.trainer.row_log_interval == 0:
            self.logger.experiment.add_figure(
                tag="train/overlap",
                figure=_create_graph(x, y, prediction),
                global_step=self.global_step,
            )
            plt.cla()
            plt.clf()
            plt.close()

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        prediction = self.forward(x)
        loss = self.criterion(prediction, y)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        # optimizer = optim.RMSprop(self.network.parameters())
        optimizer = optim.Adam(self.network.parameters())

        return optimizer

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


def _create_graph(x, y, prediction) -> None:
    x_cpu = x.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()
    prediction_cpu = prediction.detach().cpu().numpy()

    rows, cols = math.ceil(math.sqrt(x_cpu.shape[0])), x_cpu.shape[1]
    figsize = (6 * cols, 4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for batch in range(rows):
        for channel in range(cols):
            ax = axes[batch, channel]
            index_x = range(x_cpu.shape[2])
            index_y = range(index_x[-1], index_x[-1] + y_cpu.shape[2])
            ax.plot(index_x, x_cpu[batch, channel, :], color="b", label="input")
            ax.scatter(index_y, y_cpu[batch, channel, :], color="g", label="y")
            ax.scatter(
                index_y,
                prediction_cpu[batch, channel, :],
                color="orange",
                label="prediction",
            )
            ax.legend()

    return fig


def _main():
    """簡易実行テスト用スクリプト."""
    logging.basicConfig(level=logging.INFO)

    # show trainer
    logger.info(ForecastTrainer(nn.Module()))


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
