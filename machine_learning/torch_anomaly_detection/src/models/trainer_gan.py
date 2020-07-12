"""PyTorch Lightning を利用した GAN の学習用モジュール."""
# default packages
import argparse
import logging
import typing as t

# third party packages
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as torch_data
import torchvision.utils as torchvision_utils

# logger
logger = logging.getLogger(__name__)


class GANTrainer(pl.LightningModule):
    """GAN 用の学習クラス."""

    def __init__(
        self, generator: nn.Module, discriminator: nn.Module, hparams: t.Dict = dict()
    ) -> None:
        super(GANTrainer, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.hparams = argparse.Namespace(**hparams)

        self.criterion_g = nn.BCELoss()
        self.criterion_d = nn.BCELoss()

        self.lr_g = 0.0002
        self.lr_d = 0.0002
        self.optim_betas_g = (0.5, 0.999)
        self.optim_betas_d = (0.5, 0.999)
        self.scheduler_gmma_g = 0.95
        self.scheduler_gmma_d = 0.95

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_nb, optimizer_idx):
        self.last_imgs = batch

        # generator
        if optimizer_idx == 0:
            z = torch.randn(
                batch.shape[0], self.generator.latent_dim, device=batch.device
            )
            self.generated_imgs = self(z)
            labels = torch.ones(batch.size(0), 1, device=batch.device)
            labels_hat = self.discriminator(self.generated_imgs)
            loss_g = self.criterion_g(labels_hat, labels)

            self.validation_logs = {"loss_g": loss_g}
            tensorboard_logs_g = {"loss_g": loss_g}
            return {
                "loss": loss_g,
                "loss_g": loss_g,
                "log": tensorboard_logs_g,
            }

        # discriminator
        if optimizer_idx == 1:
            valid = torch.ones(batch.size(0), 1, device=batch.device)  # all fake
            labels_hat = self.discriminator(batch)
            real_loss = self.criterion_d(labels_hat, valid)

            fake = torch.zeros(batch.size(0), 1, device=batch.device)  # all real
            labels_hat = self.discriminator(self.generated_imgs.detach())
            fake_loss = self.criterion_d(labels_hat, fake)

            loss_d = (real_loss + fake_loss) / 2
            tensorboard_logs_d = {"loss_d": loss_d}
            return {
                "loss": loss_d,
                "loss_d": loss_d,
                "log": tensorboard_logs_d,
                **self.validation_logs,
            }

    def training_epoch_end(self, outputs):
        avg_loss_g = torch.stack(
            [x["loss_g"] for x in outputs if "loss_g" in x.keys()]
        ).mean()
        avg_loss_d = torch.stack(
            [x["loss_d"] for x in outputs if "loss_d" in x.keys()]
        ).mean()

        tensorboard_logs = {"train_loss_g": avg_loss_g, "train_loss_d": avg_loss_d}
        return {
            "train_loss_g": avg_loss_g,
            "train_loss_d": avg_loss_d,
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_nb):
        # generator
        z = torch.randn(batch.shape[0], self.generator.latent_dim, device=batch.device)
        generated_val_imgs = self(z)
        labels = torch.ones(batch.size(0), 1, device=batch.device)
        labels_hat = self.discriminator(generated_val_imgs)
        loss_g = self.criterion_g(labels_hat, labels)

        # discriminator
        valid = torch.ones(batch.size(0), 1, device=batch.device)  # all fake
        labels_hat = self.discriminator(batch)
        real_loss = self.criterion_d(labels_hat, valid)

        fake = torch.ones(batch.size(0), 1, device=batch.device)  # all real
        labels_hat = self.discriminator(generated_val_imgs)
        fake_loss = self.criterion_d(labels_hat, fake)

        loss_d = (real_loss + fake_loss) / 2

        return {"val_loss": loss_d, "val_loss_g": loss_g, "val_loss_d": loss_d}

    def validation_epoch_end(self, outputs):
        avg_loss_g = torch.stack(
            [x["val_loss_g"] for x in outputs if "val_loss_g" in x.keys()]
        ).mean()
        avg_loss_d = torch.stack(
            [x["val_loss_d"] for x in outputs if "val_loss_d" in x.keys()]
        ).mean()

        tensorboard_logs = {"val_loss_g": avg_loss_g, "val_loss_d": avg_loss_d}
        return {
            "val_loss": avg_loss_d,
            "val_loss_g": avg_loss_g,
            "val_loss_d": avg_loss_d,
            "log": tensorboard_logs,
        }

    def configure_optimizers(self):
        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=self.optim_betas_g
        )
        optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas_d
        )
        scheduler_g = lr_scheduler.ExponentialLR(
            optimizer_g, gamma=self.scheduler_gmma_g
        )
        scheduler_d = lr_scheduler.ExponentialLR(
            optimizer_d, gamma=self.scheduler_gmma_d
        )

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

    def on_epoch_end(self) -> None:
        z = torch.randn(8, self.generator.latent_dim, device=self.last_imgs.device)
        sample_imgs = self(z)
        grid = torchvision_utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

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
