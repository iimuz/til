"""PyTorch Lightning を利用した GAN の学習用モジュール."""
# default packages
import argparse
import dataclasses as dc
import enum
import logging
import pprint
import random
import shutil
import sys
import typing as t

# third party packages
import pytorch_lightning as pl
import pytorch_lightning.logging as pl_logging
import pytorch_lightning.callbacks as pl_callbacks
import torch
import torch.cuda as tc
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as td
import torchvision.utils as tv_utils
import torchvision.transforms as tv_transforms

# my packages
import src.data.celeba_torch as celeba_torch
import src.data.dataset_torch as ds_torch
import src.data.directories as directories
import src.data.mvtecad as mvtecad
import src.data.mvtecad_torch as mvtecad_torch
import src.data.utils as ut
import src.models.gan_vanila as gan_vanila

# logger
_logger = logging.getLogger(__name__)


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

        self.lr_g = 2e-4  # default 0.0002
        self.lr_d = 2e-4  # default 0.0002
        self.optim_betas_g = (0.5, 0.999)
        self.optim_betas_d = (0.5, 0.999)
        self.scheduler_gmma_g = 0.90  # default 0.95
        self.scheduler_gmma_d = 0.90  # default 0.95

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
            tensorboard_logs_g = {"train/loss_g": loss_g}
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
            tensorboard_logs_d = {"train/loss_d": loss_d}
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

        tensorboard_logs = {
            "train/epoch_loss_g": avg_loss_g,
            "train/epoch_loss_d": avg_loss_d,
        }
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

        tensorboard_logs = {
            "valid/epoch_loss_g": avg_loss_g,
            "valid/epoch_loss_d": avg_loss_d,
        }
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
        z = torch.randn(64, self.generator.latent_dim, device=self.last_imgs.device)
        sample_imgs = self(z)
        grid = tv_utils.make_grid(sample_imgs, nrow=8)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader_train

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader_valid

    def set_dataloader(self, train: td.DataLoader, valid: td.DataLoader) -> None:
        self.dataloader_train = train
        self.dataloader_valid = valid


@dc.dataclass
class Config:
    dataset_name: str = "CelebA"
    network_name: str = "Vanila"
    in_channels: int = 3
    out_channels: int = 1
    latent_dim: int = 62
    resize_image: t.Tuple[int, int] = (64, 64)

    batch_size: int = 144
    num_workers: int = 4

    random_seed: int = 42

    cache_dir: str = "gan_vanila_celeba"
    save_top_k: int = 2
    save_weights_only: bool = False

    experiment_version: int = 0
    resume: bool = False

    early_stop: bool = True
    min_epochs: int = 30
    max_epochs: int = 1000

    log_dir: str = "gan_train"
    use_gpu: bool = True
    progress_bar_refresh_rate: int = 1
    profiler: bool = True


class DatasetName(enum.Enum):
    """データセットを指定するための設定値."""

    CELEBA = "CelebA"
    MVTECAD_HAZELNUT = "MVTecAD_Hazelnut"

    @classmethod
    def value_of(cls, name: str) -> "DatasetName":
        """設定値の文字列から Enum 値を返す.

        Raises:
            ValueError: 指定した文字列が設定値にない場合

        Returns:
            [type]: Enum の値
        """
        for e in DatasetName:
            if e.value == name:
                return e

        raise ValueError(f"invalid value: {name}")


class NetworkName(enum.Enum):
    """ネットワークを指定するための設定値."""

    VANILA = "Vanila"

    @classmethod
    def value_of(cls, name: str) -> "NetworkName":
        """設定値の文字列から Enum 値を返す.

        Raises:
            ValueError: 指定した文字列が設定値にない場合

        Returns:
            [type]: Enum の値
        """
        for e in NetworkName:
            if e.value == name:
                return e

        raise ValueError(f"invalid value: {name}")


def get_dataset(
    name: DatasetName, transforms: tv_transforms.Compose, **kwargs
) -> t.Tuple[td.Dataset, td.Dataset]:
    if name == DatasetName.CELEBA:
        dataset_train = celeba_torch.DatasetAE(
            transforms=transforms, mode=ds_torch.Mode.TRAIN
        )
        dataset_valid = celeba_torch.DatasetAE(
            transforms=transforms, mode=ds_torch.Mode.VALID
        )
        return dataset_train, dataset_valid

    if name == DatasetName.MVTECAD_HAZELNUT:
        dataset_train = mvtecad_torch.DatasetAE(
            kind=mvtecad.Kind.HAZELNUT, transforms=transforms, mode=ds_torch.Mode.TRAIN
        )
        dataset_valid = mvtecad_torch.DatasetAE(
            kind=mvtecad.Kind.HAZELNUT, transforms=transforms, mode=ds_torch.Mode.VALID
        )
        return dataset_train, dataset_valid

    raise Exception(f"not implemented dataset: {name}")


def get_network(
    name: NetworkName, genenerator_params: t.Dict, discriminator_params: t.Dict
) -> t.Tuple[nn.Module, nn.Module]:
    if name == NetworkName.VANILA:
        return (
            gan_vanila.Generator(**genenerator_params),
            gan_vanila.Discriminator(**discriminator_params),
        )

    raise Exception(f"not implemented network: {name}")


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
            tv_transforms.Resize(image_size),
            tv_transforms.ToTensor(),
        ]
    )

    return transforms


def train(config: Config):
    """学習処理の実行スクリプト."""
    transforms = get_transforms(config.resize_image)
    dataset_type = DatasetName.value_of(config.dataset_name)
    dataset_train, dataset_valid = get_dataset(dataset_type, transforms)

    dataloader_train = td.DataLoader(
        dataset_train,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_random,
    )
    dataloader_valid = td.DataLoader(
        dataset_valid,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_random,
    )

    params = dc.asdict(config)
    pl.seed_everything(config.random_seed)

    generator, discriminator = get_network(
        NetworkName.value_of(config.network_name),
        genenerator_params={"latent_dim": config.latent_dim},
        discriminator_params={
            "in_channels": config.in_channels,
            "out_channels": config.out_channels,
        },
    )
    model = GANTrainer(generator, discriminator, params)
    model.set_dataloader(dataloader_train, dataloader_valid)

    cache_dir = directories.get_processed().joinpath(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    model_checkpoint = pl_callbacks.ModelCheckpoint(
        filepath=str(cache_dir),
        monitor="val_loss",
        save_last=True,
        save_top_k=config.save_top_k,
        save_weights_only=config.save_weights_only,
        mode="min",
        period=1,
    )

    experiment_dir = cache_dir.joinpath(
        "default", f"version_{config.experiment_version}"
    )
    pl_logger = pl_logging.TensorBoardLogger(
        save_dir=str(cache_dir), version=config.experiment_version
    )
    trainer_params = dict()
    if config.resume:
        trainer_params["resume_from_checkpoint"] = str(cache_dir.joinpath("last.ckpt"))
    elif experiment_dir.exists():
        shutil.rmtree(experiment_dir)
        for filepath in cache_dir.glob("*.ckpt"):
            filepath.unlink()
        for filepath in cache_dir.glob("*.pth"):
            filepath.unlink()

    pl_trainer = pl.Trainer(
        early_stop_callback=config.early_stop,
        default_root_dir=str(cache_dir),
        fast_dev_run=False,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        gpus=[0] if config.use_gpu and tc.is_available() else None,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        profiler=config.profiler,
        checkpoint_callback=model_checkpoint,
        logger=pl_logger,
        log_gpu_memory=True,
        **trainer_params,
    )
    pl_trainer.fit(model)

    for ckptfile in cache_dir.glob("*.ckpt"):
        model = model.load_from_checkpoint(
            str(ckptfile), generator, discriminator, params
        )

        pthfile = cache_dir.joinpath(ckptfile.stem + "_generator.pth")
        torch.save(model.generator.state_dict(), pthfile)
        pthfile = cache_dir.joinpath(ckptfile.stem + "_discriminator.pth")
        torch.save(model.discriminator.state_dict(), pthfile)


def _worker_init_random(worker_id: int) -> None:
    random.seed(worker_id)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)

        if len(sys.argv) == 1:
            _config = Config()
        elif len(sys.argv) == 2:
            _config = ut.load_yaml(sys.argv[1], lambda d: Config(**d))
        else:
            raise Exception(
                "input arguments error."
                " usage: python path/to/script.py"
                " or python path/to/script.py path/to/config.yml"
            )
        _logger.info(f"config: {pprint.pformat(dc.asdict(_config))}")

        train(_config)
    except Exception as e:
        _logger.exception(e)
