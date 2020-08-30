"""Autoencoder 系の学習用スクリプト."""
# default
import argparse
import dataclasses as dc
import enum
import logging
import os
import pathlib
import pprint
import sys
import tempfile
import typing as t

# third party
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch as mlf_pytorch
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
import torch.cuda as tc
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# my packages
import src.data.celeba_lightning as celeba_lightning
import src.data.mvtecad as mvtecad
import src.data.mvtecad_lightning as mvtech_lighning
import src.data.directories as directories
import src.data.utils as ut
import src.models.ae_cnn as ae_cnn

# logger
_logger = logging.getLogger(__name__)


class AETrainer(pl.LightningModule):
    """Autoencoder 用の学習クラス."""

    def __init__(self, network: nn.Module, record_params: t.Dict = dict()) -> None:
        super(AETrainer, self).__init__()

        self.network = network
        self.hparams = argparse.Namespace(**record_params)
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-2
        self.sgd_momentum = 0.9

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_nb):
        decode = self.forward(batch)
        loss = self.criterion(decode, batch)

        if batch_nb % 100 == 0:
            num_display = 4
            _save_artifact_image(
                batch[:num_display].detach().cpu().numpy(),
                decode[:num_display].detach().cpu().numpy(),
                filename=f"train_{self.global_step:04}.png",
                mlflog=self.logger,
            )

        tensorboard_logs = {"train/loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        tensorboard_logs = {"train/epoch_loss": avg_loss}
        return {"train_loss": avg_loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        decode = self.forward(batch)
        loss = self.criterion(decode, batch)

        if batch_nb % 100 == 0:
            num_display = 4
            _save_artifact_image(
                batch[:num_display].detach().cpu().numpy(),
                decode[:num_display].detach().cpu().numpy(),
                filename=f"valid_{self.global_step:04}.png",
                mlflog=self.logger,
            )

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        tensorboard_logs = {"valid/epoch_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

        return [optimizer], [scheduler]


@dc.dataclass
class Config:
    dataset_name: str = "MVTecAD_Hazelnut"
    network_name: str = "SimpleCBR"
    in_channels: int = 3
    out_channels: int = 3
    resize_image: t.Tuple[int, int] = (128, 128)

    batch_size: int = 144
    num_workers: int = 4

    random_seed: int = 42

    cache_dir: str = "simple_cbr_mvtecad_hazelnut"
    save_top_k: int = 1
    save_weights_only: bool = False

    resume: bool = False

    early_stop: bool = True
    min_epochs: int = 30
    max_epochs: int = 1000

    log_dir: str = "ae_train"
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

    SIMPLE_CBR = "SimpleCBR"
    SIMPLE_CR = "SimpleCR"

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


def get_datamodule(
    name: DatasetName, image_size: t.Tuple[int, int], batch_size: int, num_workers: int,
) -> pl.LightningDataModule:
    if name == DatasetName.CELEBA:
        return celeba_lightning.DataModuleAE(
            image_size=image_size, batch_size=batch_size, num_workers=num_workers
        )

    if name == DatasetName.MVTECAD_HAZELNUT:
        return mvtech_lighning.DataModuleAE(
            kind=mvtecad.Kind.HAZELNUT,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    raise ValueError(f"not implemented dataset: {name}")


def get_network(name: NetworkName, **kwargs) -> nn.Module:
    if name == NetworkName.SIMPLE_CBR:
        return ae_cnn.SimpleCBR(**kwargs)

    if name == NetworkName.SIMPLE_CR:
        return ae_cnn.SimpleCR(**kwargs)

    raise Exception(f"not implemented network: {name}")


def train(config: Config):
    """学習処理の実行スクリプト."""
    params = dc.asdict(config)
    pl.seed_everything(config.random_seed)

    # 中間データの保存設定
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

    # ログ設定
    pl_logger = pl_loggers.MLFlowLogger(
        experiment_name="example",
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", None),
        tags={
            "mlflow.source.name": pathlib.Path(__file__).name,
            "mlflow.source.git.commit": ut.get_commit_id(),
        },
    )

    # 学習を途中から再開する場合などの設定
    trainer_params = dict()
    if config.resume:
        trainer_params["resume_from_checkpoint"] = str(cache_dir.joinpath("last.ckpt"))
    for filepath in cache_dir.glob("epoch*.ckpt"):
        filepath.unlink()

    # ネットワーク、データセットの取得及び学習
    network = get_network(
        NetworkName.value_of(config.network_name),
        in_channels=config.in_channels,
        out_channels=config.out_channels,
    )
    model = AETrainer(network, params)
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
    datamodule = get_datamodule(
        DatasetName.value_of(config.dataset_name),
        image_size=config.resize_image,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    pl_trainer.fit(model, datamodule)

    # ログに追加情報を設定
    mlf_client = mlflow.tracking.MlflowClient()
    for key, val in pl_trainer.profiler.recorded_durations.items():
        for idx, v in enumerate(val):
            mlf_client.log_metric(pl_logger.run_id, key, v, step=idx)
        mlf_client.log_metric(pl_logger.run_id, f"{key}_mean", np.mean(val))
        mlf_client.log_metric(pl_logger.run_id, f"{key}_sum", np.sum(val))
    for ckptfile in cache_dir.glob("epoch*.ckpt"):
        model = model.load_from_checkpoint(str(ckptfile), network, params)
        with tempfile.TemporaryDirectory() as dname:
            mlf_model_path = pathlib.Path(dname).joinpath(ckptfile.stem)
            mlf_pytorch.save_model(model.network, mlf_model_path)
            mlf_client.log_artifact(pl_logger.run_id, mlf_model_path)


def _save_artifact_image(
    batch: np.ndarray,
    decode: np.ndarray,
    filename: str,
    mlflog: pl_loggers.MLFlowLogger,
) -> None:
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

    with tempfile.TemporaryDirectory() as dname:
        filepath = pathlib.Path(dname).joinpath(filename)
        fig.savefig(filepath, bbox_inches="tight", pad_inches=0)
        fig.clf()
        plt.close()
        mlflog.experiment.log_artifact(mlflog.run_id, filepath)


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
