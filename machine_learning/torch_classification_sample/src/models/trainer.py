"""分類器学習用スクリプト."""
# default
import dataclasses as dc
import logging
import os
import pathlib
import pprint
import sys
import tempfile
import typing as t

# third party packages
import mlflow
import mlflow.pytorch as mlf_pytorch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as pl_loggers

# my packages
import src.data.utils as ut
import src.data.dataset_food101 as dataset_food101

# logger
_logger = logging.getLogger(__name__)


@dc.dataclass
class Config:
    """実行用パラメータ."""

    network_name: str = "SimpleCBR"
    in_channels: int = 3
    out_channels: int = 3

    batch_size: int = 144
    num_workers: int = 4

    random_seed: int = 42

    cache_dir: str = "data/processed/SimpleCBR"
    save_top_k: int = 1
    save_weights_only: bool = False

    resume: bool = False

    early_stop: bool = False
    min_epochs: int = 30
    max_epochs: int = 500

    experiment_name: str = "Classifier"
    log_dir: str = "data/interim/SimpleCBR"
    use_gpu: bool = True
    progress_bar_refresh_rate: int = 1
    profiler: str = "simple"


class Trainer(pl.LightningModule):
    """分類器の学習クラス."""

    def __init__(self, network: nn.Module, **kwargs) -> None:
        super(Trainer, self).__init__()
        self.save_hyperparameters()

        self.network = network
        self.learning_rate = 1e-2

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.logger.experiment.log_metric(
            self.logger.run_id,
            "train_loss",
            loss.detach().cpu().item(),
            step=self.global_step,
        )

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.logger.experiment.log_metric(
            self.logger.run_id,
            "valid_loss",
            loss.detach().cpu().item(),
            step=self.global_step,
        )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [schedular]


def train(config: Config):
    """学習処理の実行スクリプト."""
    pl.seed_everything(config.random_seed)

    # 学習を途中から再開する場合などの設定
    cache_dir = pathlib.Path(config.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    trainer_params = dict()
    lastckpt = cache_dir.joinpath("last.ckpt")
    if config.resume:
        trainer_params["resume_from_checkpoint"] = str(lastckpt)
    elif lastckpt.exists():
        lastckpt.unlink()
    for filepath in cache_dir.glob("epoch*.ckpt"):
        filepath.unlink()

    # ログ設定
    pl_logger = pl_loggers.MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", None),
        tags={
            "mlflow.source.name": pathlib.Path(__file__).name,
            "mlflow.source.git.commit": ut.get_commit_id(),
        },
    )

    # ネットワーク、データセットの取得及び学習
    network = tv_models.vgg16(pretrained=False)
    params = dc.asdict(config)
    model = Trainer(network, **params)

    callbacks: t.List[t.Any] = list()
    model_checkpoint = pl_callbacks.ModelCheckpoint(
        filepath=str(cache_dir),
        monitor="val_loss",
        save_last=True,
        save_top_k=config.save_top_k,
        save_weights_only=config.save_weights_only,
        mode="min",
        period=1,
    )
    callbacks.append(model_checkpoint)
    if config.early_stop:
        callbacks.append(
            pl_callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=3,
                verbose=False,
                mode="auto",
            )
        )

    pl_trainer = pl.Trainer(
        default_root_dir=str(cache_dir),
        fast_dev_run=False,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        gpus=[0] if config.use_gpu and cuda.is_available() else None,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        profiler=config.profiler,
        callbacks=callbacks,
        logger=pl_logger,
        log_gpu_memory=True,
        **trainer_params,
    )
    datamodule = dataset_food101.Food101WithLableModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    pl_trainer.fit(model, datamodule)

    # ログに追加情報を設定
    mlf_client = mlflow.tracking.MlflowClient()
    for ckptfile in cache_dir.glob("epoch*.ckpt"):
        model = model.load_from_checkpoint(str(ckptfile), network, **params)
        with tempfile.TemporaryDirectory() as dname:
            mlf_model_path = pathlib.Path(dname).joinpath(ckptfile.stem)
            mlf_pytorch.save_model(model.network, mlf_model_path)
            mlf_client.log_artifact(pl_logger.run_id, mlf_model_path)


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
