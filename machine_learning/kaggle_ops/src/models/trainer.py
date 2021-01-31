"""PyTorch Lightning用学習モジュール."""
# default packages
import dataclasses
import logging
import os
import pathlib
import sys

# third party packages
import mlflow
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import torch.cuda as torch_cuda
import torch.nn.functional as F
import torch.optim as optim
import yaml

# my packages
import src.data.dataset as dataset
import src.models.network as network
import src.models.pl_utils as pl_utils

# logger
_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    """スクリプト実行用設定値."""

    pass


class PlModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = network.Mnist()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def main(config: Config) -> None:
    MLFLOW_TRACKING_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", "file:./data/processed/mlruns"
    )
    MLFLOW_ARTIFACT_LOCATION = os.environ.get(
        "MLFLOW_ARTIFACT_LOCATION", "file:./data/processed/mlruns/artifacts"
    )
    CHECKPOINT_DIR = os.environ.get(
        "CHECKPOINT_DIR", "data/processed/pl_model_checkpoint"
    )

    mlf_logger = pl_utils.MLFlowLogger(
        tracking_uri=MLFLOW_TRACKING_URI, artifact_location=MLFLOW_ARTIFACT_LOCATION
    )
    mlflow.log_params(dataclasses.asdict(config))

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=CHECKPOINT_DIR,
        filename="sample_{epoch:02d}",
    )

    model = PlModel()
    mnist = dataset.Mnist()
    trainer = pl.Trainer(
        min_epochs=1,
        max_epochs=5,
        logger=mlf_logger,
        callbacks=[checkpoint_callback],
        gpus=torch_cuda.device_count(),
    )
    trainer.fit(model, mnist)


def _load_config(filepath: pathlib.Path) -> Config:
    """設定ファイルを読み込む."""
    with open(str(filepath), "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    config = Config(**data)

    return config


if __name__ == "__main__":
    try:
        debug_mode = True if os.environ.get("MODE_DEBUG", "") == "True" else False
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)

        if len(sys.argv) == 2:
            _config = _load_config(pathlib.Path(sys.argv[1]))
        else:
            _config = Config()

        main(_config)
    except Exception as e:
        _logger.exception(e)
        sys.exit("Fail")
