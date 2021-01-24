"""PyTorch Lightning用学習モジュール."""
# default packages
import dataclasses
import logging
import os
import pathlib
import sys

# third party packages
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch.nn.functional as F
import torch.optim as optim
import yaml

# my packages
import src.data.dataset as dataset
import src.models.net_cnn as net_cnn

# logger
_logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    """スクリプト実行用設定値."""

    experiment_name: str = "default"


class PlModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.network = net_cnn.Mnist()

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

    mlf_logger = pl_loggers.MLFlowLogger(
        experiment_name=config.experiment_name,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    model = PlModel()
    mnist = dataset.Mnist()
    trainer = pl.Trainer(logger=mlf_logger)
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
