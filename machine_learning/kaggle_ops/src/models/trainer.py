"""PyTorch Lightning用学習モジュール."""
# default packages
import logging
import os
import sys

# third party packages
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim

# my packages
import src.data.dataset as dataset
import src.models.net_cnn as net_cnn

# logger
_logger = logging.getLogger(__name__)


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

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    model = PlModel()
    mnist = dataset.Mnist()
    trainer = pl.Trainer()
    trainer.fit(model, mnist)


if __name__ == "__main__":
    try:
        debug_mode = True if os.environ.get("MODE_DEBUG", "") == "True" else False
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
        sys.exit("Fail")
