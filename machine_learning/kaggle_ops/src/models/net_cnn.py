"""テスト用の簡易ネットワークの定義."""
# default packages
import logging
import os
import sys

# third party packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

# logger
_logger = logging.getLogger(__name__)


class Mnist(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nbatch, channels, width, height = x.shape

        v = x.view(nbatch, -1)
        v = self.layer(v)

        prob = F.log_softmax(v, dim=1)
        return prob


def main() -> None:
    """ネットワーク構成を確認するためのスクリプト."""
    torchinfo.summary(Mnist())


if __name__ == "__main__":
    try:
        debug_mode = True if os.environ.get("MODE_DEBUG", "") == "True" else False
        logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
        sys.exit("Fail")
