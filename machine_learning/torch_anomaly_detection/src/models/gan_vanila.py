"""基本的な GAN を提供するモジュール.

Notes:
    参考資料: `http://aidiary.hatenablog.com/entry/20180311/1520762446`
"""
# default packages
import logging
import typing as t

# third party pacakges
import numpy as np
import torch.nn as nn
from torch.nn.modules import padding

# logger
logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """Generator."""

    def __init__(self, latent_dim: int) -> None:
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, (3, 3), stride=1, padding=1, padding_mode="reflect"),
            nn.Sigmoid(),
        )

    def forward(self, x):
        code = self.encoder(x)
        code = code.view(-1, 128, 16, 16)
        decode = self.decoder(code)

        return decode


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Discriminator, self).__init__()

        self.channels = np.array([64, 128])

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                self.channels[0],
                self.channels[1],
                3,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(self.channels[1]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        result = self.conv(x)
        result = result.view(-1, 128 * 16 * 16)
        result = self.fc(result)

        return result
