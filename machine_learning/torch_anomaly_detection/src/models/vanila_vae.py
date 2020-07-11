"""基本的な VAE を提供するモジュール.

Notes:
    - reference:
        `https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py`
"""
# default packages
import logging
import typing as t

# third party packages
import numpy as np
import torch
import torch.nn as nn

# logger
logger = logging.getLogger(__name__)


class CBA2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: t.Union[int, t.Tuple[int, int]],
        stride: t.Union[int, t.Tuple[int, int]] = 1,
        padding: t.Union[int, t.Tuple[int, int]] = 0,
        dilation: t.Union[int, t.Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "reflect",
        use_activation: bool = True,
    ):
        super(CBA2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if use_activation:
            self.conv.add_module("Act", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class VAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, image_size: t.Tuple[int, int]
    ) -> None:
        super(VAE, self).__init__()
        self.channels = np.array([64, 128, 256, 512])
        self.image_size = image_size
        self.down_ratio = 16
        self.encoded_size = (
            self.image_size[0] // self.down_ratio,
            self.image_size[1] // self.down_ratio,
        )
        self.encoded_dim = (
            self.encoded_size[0] * self.encoded_size[1] * self.channels[-1]
        )
        self.latent_dim = 512

        self.encoder = nn.Sequential(
            CBA2d(in_channels, self.channels[0], (5, 5), stride=2, padding=2),
            CBA2d(self.channels[0], self.channels[1], (5, 5), stride=2, padding=2),
            CBA2d(self.channels[1], self.channels[2], (5, 5), stride=2, padding=2),
            CBA2d(self.channels[2], self.channels[3], (5, 5), stride=2, padding=2),
        )
        self.fc_mean = nn.Linear(self.encoded_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoded_dim, self.latent_dim)

        self.fc_decode = nn.Linear(self.latent_dim, self.encoded_dim)
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[3], self.channels[2], (5, 5), padding=2),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[2], self.channels[1], (5, 5), padding=2),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[1], self.channels[0], (5, 5), padding=2),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[0], out_channels, (5, 5), padding=2),
            nn.Tanh(),
        )

    def decode(self, z):
        decoded = self.fc_decode(z)
        decoded = decoded.view(
            -1, self.channels[3], self.encoded_size[0], self.encoded_size[1]
        )
        decoded = self.decoder(decoded)

        return decoded

    def encode(self, x):
        code = self.encoder(x)
        code = code.view(-1, self.encoded_dim)
        mean = self.fc_mean(code)
        logvar = self.fc_logvar(code)

        return mean, logvar

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        decoded = self.decode(z)

        return decoded, mean, logvar

    def reparametrize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device="cuda:0")
        z = mean + eps * std

        return z


def loss_function(x, decode, mean, logvar):
    coef_kl_loss = 5e-3

    sse_loss = nn.BCELoss()(decode, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    loss = sse_loss + coef_kl_loss * kl_loss

    return loss
