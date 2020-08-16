"""学習済みのモデルを Encoder に利用する VAE を提供するモジュール."""
# default package
import logging
import traceback
import typing as t


# third party packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchsummary as ts

# my packages
import src.data.utils as ut

# logger
logger = logging.getLogger(__name__)


class LBA(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        momentum: float = 0.1,
        use_activation: bool = True,
    ):
        super(LBA, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_features, out_features, bias),
            nn.BatchNorm1d(out_features, momentum=momentum),
        )
        if use_activation:
            self.conv.add_module("Act", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


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
            self.conv.add_module("Act", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class TransferVAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, image_size: t.Tuple[int, int]
    ) -> None:
        super(TransferVAE, self).__init__()

        self.channels = np.array([32, 64, 128, 256, 512])
        self.encoder_fc_channels = np.array([1536, 1024])

        self.image_size = image_size
        self.down_ratio = 2 ** 5
        self.encoded_size = (
            self.image_size[0] // self.down_ratio,
            self.image_size[1] // self.down_ratio,
        )
        self.encoded_dim = (
            self.encoded_size[0] * self.encoded_size[1] * self.channels[-1]
        )
        self.latent_dim = 512

        resnet = tv_models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*modules)
        self.encoder_fc = nn.Sequential(
            LBA(resnet.fc.in_features, self.encoder_fc_channels[0], momentum=1e-2),
            LBA(
                self.encoder_fc_channels[0], self.encoder_fc_channels[1], momentum=1e-2
            ),
        )
        self.fc_mean = nn.Linear(self.encoder_fc_channels[1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_fc_channels[1], self.latent_dim)

        self.decode_input = nn.Sequential(
            LBA(self.latent_dim, self.encoder_fc_channels[1]),
            LBA(self.encoder_fc_channels[1], self.encoder_fc_channels[0]),
            LBA(self.encoder_fc_channels[0], self.encoded_dim),
        )
        self.decoder = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[4], self.channels[3], (3, 3), padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[3], self.channels[2], (3, 3), padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[2], self.channels[1], (3, 3), padding=1),
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[1], self.channels[0], (3, 3), padding=1),
        )
        self.decode_final = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            CBA2d(self.channels[0], self.channels[0], (3, 3), padding=1),
            CBA2d(self.channels[0], out_channels, (3, 3), padding=1),
            nn.Tanh(),
        )

    def decode(self, z):
        decoded = self.decode_input(z)
        decoded = decoded.view(
            -1, self.channels[-1], self.encoded_size[0], self.encoded_size[1]
        )
        decoded = self.decoder(decoded)
        decoded = self.decode_final(decoded)

        return decoded

    def encode(self, x):
        code = self.encoder(x)
        code = code.view(x.shape[0], -1)
        code = self.encoder_fc(code)
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
        eps = torch.randn_like(std)
        z = mean + eps * std

        return z

    @classmethod
    def loss_function(cls, x, decode, mean, logvar):
        coef_kl_loss = 8e-4  # batch_size / (train or valid image_num) = 144 / 202600

        reconstruct_loss = F.mse_loss(decode, x)
        kl_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim=1), dim=0
        )
        loss = reconstruct_loss + coef_kl_loss * kl_loss

        return loss


def _main() -> None:
    ut.init_root_logger()

    network = TransferVAE(3, 3, (64, 64))
    network = network.to("cuda")
    ts.summary(network, input_size=(3, 64, 64))


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
