"""簡単な CNN を利用した Autoencoder."""
# default packages
import logging
import typing as t

# third party packages
import numpy as np
import torch.nn as nn

# logger
logger = logging.getLogger(__name__)


class CBR2d(nn.Module):
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
        super(CBR2d, self).__init__()
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
            self.conv.add_module("activation", nn.ReLU())

    def forward(self, x):
        return self.conv(x)


class Simple(nn.Module):
    """簡単なネットワーク.

    Notes:
        - 参考資料: `https://github.com/cheapthrillandwine/Improving_Unsupervised_Defect_Segmentation/blob/master/Improving_AutoEncoder_Samples.ipynb`
    """
    def __init__(self, in_channels: int, out_channels: int,) -> None:
        super(Simple, self).__init__()

        self.channels = np.array([32, 64, 128, 64, 32]) * 4

        self.encoder = nn.Sequential(
            CBR2d(in_channels, self.channels[0], (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            CBR2d(self.channels[0], self.channels[0], (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            CBR2d(self.channels[0], self.channels[0], (3, 3), padding=1),
            CBR2d(self.channels[0], self.channels[1], (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            CBR2d(self.channels[1], self.channels[1], (3, 3), padding=1),
            CBR2d(self.channels[1], self.channels[2], (3, 3), padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            CBR2d(
                self.channels[2],
                self.channels[2],
                (3, 3),
                padding=1,
                use_activation=False,
            ),
            # CBR2d(self.channels[2], self.channels[3], (3, 3), padding=1),
            # CBR2d(self.channels[3], self.channels[4], (3, 3), padding=1),
            # CBR2d(
            #     self.channels[4], out_channels, (3, 3), padding=1, use_activation=False
            # ),
        )
        self.decoder = nn.Sequential(
            # CBR2d(out_channels, self.channels[4], (3, 3), padding=1),
            # CBR2d(self.channels[4], self.channels[3], (3, 3), padding=1),
            # CBR2d(self.channels[3], self.channels[2], (3, 3), padding=1),
            CBR2d(self.channels[2], self.channels[2], (3, 3), padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            CBR2d(self.channels[2], self.channels[1], (3, 3), padding=1),
            CBR2d(self.channels[1], self.channels[1], (3, 3), padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            CBR2d(self.channels[1], self.channels[0], (3, 3), padding=1),
            CBR2d(self.channels[0], self.channels[0], (3, 3), padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            CBR2d(self.channels[0], self.channels[0], (3, 3), padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            CBR2d(
                self.channels[0], out_channels, (3, 3), padding=1, use_activation=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        code = self.encoder(x)
        decode = self.decoder(code)

        return decode
