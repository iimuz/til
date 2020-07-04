"""Dilated CNN."""
# default packages
import logging
import traceback

# third party packages
import torch
import torch.nn as nn
import torch.nn.functional as F

# logger
logger = logging.getLogger(__name__)


class CausalConv1d(nn.Conv1d):
    """Causal Conv 1D.

    Notes:
        - `https://github.com/pytorch/pytorch/issues/1333#issuecomment-453702879`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class CR1(nn.Module):
    """CausalConv1D + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super(CR1, self).__init__()

        self.causal_conv = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        out = self.causal_conv(x)
        out = F.relu(out)

        return out


class DCNN(nn.Module):
    """Dilated CNN.

    Notes:
        - `https://github.com/microsoft/forecasting/blob/master/fclib/fclib/models/dilated_cnn.py`
    """

    def __init__(
        self, input_feature: int, output_size: int, dilated_layers: int
    ) -> None:
        super(DCNN, self).__init__()

        seq_len = 128
        kernel_size = 2
        self.causal_conv0 = CR1(input_feature, 32, kernel_size, dilation=1,)
        self.causal_conv_seq = nn.Sequential(
            *[
                CR1(32, 32, kernel_size, dilation=2 ** i,)
                for i in range(1, dilated_layers)
            ]
        )
        self.conv_out = nn.Conv1d(32 * 2, 8, 1)
        self.fc = nn.Linear(8 * seq_len, output_size, bias=True)

    def forward(self, x):
        view = x.permute(0, 2, 1)

        out_first = self.causal_conv0(view)
        out_end = self.causal_conv_seq(out_first)

        out = torch.cat([out_first, out_end], dim=1)
        out = self.conv_out(out)
        out = self.fc(torch.flatten(out, start_dim=1))

        return out


def _main() -> None:
    """簡易実行のテスト用スクリプト."""
    logging.basicConfig(level=logging.INFO)

    logger.info("dilated cnn")
    logger.info(DCNN(input_feature=8, output_size=2, dilated_layers=4))


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
