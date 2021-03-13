"""NNIで探索するためのモデル定義モジュール."""
# third party packages
import torch
import torch.nn as nn


class FactorizedReduce(nn.Module):
    """factorized pointwiseを利用した特徴マップサイズの縮小 (stride=2)."""

    def __init__(self, c_in: int, c_out: int, affine: bool = True) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)

        return out


class StdConv(nn.Module):
    """Standard conv: ReLU - Conv - BN"""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        padding: int,
        affine: bool = True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(c_out, affine=affine),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cell(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        channels_pp: int,
        channels_p: int,
        channels: int,
        reduction_p: bool,
        reduction: bool,
    ) -> None:
        super().__init__()

        self.reduction = reduction
        self.n_nodes = n_nodes

        if reduction_p:
            self.preproc0 = FactorizedReduce(channels_pp, channels, affine=False)
        else:
            self.preproc0 = StdConv(
                channels_pp, channels, 1, stride=1, padding=0, affine=False
            )
        self.preproc1 = StdConv(
            channels_p, channels, 1, stride=1, padding=0, affine=False
        )


class CNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        in_channels: int,
        channels: int,
        n_classes: int,
        n_layers: int,
        n_nodes: int = 4,
        stem_multiplier: int = 3,
        auxiliary: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.aux_pos = 2 * n_layers // 3 if auxiliary else -1

        c_cur = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c_cur, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_cur),
        )

        reduce_and_double_channels_list = [n_layers // 3, 2 * n_layers // 3]
        channels_pp, channels_p, c_cur = c_cur, c_cur, channels
        self.cells = nn.ModuleList()
        reduction_p, reduction = False, False
        for i in range(n_layers):
            reduction_p, reduction = reduction, False
            if i in reduce_and_double_channels_list:
                c_cur *= 2
                reduction = True
