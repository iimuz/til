"""LSTM を利用した簡易予測モデル."""
# default packages
import logging
import traceback

# thrid party packages
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# logger
logger = logging.getLogger(__name__)


class SingleLSTM(nn.Module):
    """単層のLSTMアーキテクチャ."""

    def __init__(self, input_feature: int, output_size: int) -> None:
        super(SingleLSTM, self).__init__()

        self.hidden_size = 32

        self.lstm = nn.LSTM(
            input_feature,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.hidden_size, output_size, bias=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        output = F.relu(h_n.view((x.shape[0], -1)))
        output = self.fc(output)

        return output


class DoubleLSTM(nn.Module):
    """2層のLSTMアーキテクチャ."""

    def __init__(self, input_feature: int, output_size: int) -> None:
        super(DoubleLSTM, self).__init__()

        self.hidden_sizes = np.array([32, 16])

        self.lstm1 = nn.LSTM(
            input_feature,
            hidden_size=self.hidden_sizes[0],
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            self.hidden_sizes[0],
            hidden_size=self.hidden_sizes[1],
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.hidden_sizes[1], output_size, bias=True)

    def forward(self, x):
        """Forward

        Notes:
            - x is (Batch, Sequence Length, Feature dims)
        """
        output, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(output)
        output = F.relu(h_n.view((x.shape[0], -1)))
        output = self.fc(output)

        return output


class LayeredDoubleLSTM(nn.Module):
    """複数の層を持つ、2層のLSTMアーキテクチャ."""

    def __init__(self, input_feature: int, output_size: int) -> None:
        super(LayeredDoubleLSTM, self).__init__()

        self.hidden_size = np.array([32, 16])
        self.layer_size = np.array([2, 2])

        self.lstm1 = nn.LSTM(
            input_feature,
            hidden_size=self.hidden_size[0],
            num_layers=self.layer_size[0],
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            self.hidden_size[0],
            hidden_size=self.hidden_size[1],
            num_layers=self.layer_size[1],
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.hidden_size[1], output_size, bias=True)

    def forward(self, x):
        """Forward

        Notes:
            - x is (Batch, Sequence Length, Feature dims)
        """
        output, _ = self.lstm1(x)
        _, (h_n, _) = self.lstm2(output)
        output = F.relu(h_n.view((x.shape[0], -1)))
        output = self.fc(output)

        return output


def _main() -> None:
    """簡易実行テスト用スクリプト."""
    logging.basicConfig(level=logging.INFO)

    # show layer
    logger.info(SingleLSTM(input_feature=8, output_size=2))


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
