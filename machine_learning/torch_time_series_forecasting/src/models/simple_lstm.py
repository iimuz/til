"""LSTM を利用した簡易予測モデル."""
# default packages
import logging
import traceback

# thrid party packages
import torch.nn as nn
import torch.nn.functional as F

# logger
logger = logging.getLogger(__name__)


class SingleLSTM(nn.Module):
    """単層のLSTMアーキテクチャ."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super(SingleLSTM, self).__init__()

        self.hidden_size = 32

        self.lstm = nn.LSTM(
            input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.hidden_size, output_size, bias=True)

    def forward(self, x):
        code, _ = self.lstm(x)
        code = F.relu(code)
        code = self.fc(code)

        return code


def _main() -> None:
    """簡易実行テスト用スクリプト."""
    logging.basicConfig(level=logging.INFO)

    # show layer
    logger.info(SingleLSTM(input_size=8, output_size=2))


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
