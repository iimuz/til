"""PyTorch 用 Jena Climate のデータセット."""
# default packages
import enum
import logging
import traceback

# third party packages
import numpy as np
import pandas as pd
import torch.utils.data as torch_data

# my packages
import src.data.dataset as dataset

# logger
logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    """データセットの出力モード."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Dataset(torch_data.Dataset):
    """Jena Climate のデータセット."""

    def __init__(
        self,
        data: np.ndarray,
        input_length: int = 2,
        forecast_length: int = 1,
        mode: Mode = Mode.TRAIN,
    ):
        self.data = data.astype(np.float32)
        self.input_length = input_length
        self.forecast_length = forecast_length
        self.mode = mode

        self.type = self.mode.value

    def __getitem__(self, idx):
        idx_end = idx + self.input_length
        data = self.data[idx:idx_end].T

        if self.mode == Mode.TEST:
            return data

        idx_forecast_end = idx_end + self.forecast_length
        forecast = self.data[idx_end:idx_forecast_end].T

        return data, forecast

    def __len__(self) -> int:
        length = self.data.shape[0] - self.input_length + 1
        length -= self.forecast_length if self.mode != Mode.TEST else 0

        return length


def _main() -> None:
    """動作確認用の実行スクリプト."""
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(dataset.get_raw_filepath())
    df = df[["p (mbar)", "T (degC)", "Tpot (K)"]]
    data = Dataset(df.to_numpy())
    data_loader = torch_data.DataLoader(
        data, batch_size=4, shuffle=False, num_workers=0, pin_memory=False
    )

    x, y = next(iter(data_loader))
    logger.info(x)
    logger.info(y)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
