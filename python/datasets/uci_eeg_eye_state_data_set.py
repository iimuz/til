import pathlib
import sys
from logging import getLogger
from urllib import request

# thrid party
import pandas as pd
from scipy.io import arff

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# logger
logger = getLogger(__name__)


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Args:
        tqdm (tqdm): tqdm
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        """ update function
        Args:
            b (int, optional): Number of blocks transferred. Defaults to 1.
            bsize (int, optional): Size of each block (in tqdm units). Defaults to 1.
            tsize ([type], optional): Total size (in tqdm units). Defaults to None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _main() -> None:
    """データセットのダウンロードを pandas.DataFrame への変換及び簡易表示を実行します。

    Note:
        - EEG Eye State Data Set: `http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State`
        - 目が開いているか閉じているかを判定するデータセットです。
        - 117秒間の連続データセットで、15次元の特徴量を持ちます。
        - eyeDetectionが0の時に目が開いており、1の時に目が閉じています。
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    DOWNLOAD_PATH = pathlib.Path("_data/EEG_Eye_State.arff")

    # EEG Eye State のダウンロード
    if DOWNLOAD_PATH.exists() is False:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=URL.split("/")[-1]
        ) as t:
            request.urlretrieve(
                URL, filename=str(DOWNLOAD_PATH), reporthook=t.update_to
            )

    # 読み込みと表示
    dataset, meta = arff.loadarff(str(DOWNLOAD_PATH))

    # 読み込みデータを pandas.DataFrame に変換
    df = pd.DataFrame(dataset, columns=meta.names())

    # display dataset
    logger.info("-" * 10 + " data " + "-" * 10)
    logger.info(df.head())

    logger.info("-" * 10 + " info " + "-" * 10)
    logger.info(df.info())


if __name__ == "__main__":
    _main()
