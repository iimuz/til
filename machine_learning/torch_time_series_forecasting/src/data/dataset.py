"""データセットをダウンロードするためのスクリプトです."""
# default packages
import logging
import pathlib
import traceback
import urllib.request as request

# third party
import pandas as pd
import tqdm as tqdm_std

# my packages
import src.data.directory as directory

# logger
logger = logging.getLogger(__name__)


class TqdmUpTo(tqdm_std.tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Args:
        tqdm (tqdm): tqdm
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """ update function

        Args:
            b (int, optional): Number of blocks transferred. Defaults to 1.
            bsize (int, optional): Size of each block (in tqdm units). Defaults to 1.
            tsize ([type], optional): Total size (in tqdm units). Defaults to None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_raw_filepath() -> pathlib.Path:
    url = get_raw_url()
    filepath = directory.get_raw().joinpath(url.split("/")[-1])
    return filepath


def get_raw_url() -> str:
    url = (
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        "jena_climate_2009_2016.csv.zip"
    )
    return url


def main() -> None:
    """メインの実行スクリプト."""
    logging.basicConfig(level=logging.INFO)

    filepath = get_raw_filepath()
    if filepath.exists() is False:
        url = get_raw_url()
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=filepath.name
        ) as pbar:
            request.urlretrieve(
                url, filename=filepath, reporthook=pbar.update_to, data=None
            )
    else:
        logger.info(f"data already exists: {filepath}")

    # show dataset description.
    df = pd.read_csv(filepath)
    logger.info(df.info())
    logger.info(df.head())
    logger.info(df.tail())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
