import pathlib
import shutil

from logging import getLogger
from urllib import request

logger = getLogger(__name__)


def download_file(src: str, dst: str) -> None:
    """指定された URL から指定されたパスへファイルをダウンロードする

    Args:
        src (str): ダウンロードする URL
        dst (str): 保存するローカルパス
    """
    with request.urlopen(src) as res, open(str(dst), "wb") as f:
        shutil.copyfileobj(res, f)


def main() -> None:
    """XGBoost が用意しているデモ用データセットのダウンロードを行う
    """
    BASE_URL = "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/"
    DST_DIR = pathlib.Path("_data")
    FILE_LIST = ["agaricus.txt.test", "agaricus.txt.train"]

    for filename in FILE_LIST:
        src = BASE_URL + filename
        dst = DST_DIR.joinpath(filename)
        logger.info(f"download {dst} from {src}")
        download_file(src, str(dst))


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
