"""プロジェクトのディレクトリ構成を共通で管理するためのモジュールです."""
# default
import logging
import os
import pathlib

# my packages
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


def get_data() -> pathlib.Path:
    """データディレクトリのパスを返す.

    Returns:
        pathlib.Path: データディレクトリ.
    """
    data_dir = os.environ.get("DATA_DIR", "data")
    return pathlib.Path(data_dir)


def get_interim() -> pathlib.Path:
    """interim データディレクトリのパスを返す.

    Returns:
        pathlib.Path: interimデータディレクトリ.
    """
    return get_data().joinpath("interim")


def get_raw() -> pathlib.Path:
    """rawデータディレクトリのパスを返す.

    Returns:
        pathlib.Path: rawデータディレクトリ.
    """
    return get_data().joinpath("raw")


def main() -> None:
    """実行確認用スクリプト."""
    _logger.info(f"data directory: {get_data()}")
    _logger.info(f"raw directory: {get_raw()}")


if __name__ == "__main__":
    try:
        ut.init_root_logger()
        main()
    except Exception as e:
        logging.exception(e)
