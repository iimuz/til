"""プロジェクトのディレクトリ構成を共通で管理するためのモジュールです."""
# default
import logging
import os
import pathlib
import traceback

# my packages
import src.data.log_utils as log_utils

# logger
logger = logging.getLogger(__name__)


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


def _main() -> None:
    """実行確認用スクリプト."""
    log_utils.init_root_logger()

    # show directory path
    logger.info(f"data directory: {get_data()}")
    logger.info(f"raw directory: {get_raw()}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
