"""プロジェクトのディレクトリ構成を共通で管理するためのモジュールです."""
# default
import logging
import pathlib
import traceback

# logger
logger = logging.getLogger(__name__)


def get_data() -> pathlib.Path:
    """データディレクトリのパスを返す.

    Returns:
        pathlib.Path: データディレクトリ.
    """
    return pathlib.Path("data")


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
    return pathlib.Path("data").joinpath("raw")


def _main() -> None:
    """実行確認用スクリプト."""
    logging.basicConfig(level=logging.INFO)

    # show directory path
    logger.info(f"data directory: {get_data()}")
    logger.info(f"raw directory: {get_raw()}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
