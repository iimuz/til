"""入出力系に対する共有処理を記述するモジュール."""
# default packages
import errno
import logging
import os
import pathlib
import typing as t

# third party packages
import yaml

# logger
logger = logging.getLogger(__name__)


# type
T = t.TypeVar("T")


def load_yaml(filepath: pathlib.Path, loader: t.Callable[[t.Dict], T]) -> T:
    """yamlファイルをロードし、指定した変換器を利用した結果を返す.

    Args:
        filepath (pathlib.Path): 読み込むファイルパス
        loader (t.Callable[[t.Dict], T]): 読み込んだ結果を辞書型から特定の型へ変換する変換器

    Raises:
        FileNotFoundError: 指定したファイルがない場合に発生する.

    Returns:
        T: 変換結果
    """
    if not filepath.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filepath)

    with open(str(filepath), "r") as f:
        result = yaml.load(f, Loader=yaml.SafeLoader)

    converted = loader(result)

    return converted
