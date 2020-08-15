"""単発の共通処理モジュール."""
# default packages
import errno
import logging
import os
import pathlib
import sys
import typing as t

# third party packages
import tqdm
import yaml

# logger
logger = logging.getLogger(__name__)

# type
T = t.TypeVar("T")


class TqdmUpTo(tqdm.tqdm):
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


def handler_stdout(level: int = logging.INFO) -> logging.Handler:
    """標準出力用ログハンドラ."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)

    return handler


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


def init_root_logger(level: int = logging.INFO) -> None:
    """ルートロガーの状態を初期化する."""
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler_stdout(level))
