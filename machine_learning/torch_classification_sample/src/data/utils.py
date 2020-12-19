"""小型の便利モジュール集."""
# default packages
import contextlib
import errno
import logging
import os
import pathlib
import random
import sys
import subprocess
import time
import typing as t

# third party packages
import tqdm.autonotebook as tqdm
import yaml

# type
T = t.TypeVar("T")

# logger
_logger = logging.getLogger(__name__)


class TqdmUpTo(tqdm.tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Args:
        tqdm (tqdm): tqdm
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """update function

        Args:
            b (int, optional): Number of blocks transferred. Defaults to 1.
            bsize (int, optional): Size of each block (in tqdm units). Defaults to 1.
            tsize ([type], optional): Total size (in tqdm units). Defaults to None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_commit_id() -> str:
    """Get current git commit hash.

    Returns:
        str: git commit hash.
    """
    cmd = "git rev-parse --short HEAD"
    commid_id = subprocess.check_output(cmd.split()).strip().decode("utf-8")

    return commid_id


def init_root_logger(level: int = logging.INFO) -> None:
    """ルートロガーの状態を初期化する."""
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(_handler_stdout(level))


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


@contextlib.contextmanager
def timer(name: str, print_log: t.Any = print) -> None:
    start = time.time()
    yield
    end = time.time()

    print_log("{name}: {time:.3f} sec".format(name=name, time=end - start))


def worker_init_random(worker_id: int) -> None:
    random.seed(worker_id)


def _handler_stdout(level: int = logging.INFO) -> logging.Handler:
    """標準出力用ログハンドラ."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)

    return handler
