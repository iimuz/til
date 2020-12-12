"""小型の便利モジュール集."""
# default packages
import logging
import sys

# third party packages
import tqdm.autonotebook as tqdm

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


def init_root_logger(level: int = logging.INFO) -> None:
    """ルートロガーの状態を初期化する."""
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(_handler_stdout(level))


def _handler_stdout(level: int = logging.INFO) -> logging.Handler:
    """標準出力用ログハンドラ."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)

    return handler
