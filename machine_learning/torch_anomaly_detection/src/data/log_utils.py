"""ログ関連のモジュール."""
# default packages
import logging
import sys

# logger
logger = logging.getLogger(__name__)


def handler_stdout(level: int = logging.INFO) -> logging.Handler:
    """標準出力用ログハンドラ."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)

    return handler


def init_root_logger(level: int = logging.INFO) -> None:
    """ルートロガーの状態を初期化する."""
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler_stdout(level))
