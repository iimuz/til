"""log hander module."""
# default packages
import logging
import sys


def stdout_handler(level: int = logging.INFO) -> logging.Handler:
    """標準出力用ログハンドラ."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level=level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)

    return handler
