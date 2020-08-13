"""単発の共通処理モジュール."""
# default packages
import logging
import pathlib
import sys
import typing as t

# third party packages
import yaml

# logger
logger = logging.getLogger(__name__)


def load_config_from_input_args(
    converter: t.Callable[[t.Dict], t.Any]
) -> t.Optional[t.Any]:
    """入力引数の第一引数を設定ファイルとして読み込む.

    Returns:
        t.Optional[t.Any]: 読み込んだ設定ファイル情報
    """
    if len(sys.argv) != 2:
        logger.error(f"input arguments error: {sys.argv}")
        return None

    config_path = pathlib.Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"config does not exist: {config_path}")
        return None

    with open(str(config_path), "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    return converter(config)
