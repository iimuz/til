import pathlib
from logging import getLogger
from typing import Tuple

import imageio
import tensorflow as tf

logger = getLogger(__name__)


def get_checkpoint_and_manager(
    save_dir: str, max_to_keep=3, **ckpt_kwargs
) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
    """チェックポイントの取得

    Args:
        save_dir (str): チェックポイントの保存ディレクトリ
        max_to_keep (int, optional): チェックポイントの最大保存数. Defaults to 3.

    Returns:
        Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]: (チェックポイント, チェックポイントマネージャ)
    """
    checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(checkpoint, save_dir, max_to_keep=max_to_keep)

    return checkpoint, manager


def restore_latest(
    checkpoint: tf.train.Checkpoint, manager: tf.train.CheckpointManager
) -> None:
    """最新のチェックポイントを復元

    Args:
        checkpoint (tf.train.Checkpoint): チェックポイント
        manager (tf.train.CheckpointManager): チェックポイントマネージャ
    """
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f"Restore from {manager.latest_checkpoint}")
    else:
        logger.info("Initialize from scratch.")


def save_gif(search_dir: str, search_query: str, filepath: str) -> None:
    """ファイルを gif としてまとめる

    Args:
        search_dir (str): ファイル探索ディレクトリ
        search_query (str): ファイル探索クエリ
        filepath (str): 保存するファイルパス
    """
    with imageio.get_writer(filepath, mode="I") as writer:
        filenames = sorted(pathlib.Path(search_dir).glob(search_query))
        for filename in filenames:
            image = imageio.imread(str(filename))
            writer.append_data(image)
            logger.info(f"write gif from {filename}")
