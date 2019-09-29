from logging import getLogger
from typing import Tuple

import tensorflow as tf

logger = getLogger(__name__)


def get_checkpoint_and_manager(
    save_dir: str, max_to_keep=3, **ckpt_kwargs
) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
    """ネットワークの保存用チェックポイントとマネージャを取得する

    Args:
        save_dir (str): データの保存先ディレクトリ
        max_to_keep (int, optional): 最大保存数. Defaults to 3.

    Returns:
        Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]: (チェックポイント, マネージャ)
    """
    checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(checkpoint, save_dir, max_to_keep=max_to_keep)

    return checkpoint, manager


def restore_latest(
    checkpoint: tf.train.Checkpoint, manager: tf.train.CheckpointManager
) -> None:
    """最新のデータがあれば復元する

    Args:
        checkpoint (tf.train.Checkpoint): 設定済みチェックポイント
        manager (tf.train.CheckpointManager): 設定済みマネージャ
    """
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f"Restored from {manager.latest_checkpoint}")
    else:
        logger.info(f"Initialize from scratch.")
