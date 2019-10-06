from logging import getLogger
from typing import Tuple

import tensorflow as tf

logger = getLogger(__name__)


def get_checkpoint_and_manager(
    save_dir: str, max_to_keep=3, **ckpt_kwargs
) -> Tuple[tf.train.Checkpoint, tf.train.CheckpointManager]:
    checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
    manager = tf.train.CheckpointManager(checkpoint, save_dir, max_to_keep=max_to_keep)

    return checkpoint, manager


def restore_latest(
    checkpoint: tf.train.Checkpoint, manager: tf.train.CheckpointManager
) -> None:
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info(f"Restore from {manager.latest_checkpoint}")
    else:
        logger.info("Initialize from scratch.")
