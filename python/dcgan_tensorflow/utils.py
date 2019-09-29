import pathlib
from logging import getLogger
from typing import Tuple

import imageio
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


def save_gif(search_dir: str, search_query: str, filepath: str) -> None:
    """ファイルを探して gif としてまとめる

    Args:
        search_dir (str): ファイル探索ディレクトリ
        search_query (str): ファイル探索クエリ
        filepath (str): 保存するファイルパス
    """
    with imageio.get_writer(filepath, mode="I") as writer:
        filenames = pathlib.Path(search_dir).glob(search_query)
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
            logger.info(f"write gif from {filename}")
