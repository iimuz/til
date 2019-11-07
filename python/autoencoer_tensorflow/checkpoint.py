# default packages
from logging import getLogger

# thrid party
import tensorflow as tf

logger = getLogger(__name__)


class Checkpoint:
    """学習の経過を保存するためのクラス
    """

    def __init__(
        self, save_dir: str, max_to_keep: int = 3, restore: bool = True, **ckpt_kwargs
    ) -> None:
        """Initialize

        Args:
            save_dir (str): 保存先ディレクトリ
            max_to_keep (int, optional): 最大保持数. Defaults to 3.
            restore (bool, optional): 初期化時に保存ディレクトリから最新データを復元するかどうか. Defaults to True.
        """
        self._checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
        self._manager = tf.train.CheckpointManager(
            self._checkpoint, save_dir, max_to_keep=max_to_keep
        )
        if restore:
            self.restore()

    def restore(self) -> None:
        """保存ディレクトリから最新の結果を復元する
        """
        self._checkpoint.restore(self._manager.latest_checkpoint)
        if self._manager.latest_checkpoint:
            logger.info(f"Restored from {self._manager.latest_checkpoint}")
        else:
            logger.info(f"Initialize from scratch.")

    def save(self) -> None:
        """保存ディレクトリに結果を保存する
        """
        self._manager.save()

    def save_counter(self) -> int:
        """現在のデータが何回目のデータかを返す

        Returns:
            int: 現在の進捗
        """
        return self._checkpoint.save_counter.numpy()
