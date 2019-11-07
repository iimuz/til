# default packages
from logging import getLogger

# thrid party
import tensorflow as tf

logger = getLogger(__name__)


class Checkpoint:
    def __init__(
        self, save_dir: str, max_to_keep: int = 3, restore: bool = True, **ckpt_kwargs
    ) -> None:
        self._checkpoint = tf.train.Checkpoint(**ckpt_kwargs)
        self._manager = tf.train.CheckpointManager(
            self._checkpoint, save_dir, max_to_keep=max_to_keep
        )
        if restore:
            self.restore()

    def restore(self) -> None:
        self._checkpoint.restore(self._manager.latest_checkpoint)
        if self._manager.latest_checkpoint:
            logger.info(f"Restored from {self._manager.latest_checkpoint}")
        else:
            logger.info(f"Initialize from scratch.")

    def save(self) -> None:
        self._manager.save()

    def save_counter(self) -> int:
        return self._checkpoint.save_counter.numpy()
