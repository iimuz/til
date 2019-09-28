import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf

from logging import getLogger
from tensorflow.python.keras.callbacks import History

logger = getLogger(__name__)


def load_checkpoints(
    model: tf.keras.Model, save_dir="_data/ckpt"
) -> tf.keras.callbacks.ModelCheckpoint:
    """最新のチェックポイントをロードし、 callback を返す

    Args:
        model (tf.keras.Model): ロードするモデル
        save_dir (str, optional): チェックポイントの保存ディレクトリ. Defaults to "_data/ckpt".

    Returns:
        tf.keras.callbacks.ModelCheckpoint: チェックポイントのコールバック
    """
    checkpoint_latest = tf.train.latest_checkpoint(save_dir)
    if checkpoint_latest is not None:
        logger.info(f"load from {checkpoint_latest}")
        model.load_weights(checkpoint_latest)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(pathlib.Path(save_dir).joinpath("ckpt{epoch:04d}.ckpt")),
        save_weights_only=True,
        verbose=1,
        save_freq="epoch",
    )

    return checkpoint


def plot_history(history: History) -> None:
    """履歴をグラフとして表示し、保存する

    Args:
        history (History): 表示するグラフの履歴
    """
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")
    plt.show()
    plt.savefig("_data/accuracy.png")

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Loss")
    plt.ylim([0, 1])
    plt.title("Training and Validation Loss")
    plt.show()
    plt.savefig("_data/loss.png")
