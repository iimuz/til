from logging import getLogger

import numpy as np
import tensorflow as tf

logger = getLogger(__name__)


def dataset_formatter(
    dataset: np.ndarray, width: int, height: int, channels: int
) -> np.ndarray:
    """データセットをバイナリ画像の集合へ変換する

    Args:
        dataset (np.ndarray): 変換する画像セット
        width (int): 一枚当たりの画像の横幅
        height (int): 一枚当たりの画像の縦幅
        channels (int): 一枚当たりの画像のチャンネル数

    Returns:
        np.ndarray: 変換した画像セット
    """
    dataset = dataset.reshape(dataset.shape[0], width, height, channels)
    dataset = dataset.astype(np.float32)
    dataset /= 255.0
    dataset[dataset >= 0.5] = 1
    dataset[dataset < 0.5] = 0

    return dataset


def get_dataset() -> np.ndarray:
    """画像データセットを取得する

    Returns:
        np.ndarray: 画像データセット
    """
    width = 28
    height = 28
    channels = 1

    (train_dataset, _), (test_dataset, _) = tf.keras.datasets.mnist.load_data()
    train_dataset = dataset_formatter(train_dataset, width, height, channels)
    test_dataset = dataset_formatter(test_dataset, width, height, channels)

    return train_dataset, test_dataset


def get_batch_dataset(
    train_buff: int = 60000, batch_size: int = 128
) -> tf.data.Dataset:
    """バッチごとの画像データセットを取得する

    Args:
        train_buff (int, optional): シャッフルに利用するバッファサイズ. Defaults to 60000.
        batch_size (int, optional): バッチサイズ. Defaults to 128.

    Returns:
        tf.data.Dataset: バッチ単位の画像データセット
    """
    train_dataset, test_dataset = get_dataset()
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_dataset)
        .shuffle(train_buff)
        .batch(batch_size)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset).batch(batch_size)

    return train_dataset, test_dataset


def _main() -> None:
    """簡易動作確認するクリプト
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(f"eager execution: {tf.executing_eagerly()}")

    train_dataset, test_dataset = get_batch_dataset()
    logger.info(f"train_dataset: {type(train_dataset)}")

    for image in train_dataset.take(1):
        _show_and_save_image(
            tf.reshape(image[0, :, :, :], (28, 28)), "image", "_data/image.png"
        )


def _show_and_save_image(image: tf.Tensor, title: str, filepath: str) -> None:
    """画像の保存と表示を実行する

    Args:
        image (tf.Tensor): 画像データセット
        title (str): 画像のタイトル
        filepath (str): 保存するファイルパス
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.savefig(filepath)
    plt.show()


if __name__ == "__main__":
    _main()
