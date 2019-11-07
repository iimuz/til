# default packages
from logging import getLogger
from typing import Tuple

# third party
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = getLogger(__name__)


def show_images(
    inputs: tf.Tensor,
    reconstruct: tf.Tensor,
    image_shape: Tuple[int, int, int],
    filepath: str = None,
) -> None:
    """入力画像と再構成画像を一枚の画像に出力する

    Args:
        inputs (tf.Tensor): 入力画像群
        reconstruct (tf.Tensor): 再構成画像群
        image_shape (Tuple[int, int, int]): 画像のサイズ
        filepath (str, optional): 保存するファイルパス. Defaults to None.
    """
    batch_size = inputs.shape[0]
    rows = 8
    cols = batch_size // rows

    inputs_concat = _concat_images(inputs.numpy(), image_shape, rows, cols)
    reconstruct_concat = _concat_images(reconstruct.numpy(), image_shape, rows, cols)
    show_image = np.hstack([inputs_concat, reconstruct_concat])

    plt.figure()
    plt.imshow(show_image, cmap="gray")

    if filepath is not None:
        plt.savefig(filepath)

    plt.close()


def _concat_images(
    images: np.ndarray, image_shape: Tuple[int, int, int], rows: int, cols: int
) -> np.ndarray:
    """バッチ単位の画像群を一枚の画像に結合する

    Args:
        images (np.ndarray): 画像群
        image_shape (Tuple[int, int, int]): 一枚当たりの画像サイズ
        rows (int): 画像を結合するときに行数
        cols (int): 画像を結合するときの列数

    Returns:
        np.ndarray: 一枚にした画像
    """
    output = images.reshape(
        (rows, cols, image_shape[0], image_shape[1], image_shape[2])
    )
    output = np.hstack(output)  # 縦方向に画像を結合。(cols, height * rows, width, channel) になる。
    output = np.hstack(output)  # 横方向に画像を結合。(height * rows, width * cols, channel) になる。

    # 1 channel しかない場合は、最終軸を削除
    if output.shape[2] == 1:
        output = output.reshape((output.shape[0], output.shape[1]))

    return output
