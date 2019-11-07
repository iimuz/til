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


def _concat_images(
    images: np.ndarray, image_shape: Tuple[int, int, int], rows: int, cols: int
) -> np.ndarray:
    output = images.reshape(
        (rows, cols, image_shape[0], image_shape[1], image_shape[2])
    )
    output = np.hstack(output)  # 縦方向に画像を結合。(cols, height * rows, width, channel) になる。
    output = np.hstack(output)  # 横方向に画像を結合。(height * rows, width * cols, channel) になる。

    # 1 channel しかない場合は、最終軸を削除
    if output.shape[2] == 1:
        output = output.reshape((output.shape[0], output.shape[1]))

    return output
