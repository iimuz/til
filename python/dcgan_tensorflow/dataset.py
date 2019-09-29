import tensorflow as tf

from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.ops import EagerTensor
from typing import Tuple


def image_formatter(
    image: EagerTensor, image_size: Tuple[int, int] = (28, 28)
) -> EagerTensor:
    """画像の前処理

    Args:
        image (EagerTensor): 入力画像
        image_size (Tuple[int, int], optional): 出力する画像サイズ. Defaults to (28, 28).

    Returns:
        EagerTensor: 出力画像
    """
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [image_size[0], image_size[1], 1])
    image = (image - 127.5) / 127.5
    image = tf.image.resize(image, image_size)

    return image


def get_batch_dataset(
    image_size: Tuple[int, int] = (28, 28),
    shuffle_buffer_size: int = 60000,
    shuffle_seed: int = 0,
    batch_size: int = 256,
) -> Tuple[DatasetV1Adapter, DatasetV1Adapter]:
    """バッチサイズのデータセットを取得する

    Args:
        image_size (Tuple[int, int], optional): 画像サイズ. Defaults to (28, 28).
        shuffle_buffer_size (int, optional): シャッフルの際に使うバッファサイズ. Defaults to 60000.
        shuffle_seed (int, optional): シャッフルのシード値. Defaults to 0.
        batch_size (int, optional): バッチサイズ. Defaults to 256.

    Returns:
        Tuple[DatasetV1Adapter, DatasetV1Adapter]: (学習用セット, 検証用セット)
    """
    train_dataset, test_dataset = get_dataset(image_size)
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_dataset = train_dataset.shuffle(
        buffer_size=shuffle_buffer_size, seed=shuffle_seed
    ).batch(batch_size)
    test_dataset = test_dataset.map(
        lambda image: image_formatter(image, image_size)
    ).batch(batch_size)

    return train_dataset, test_dataset


def get_dataset(
    image_size: Tuple[int, int] = (28, 28),
) -> Tuple[DatasetV1Adapter, DatasetV1Adapter]:
    """データセットを取得する

    Args:
        image_size (Tuple[int, int], optional): 取得する画像サイズ. Defaults to (28, 28).

    Returns:
        Tuple[DatasetV1Adapter, DatasetV1Adapter]: (学習用画像セット, テスト用画像セット)
    """
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).map(
        lambda image: image_formatter(image, image_size)
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)

    return train_dataset, test_dataset


def _main() -> None:
    """簡易テスト用スクリプト
    """
    import tensorflow.compat.v1 as tfv1

    tfv1.enable_eager_execution()
    raw_train, _ = get_dataset()
    for idx, image in enumerate(raw_train.take(2)):
        _show_and_save_image(
            tf.reshape(image, (28, 28)), f"image_{idx}", f"_data/image_{idx}.png"
        )


def _show_and_save_image(image: EagerTensor, title: str, filepath: str) -> None:
    """画像を表示し、保存する

    Args:
        image (EagerTensor): 画像
        title (str): タイトル
        filepath (str): 保存するファイルパス
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()
    plt.savefig(filepath)


if __name__ == "__main__":
    _main()
