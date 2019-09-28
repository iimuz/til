import tensorflow as tf
import tensorflow_datasets as tfds

from logging import getLogger

from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter
from tensorflow.python.framework.ops import EagerTensor
from tensorflow_datasets.core.dataset_info import DatasetInfo
from typing import Tuple

logger = getLogger(__name__)


def image_formatter(
    image: EagerTensor, label: EagerTensor, image_size=(160, 160)
) -> Tuple[EagerTensor, EagerTensor]:
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, image_size)

    return image, label


def get_dataset(
    split_weights=(8, 1, 1)
) -> Tuple[DatasetV1Adapter, DatasetV1Adapter, DatasetV1Adapter, DatasetInfo]:
    splits = tfds.Split.TRAIN.subsplit(weighted=split_weights)

    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        "cats_vs_dogs", split=list(splits), with_info=True, as_supervised=True
    )

    return raw_train, raw_validation, raw_test, metadata


def get_preprocessed_dataset(
    split_weights=(8, 1, 1), image_size=(160, 160)
) -> Tuple[DatasetV1Adapter, DatasetV1Adapter, DatasetV1Adapter, DatasetInfo]:
    def converter(dataset):
        return dataset.map(
            lambda image, label: image_formatter(image, label, image_size)
        )

    raw_train, raw_validation, raw_test, metadata = get_dataset(split_weights)
    raw_train = converter(raw_train)
    raw_validation = converter(raw_validation)
    raw_test = converter(raw_test)

    return raw_train, raw_validation, raw_test, metadata


def get_batch_dataset(
    split_weights=(8, 1, 1),
    image_size=(160, 160),
    batch_size=32,
    shuffle_buffer_size=1000,
    shuffle_seed=None,
) -> Tuple[DatasetV1Adapter, DatasetV1Adapter, DatasetV1Adapter, DatasetInfo]:
    """バッチサイズのデータを取得するためのアダプタを取得する

    Args:
        split_weights (tuple, optional): (学習、検証、テスト)の. Defaults to (8, 1, 1).
        image_size (tuple, optional): [description]. Defaults to (160, 160).
        batch_size (int, optional): [description]. Defaults to 32.
        shuffle_buffer_size (int, optional): [description]. Defaults to 1000.
        shuffle_seed ([type], optional): [description]. Defaults to None.

    Returns:
        Tuple[DatasetV1Adapter, DatasetV1Adapter, DatasetV1Adapter, DatasetInfo]: [description]
    """

    def batch_converter(dataset):
        return dataset.batch(batch_size)

    raw_train, raw_validation, raw_test, metadata = get_preprocessed_dataset()
    raw_train = batch_converter(
        raw_train.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    )
    raw_validation = batch_converter(raw_validation)
    raw_test = batch_converter(raw_test)

    return raw_train, raw_validation, raw_test, metadata


def _main() -> None:
    """スクリプトの動作簡易確認用
    """
    import logging
    import tensorflow.compat.v1 as tfv1

    logging.basicConfig(level=logging.INFO)
    tfds.disable_progress_bar()
    tfv1.enable_eager_execution()

    # datasets
    raw_train, _, _, metadata = get_dataset()
    get_label_name = metadata.features["label"].int2str
    for image, label in raw_train.take(2):
        _show_and_save_image(image, get_label_name(label), "normal")

    # preprocessed datasets
    raw_train, _, _, metadata = get_preprocessed_dataset()
    get_label_name = metadata.features["label"].int2str
    for image, label in raw_train.take(2):
        _show_and_save_image(image, get_label_name(label), "preprocessed")

    # batch data
    raw_train, _, _, metadata = get_batch_dataset()
    for image_batch, label_batch in raw_train.take(1):
        logger.info(f"batch image shape = {image_batch.shape}")
        logger.info(f"batch label shape = {label_batch.shape}")


def _show_and_save_image(image: EagerTensor, label: str, prefix: str) -> None:
    """画像を表示し保存する

    Args:
        image (EagerTensor): 画像データ
        label (str): ラベル名称
        prefix (str): 保存時のファイル名のプレフィックス
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(image)
    plt.title(label)
    plt.show()
    plt.savefig(f"_data/{prefix}_{label}.png")


if __name__ == "__main__":
    _main()
