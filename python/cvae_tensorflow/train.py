import matplotlib.pyplot as plt
import pathlib
import pickle
import time
from logging import getLogger
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from IPython import display
except:
    display = None

import dataset
import utils
from network import CVAE

logger = getLogger(__name__)


@tf.function
def compute_apply_gradients(
    model: tf.keras.Model, batch: tf.Tensor, optimizer: tf.keras.optimizers.Optimizer
) -> None:
    """勾配を計算する

    Args:
        model (tf.keras.Model): ネットワーク
        batch (tf.Tensor): バッチデータ
        optimizer (tf.keras.optimizers.Optimizer): 最適化関数
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, batch)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def compute_loss(model: tf.keras.Model, batch: tf.Tensor) -> tf.Tensor:
    """損失の計算

    Args:
        model (tf.keras.Model): ネットワーク
        batch (tf.Tensor): バッチデータ

    Returns:
        tf.Tensor: 損失
    """
    mean, logvar = model.encode(batch)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=batch
    )
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def log_normal_pdf(
    sample: tf.Tensor, mean: tf.Tensor, logvar: tf.Tensor, raxis: int = 1
) -> tf.Tensor:
    """対数正規分布

    Args:
        sample (tf.Tensor): サンプル
        mean (tf.Tensor): 平均
        logvar (tf.Tensor): 対数分散
        raxis (int, optional): raxis. Defaults to 1.

    Returns:
        tf.Tensor: 分布
    """
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis
    )


def restore_history(filepath: str) -> List:
    """損失などの履歴を復元する

    Args:
        filepath (str): 復元するファイルパス

    Returns:
        List: ELBO の履歴
    """
    elbo_history: List = []
    if pathlib.Path(filepath).exists() is False:
        return elbo_history

    with open(filepath, "rb") as f:
        res = pickle.load(f)
    elbo_history = res["elbo_history"]

    return elbo_history


def save_history(elbo_history: List, filepath: str) -> None:
    """損失などの履歴を保存する

    Args:
        elbo_history (List): ELBO の履歴
        filepath (str): 保存するファイルパス
    """
    with open(filepath, "wb") as f:
        pickle.dump({"elbo_history": elbo_history}, f)


def show_and_save_history(elbo_history: List, filepath: str) -> None:
    """損失などの変化量を表示し、保存する

    Args:
        elbo_history (List): ELBO の履歴
        filepath (str): 画像を保存するファイルパス
    """
    plt.figure()
    plt.plot(elbo_history, label="ELBO")
    plt.title("ELBO")
    plt.savefig(filepath)
    plt.show()


def show_and_save_images(images, filepath: str):
    """復号化した画像の表示と保存

    Args:
        images ([type]): 画像データセット
        filepath (str): 保存するファイル名
    """
    fig = plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis("off")

    plt.savefig(filepath)
    plt.show()


def _main() -> None:
    """簡易動作スクリプト
    """
    import logging

    logging.basicConfig(level=logging.INFO)
    logger.info(f"eager execution: {tf.executing_eagerly()}")

    epochs = 2
    latent_dim = 50
    num_example_to_generate = 16
    random_vector_for_generation = tf.random.normal(
        shape=[num_example_to_generate, latent_dim]
    )
    history_filepath = "_data/history.pkl"
    history_image_filepath = "_data/history.png"

    train_dataset, test_dataset = dataset.get_batch_dataset(
        train_buff=60000, batch_size=32
    )
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint, checkpoint_manager = utils.get_checkpoint_and_manager(
        save_dir="_data/ckpts", max_to_keep=3, model=model, optimizer=optimizer
    )
    utils.restore_latest(checkpoint, checkpoint_manager)
    elbo_history = restore_history(history_filepath)

    start_epoch = checkpoint.save_counter.numpy()
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        for train_x in train_dataset:
            compute_apply_gradients(model, train_x, optimizer)
        end_time = time.time()

        checkpoint_manager.save()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo_history.append(-loss.result())

        if display is not None:
            display.clear_output(wait=False)
        logger.info(
            f"Epoch: {epoch}"
            f", Test set ELBO: {elbo_history[-1]}"
            f", time elapse for current epoch {end_time - start_time}"
        )
        predictions = model.sample(random_vector_for_generation)
        show_and_save_images(predictions, f"_data/image_at_epoch_{epoch:04d}.png")
        show_and_save_history(elbo_history, history_image_filepath)
        save_history(elbo_history, history_filepath)
    utils.save_gif("_data/", "image_at_epoch_*.png", "_data/cvae.gif")


if __name__ == "__main__":
    _main()
