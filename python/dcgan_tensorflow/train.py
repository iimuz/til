import math
import pathlib
import pickle
import time
from logging import getLogger
from typing import List, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter

try:
    from IPython import display
except:
    display = None

import dataset
import network
import utils

logger = getLogger(__name__)


def restore_history(filepath: str) -> Tuple[List, List]:
    """評価指標の履歴を復元する

    Args:
        filepath (str): 評価指標の履歴を保存したファイルパス

    Returns:
        Tuple[List, List]: (Generator History, Discriminator History)
    """
    generator_history = []
    discriminator_history = []

    if pathlib.Path(filepath).exists() == False:
        return generator_history, discriminator_history

    with open(filepath, "rb") as f:
        res = pickle.load(f)
    generator_history = res["generator_history"]
    discriminator_history = res["discriminator_history"]

    return generator_history, discriminator_history


def save_history(
    generator_history: List, discriminator_history: List, filepath: str
) -> None:
    """評価指標の履歴をファイルに保存する

    Args:
        generator_history (List): Generator の評価指標の履歴
        discriminator_history (List): Discriminator の評価指標の履歴
        filepath (str): 保存するファイルパス
    """
    with open(filepath, "wb") as f:
        pickle.dump(
            {
                "generator_history": generator_history,
                "discriminator_history": discriminator_history,
            },
            f,
        )


def show_and_save_images(
    images: tf.Tensor, filepath: str, display_cols: int = 4
) -> None:
    """Generator の作成した画像を表示し、保存する

    Args:
        images (tf.Tensor): 出力する画像群
        filepath (str): 保存するパス
        display_cols (int, optional): 一列に並べる画像の数. Defaults to 4.
    """
    display_rows = math.ceil(images.shape.as_list()[0] / display_cols)

    plt.figure()
    for i in range(images.shape[0]):
        plt.subplot(display_rows, display_cols, i + 1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig(filepath)
    plt.show()


def show_and_save_metrics(
    generator_history: List, discriminator_history: List, filepath: str
) -> None:
    """学習中の指標を表示し、保存する

    Args:
        generator_history (List): Generator の指標
        discriminator_history (List): Discriminator の指標
        filepath (str): 保存するファイルパス
    """
    plt.figure()
    plt.plot(generator_history, label="Generator Loss")
    plt.plot(discriminator_history, label="Discriminator Loss")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(filepath)
    plt.show()


def train(
    dataset: DatasetV1Adapter,
    batch_size: int = 256,
    epochs: int = 50,
    gen_input_dim: int = 100,
    disc_input_shape: Tuple[int, int, int] = (28, 28, 1),
) -> None:
    """学習を実行する

    Args:
        dataset (DatasetV1Adapter): 学習に用いる通常画像
        batch_size (int, optional): バッチサイズ. Defaults to 256.
        epochs (int, optional): 学習するエポック数. Defaults to 50.
        gen_input_dim (int, optional): Generator に利用するノイズの次元数. Defaults to 100.
        disc_input_shape (Tuple[int, int, int], optional): 学習画像のサイズ. Defaults to (28, 28, 1).
    """
    history_filepath = "_data/history.pkl"
    history_image_filepath = "_data/history.png"

    generator = network.make_generator_model(input_dim=gen_input_dim)
    generator_optimizer = network.generator_optimizer()
    discriminator = network.make_discriminator_model(input_shape=disc_input_shape)
    discriminator_optimizer = network.discriminator_optimizer()

    checkpoint, checkpoint_manager = utils.get_checkpoint_and_manager(
        save_dir="_data/ckpts",
        max_to_keep=3,
        generator=generator,
        generator_optimizer=generator_optimizer,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
    )
    utils.restore_latest(checkpoint, checkpoint_manager)
    generator_history, discriminator_history = restore_history(history_filepath)

    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, gen_input_dim])

    generator_loss = tf.keras.metrics.Mean(name="generator_loss")
    discriminator_loss = tf.keras.metrics.Mean(name="discriminator_loss")
    start_epoch = checkpoint.save_counter.numpy()
    for epoch in range(start_epoch, epochs):
        start_learning = time.time()
        for image_batch in dataset:
            train_step(
                image_batch,
                generator,
                generator_optimizer,
                discriminator,
                discriminator_optimizer,
                batch_size,
                gen_input_dim,
                generator_loss,
                discriminator_loss,
            )
        end_learning = time.time()
        generator_history.append(generator_loss.result())
        discriminator_history.append(discriminator_loss.result())
        logger.info(
            f"Epoch {epoch},"
            f" Generator Loss: {generator_history[-1]},"
            f" Discriminator Loss: {discriminator_history[-1]},"
            f" Time: {end_learning - start_learning} sec"
        )

        checkpoint_manager.save()
        save_history(generator_history, discriminator_history, history_filepath)

        if display is not None:
            display.clera_output(wait=True)
        show_and_save_images(
            generator(seed, training=False),
            f"_data/image_at_epoch_{epoch:04d}.png",
            display_cols=4,
        )
        show_and_save_metrics(
            generator_history, discriminator_history, history_image_filepath
        )


@tf.function
def train_step(
    images: tf.Tensor,
    generator: tf.keras.Model,
    generator_optimizer: tf.keras.Optimizers.Optimizer,
    discriminator: tf.keras.Model,
    discriminator_optimizer: tf.keras.Optimizers.Optimizer,
    batch_size: int = 256,
    noise_dim: int = 100,
    train_generator_loss: List = None,
    train_discriminator_loss: List = None,
) -> None:
    """学習時の単一ステップ

    Args:
        images (tf.Tensor): 1バッチの画像群
        generator (tf.keras.Model): Generator model
        generator_optimizer (tf.keras.Optimizers.Optimizer): Generator optimizer
        discriminator (tf.keras.Model): Discriminator model
        discriminator_optimizer (tf.keras.Optimizers.Optimizer): Discriminator optimizer
        batch_size (int, optional): batch size. Defaults to 256.
        noise_dim (int, optional): input noise dimensioon for generator. Defaults to 100.
        train_generator_loss (List, optional): Output for generator loss. Defaults to None.
        train_discriminator_loss (List, optional): Output for discriminator loss. Defaults to None.
    """
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = network.generator_loss(fake_output)
        disc_loss = network.discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    if train_generator_loss is not None:
        train_generator_loss(gen_loss)
    if train_discriminator_loss is not None:
        train_discriminator_loss(disc_loss)


def _main() -> None:
    """簡易動作用スクリプト
    """
    import logging

    import tensorflow.compat.v1 as tfv1

    logging.basicConfig(level=logging.INFO)
    tfv1.enable_eager_execution()

    dataset_train, _ = dataset.get_batch_dataset()
    train(dataset_train, epochs=2)
    utils.save_gif("_data/", "image_at_epoch_*", "_data/dcgan.gif")


if __name__ == "__main__":
    _main()
