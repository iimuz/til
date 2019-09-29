import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor) -> tf.Tensor:
    """Discriminator 用の損失関数を計算する

    Args:
        real_output (tf.Tensor): 本物の画像に対する判定結果
        fake_output (tf.Tensor): 生成した画像に対する判定結果

    Returns:
        tf.Tensor: 損失
    """
    real_loss = _get_loss_function()(tf.ones_like(real_output), real_output)
    fake_loss = _get_loss_function()(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def discriminator_optimizer() -> Optimizer:
    """Discriminator 用のオプティマイザを取得する

    Returns:
        Optimizer: オプティマイザ
    """
    return tf.keras.optimizers.Adam(1e-4)


def generator_loss(fake_output: tf.Tensor) -> tf.Tensor:
    """Generator 用の損失を計算する

    Args:
        fake_output (tf.Tensor): 生成した画像の判定結果

    Returns:
        tf.Tensor: 損失
    """
    return _get_loss_function()(tf.ones_like(fake_output), fake_output)


def generator_optimizer() -> Optimizer:
    """Generator 用のオプティマイザを取得する

    Returns:
        Optimizer: オプティマイザ
    """
    return tf.keras.optimizers.Adam(1e-4)


def make_discriminator_model(input_shape=(28, 28, 1)) -> tf.keras.Model:
    """Discriminator モデルを生成する

    Args:
        input_shape (tuple, optional): 入力次元. Defaults to (28, 28, 1).

    Returns:
        tf.keras.Model: Discriminator モデル
    """
    model = tf.keras.Sequential(
        [
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape
            ),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ]
    )

    return model


def make_generator_model(input_dim=100) -> tf.keras.Model:
    """Generator モデルを生成する

    Args:
        input_dim (int, optional): 入力次元. Defaults to 100.

    Returns:
        tf.keras.Model: Generator モデル
    """
    dense_size = (7, 7, 256)
    conv2d1_channel = 128
    conv2d2_channel = 64
    conv2d3_channel = 1

    model = tf.keras.Sequential()
    model.add(
        layers.Dense(
            dense_size[0] * dense_size[1] * dense_size[2],
            use_bias=False,
            input_shape=(input_dim,),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape(dense_size))
    assert model.output_shape == (None, dense_size[0], dense_size[1], dense_size[2])

    _add_conv2d_transpose_layer(
        model,
        conv2d1_channel,
        (5, 5),
        (1, 1),
        (None, dense_size[0], dense_size[1], conv2d1_channel),
    )
    _add_conv2d_transpose_layer(
        model,
        conv2d2_channel,
        (5, 5),
        (2, 2),
        (None, dense_size[0] * 2, dense_size[1] * 2, conv2d2_channel),
    )

    model.add(
        layers.Conv2DTranspose(
            conv2d3_channel,
            (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
        )
    )
    assert model.output_shape == (
        None,
        dense_size[0] * 4,
        dense_size[1] * 4,
        conv2d3_channel,
    )

    return model


def _add_conv2d_transpose_layer(
    model: tf.keras.Model,
    channel=128,
    filter_size=(5, 5),
    strides=(1, 1),
    output_shape=(None, 7, 7, 128),
) -> None:
    """Conv2DTranspose を利用して 1 層追加する

    Args:
        model (tf.keras.Model): 層を追加するモデル
        channel (int, optional): チャネル数. Defaults to 128.
        filter_size (tuple, optional): フィルタサイズ. Defaults to (5, 5).
        strides (tuple, optional): ストライド幅. Defaults to (1, 1).
        output_shape (tuple, optional): チェック用の出力次元. Defaults to (None, 7, 7, 128).
    """
    model.add(
        layers.Conv2DTranspose(
            channel, filter_size, strides=strides, padding="same", use_bias=False
        )
    )
    assert model.output_shape == output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


def _get_loss_function() -> Loss:
    """損失関数を取得する

    Returns:
        Loss: 損失関数
    """
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)


def _main() -> None:
    """簡易動作テスト用スクリプト
    """
    import logging
    import matplotlib.pyplot as plt
    import tensorflow.compat.v1 as tfv1

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    tfv1.enable_eager_execution()

    input_dim = 100
    generator = make_generator_model()
    generator.summary()

    noise = tf.random.normal([1, input_dim], seed=0)
    generated_image = generator(noise, training=False)

    plt.figure()
    plt.imshow(generated_image[0, :, :, 0], cmap="gray")
    plt.show()
    plt.savefig("_data/generator_input_image.png")

    discriminator = make_discriminator_model()
    discriminator.summary()

    decision = discriminator(generated_image)
    logger.info(f"decision result: {decision}")


if __name__ == "__main__":
    _main()
