# default
import functools
from typing import Tuple

# third party
import tensorflow as tf
from tensorflow.keras import layers

# my packages
from src.data import history


class Autoencoder(tf.keras.Model):
    """単純な全結合Autoencoder
    """

    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        """Initialize

        Args:
            input_dim (int): 入力次元数
        """
        super(Autoencoder, self).__init__()
        self.encoder, reduce_rate = _make_encoder()
        feature_space = (input_shape[0] // reduce_rate, input_shape[1] // reduce_rate)
        self.decoder = _make_decoder(feature_space)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        code = self.encoder(inputs)
        reconstruct = self.decoder(code)
        return reconstruct

    @tf.function
    def train_step(
        self,
        inputs: tf.Tensor,
        loss_obj: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        batch_history: history.Batch = None,
    ) -> None:
        """1バッチに対する学習

        Args:
            inputs (tf.Tensor): バッチ
            loss_obj (tf.keras.losses.Loss): 損失計算オブジェクト
            optimizer (tf.keras.optimizers.Optimizer): 最適化関数
            batch_history (history.Batch, optional): 計算履歴出力用. Defaults to None.
        """
        with tf.GradientTape() as tape:
            reconstruct = self(inputs)
            loss = loss_obj(inputs, reconstruct)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if batch_history is not None:
            batch_history.loss(loss)


def reconstruct(model: Autoencoder, inputs: tf.Tensor) -> tf.Tensor:
    """入力データをモデルを利用して再構成する

    Args:
        model (Autoencoder): 利用するモデル
        inputs (tf.Tensor): 入力データ

    Returns:
        tf.Tensor: 再構成結果
    """
    return model(inputs)


def _make_decoder(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """decoderを生成する

    Args:
        input_dim (int): 入力次元
        output_dim (int): 出力次元

    Returns:
        tf.keras.Model: decoderモデル
    """
    image_shape = input_shape + (16,)
    image_dims = functools.reduce(lambda x, y: x * y, image_shape)
    model = tf.keras.Sequential(
        [
            layers.Dense(units=image_dims, activation=tf.nn.relu),
            layers.Reshape(target_shape=image_shape),
            layers.Conv2DTranspose(
                32, (3, 3), strides=(2, 2), padding="same", activation="relu"
            ),
            layers.Conv2DTranspose(
                1, (3, 3), strides=(1, 1), padding="same", activation="relu"
            ),
        ]
    )

    return model


def _make_encoder() -> Tuple[tf.keras.Model, int]:
    """encoderを生成する

    Returns:
        tf.keras.Model: decoderモデル
    """
    model = tf.keras.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), strides=(2, 2), padding="same", activation="relu"
            ),
            layers.Flatten(),
            layers.Dense(units=32),
        ]
    )

    return model, 2
