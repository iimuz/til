from logging import getLogger
from typing import Tuple

import tensorflow as tf

logger = getLogger(__name__)


class CVAE(tf.keras.Model):
    """Convolutional Variational Auto Encoder
    """

    def __init__(self, latent_dim: int = 50) -> None:
        """初期化

        Args:
            latent_dim (int, optional): 潜在空間の次元数. Defaults to 50.
        """
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation="relu"
                ),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation="relu",
                ),
                tf.keras.layers.Conv2DTranspose(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="SAME",
                    activation="relu",
                ),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"
                ),
            ]
        )

    def decode(self, batch: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
        """潜在空間から復号化する

        Args:
            batch (tf.Tensor): バッチデータ
            apply_sigmoid (bool, optional): シグモイド関数を通すかどうか. Defaults to False.

        Returns:
            tf.Tensor: 判定結果または確率
        """
        logits = self.generative_net(batch)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def encode(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """圧縮を行う

        Args:
            batch (tf.Tensor): バッチデータ

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (潜在空間での平均, 潜在空間での対数分散)
        """
        mean, logvar = tf.split(self.inference_net(batch), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
        """潜在空間での値を計算する

        Args:
            mean (tf.Tensor): 平均
            logvar (tf.Tensor): 分散の対数

        Returns:
            tf.Tensor: 潜在空間での値
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def sample(self, eps: tf.Tensor = None) -> tf.Tensor:
        """潜在空間からサンプリングを行う

        Args:
            eps (tf.Tensor, optional): 誤差. Defaults to None.

        Returns:
            tf.Tensor: サンプリング結果
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


def _main() -> None:
    """簡易動作スクリプト
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    logger.info(f"eager execution: {tf.executing_eagerly()}")

    model = CVAE()
    model.inference_net.summary()
    model.generative_net.summary()


if __name__ == "__main__":
    _main()
