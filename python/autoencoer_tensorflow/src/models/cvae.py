# default packages
from logging import getLogger
from typing import Tuple

# third party
import numpy as np
import tensorflow as tf

# my packages
from src.data import history

# logger
logger = getLogger(__name__)


class CVAE(tf.keras.Model):
    """Convolutional Variational Auto Encoder
    """

    def __init__(self) -> None:
        """初期化
        """
        super(CVAE, self).__init__()
        self.latent_dim = 50
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
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        mean, logvar = encode(self, inputs)
        z = _reparameterize(mean, logvar)
        x_logit = decode(self, z)
        return x_logit

    @tf.function
    def train_step(
        self,
        batch: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        batch_history: history.Batch = None,
    ) -> None:
        """勾配を計算する

        Args:
            batch (tf.Tensor): バッチデータ
            optimizer (tf.keras.optimizers.Optimizer): 最適化関数
        """
        with tf.GradientTape() as tape:
            loss = _compute_loss(self, batch)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))


def decode(model: CVAE, batch: tf.Tensor, apply_sigmoid: bool = False) -> tf.Tensor:
    """潜在空間から復号化する

    Args:
        batch (tf.Tensor): バッチデータ
        apply_sigmoid (bool, optional): シグモイド関数を通すかどうか. Defaults to False.

    Returns:
        tf.Tensor: 判定結果または確率
    """
    logits = model.generative_net(batch)
    if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs

    return logits


def encode(model: CVAE, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """圧縮を行う

    Args:
        batch (tf.Tensor): バッチデータ

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (潜在空間での平均, 潜在空間での対数分散)
    """
    mean, logvar = tf.split(model.inference_net(batch), num_or_size_splits=2, axis=1)
    return mean, logvar


def reconstruct(model: CVAE, inputs: tf.Tensor) -> tf.Tensor:
    mean, logvar = encode(model, inputs)
    z = _reparameterize(mean, logvar)
    x_logit = decode(model, z, apply_sigmoid=True)
    return x_logit


@tf.function
def sample(model: CVAE, eps: tf.Tensor = None) -> tf.Tensor:
    """適当な値で潜在空間から復号化する

    Args:
        eps (tf.Tensor, optional): 誤差. Defaults to None.

    Returns:
        tf.Tensor: サンプリング結果
    """
    if eps is None:
        eps = tf.random.normal(shape=(100, model.latent_dim))
    return decode(model, eps, apply_sigmoid=True)


@tf.function
def _compute_loss(model: CVAE, batch: tf.Tensor) -> tf.Tensor:
    """損失の計算

    Args:
        model (CVAE): ネットワーク
        batch (tf.Tensor): バッチデータ

    Returns:
        tf.Tensor: 損失
    """
    mean, logvar = encode(model, batch)
    z = _reparameterize(mean, logvar)
    x_logit = decode(model, z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=batch
    )
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = _log_normal_pdf(z, 0.0, 0.0)
    logqz_x = _log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def _log_normal_pdf(
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


def _reparameterize(mean: tf.Tensor, logvar: tf.Tensor) -> tf.Tensor:
    """潜在空間から値をサンプリングする

    Args:
        mean (tf.Tensor): 平均
        logvar (tf.Tensor): 分散の対数

    Returns:
        tf.Tensor: 潜在空間での値
    """
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * 0.5) + mean
