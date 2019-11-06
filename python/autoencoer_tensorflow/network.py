# default package
from typing import Tuple

# third party
import tensorflow as tf
from tensorflow.keras import layers


class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim: int) -> None:
        super(Autoencoder, self).__init__()
        dims = 32
        self.encoder = _make_encoder(input_dim, dims)
        self.decoder = _make_decoder(dims, input_dim)

    def call(self, inputs: tf.Tensor) -> None:
        code = self.encoder(inputs)
        reconstruct = self.decoder(code)
        return reconstruct

    @tf.function
    def train_step(
        self,
        inputs: tf.Tensor,
        loss_obj: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
    ) -> None:
        with tf.GradientTape() as tape:
            reconstruct = self(inputs)
            loss = loss_obj(inputs, reconstruct)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))


def _make_decoder(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([layers.Dense(output_dim, activation="relu")])

    return model


def _make_encoder(input_dim: int, output_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([layers.Dense(output_dim, activation="relu")])

    return model
