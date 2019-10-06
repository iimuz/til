from logging import getLogger

import tensorflow as tf

logger = getLogger(__name__)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim: int = 50) -> None:
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

    def decode(self, batch, apply_sigmoid=False):
        logits = self.generative_net(batch)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def encode(self, batch):
        mean, logvar = tf.split(self.inference_net(batch), num_or_size_splits=2, axis=1)
        logger.info(f"encode type: mean: {type(mean)}, logvar: {type(logvar)}")
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        val = eps * tf.exp(logvar * 0.5) + mean
        logger.info(f"reparameterize type {type(val)}")
        return val

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    logger.info(f"eager execution: {tf.executing_eagerly()}")

    model = CVAE()
    model.inference_net.summary()
    model.generative_net.summary()


if __name__ == "__main__":
    _main()
