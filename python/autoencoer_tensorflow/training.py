from logging import getLogger

# third party
import tensorflow as tf

# my pakcages
import history
import network

logger = getLogger(__name__)


def train(dataset: tf.data.Dataset) -> None:
    input_shape = 784
    epochs = 2

    model = network.Autoencoder(input_shape)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.mean_squared_error
    # loss = tf.keras.losses.binary_crossentropy

    epoch_history = history.Epoch()
    for epoch in range(epochs):
        batch_history = history.Batch()
        for batch in dataset:
            model.train_step(batch, loss, optimizer, batch_history)
        batch_history.result()
        epoch_history.result(batch_history)

        logger.info(f"epoch: {epoch}/{epochs}, {epoch_history.get_latest()}")


def _convert_types(image: tf.Tensor, dims: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.reshape(image, [dims])
    return image


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    dims = 28 * 28
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    train_ds = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .map(lambda x: _convert_types(x, dims))
        .shuffle(10000)
        .batch(128)
    )

    train(train_ds)


if __name__ == "__main__":
    _main()
