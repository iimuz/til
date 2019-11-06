# third party
import tensorflow as tf

# my pakcages
import network


def train(dataset: tf.data.Dataset) -> None:
    input_shape = 784
    epochs = 2

    model = network.Autoencoder(input_shape)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    # loss = tf.keras.losses.mean_squared_error
    loss = tf.keras.losses.binary_crossentropy
    for epoch in range(epochs):
        for batch in dataset:
            model.train_step(batch, loss, optimizer)


def _convert_types(image: tf.Tensor, dims: int) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image /= 255.0
    image = tf.reshape(image, [dims])
    return image


def _main() -> None:
    dims = 784
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
