from logging import getLogger

# third party
import tensorflow as tf
from tqdm import tqdm

# my pakcages
import history
import network
import visualize
from checkpoint import Checkpoint

logger = getLogger(__name__)


def train(dataset: tf.data.Dataset) -> None:
    image_shape = (28, 28, 1)
    input_shape = image_shape[0] * image_shape[1] * image_shape[2]
    epochs = 2
    history_filepath = "data/history.png"
    reconstruct_filepath = "data/reconstruct.png"

    model = network.Autoencoder(input_shape)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.mean_squared_error
    # loss = tf.keras.losses.binary_crossentropy

    checkpoint = Checkpoint(
        save_dir="data/ckpts",
        max_to_keep=3,
        restore=True,
        model=model,
        optimizer=optimizer,
    )
    epoch_history = history.Epoch()
    input_example = [data for data in dataset.take(1)][-1]
    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        # learning
        batch_history = history.Batch()
        for batch in dataset:
            model.train_step(batch, loss, optimizer, batch_history)

        # save results
        checkpoint.save()
        batch_history.result()
        epoch_history.result(batch_history)

        # show results
        progress_bar.set_description(f"{epoch_history.get_latest()}")
        history.show_image(epoch_history, filepath=history_filepath)
        visualize.show_images(
            input_example,
            network.reconstruct(model, input_example),
            image_shape,
            reconstruct_filepath,
        )


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
