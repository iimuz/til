import tensorflow as tf
import tensorflow_datasets as tfds


def convert_types(image, label):
    image = tf.cast(image, tf.float32) / 255
    return image, label


def get_dataset():
    batch_size = 32

    dataset = tfds.load("mnist", as_supervised=True)
    train, test = dataset["train"], dataset["test"]
    train = train.map(convert_types).shuffle(10000).batch(batch_size)
    test = test.map(convert_types).batch(batch_size)

    return train, test
