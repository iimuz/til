import datasets
import model
import tensorflow as tf
import tensorflow.compat.v1 as tfv1

from logging import getLogger

logger = getLogger(__name__)


def start(network, predictor, test):
    for images, labels in test:
        predictor.predict_step(network, images, labels)


def main():
    tfv1.enable_eager_execution()
    logger.info(f"execute eagerly = {tf.executing_eagerly()}")
    logger.info(f"is gpu available = {tf.test.is_gpu_available()}")

    logger.info("get dataset...")
    _, test = datasets.get_dataset()

    logger.info("predicting...")
    network = model.MNISTModel()
    predictor = model.Predictor()
    model.Checkpoint(network=network)

    start(network, predictor, test)
    logger.info(
        f"loss: {predictor.predict_loss.result()}, accuracy: {predictor.predict_accuracy.result()}"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    main()
