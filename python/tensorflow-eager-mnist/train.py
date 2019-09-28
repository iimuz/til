import datasets
import model
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time

from logging import getLogger

logger = getLogger(__name__)


def start(network, trainer, train, epochs, checkpoint):
    start_epoch = checkpoint.save_counter()
    for epoch in range(start_epoch, epochs):
        for images, labels in train:
            trainer.train_step(network, images, labels)
        save_path = checkpoint.save()
        logger.info(
            f"epoch: {epoch},"
            f" loss: {trainer.train_loss.result()},"
            f" accuracy: {trainer.train_accuracy.result()}"
            f" save model: {save_path}"
        )


def main():
    tfv1.enable_eager_execution()
    logger.info(f"execute eagerly = {tf.executing_eagerly()}")
    logger.info(f"is gpu available = {tf.test.is_gpu_available()}")

    logger.info("get dataset...")
    train, _ = datasets.get_dataset()

    logger.info("learning...")
    network = model.MNISTModel()
    trainer = model.Trainer()
    checkpoint = model.Checkpoint(network=network, optimizer=trainer.optimizer)

    start_learning = time.time()
    start(network, trainer, train, 5, checkpoint)
    end_learning = time.time()
    logger.info(f"learning time: {end_learning - start_learning} sec")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    main()
