# default packages
import time
from typing import List

# thrid party
import matplotlib.pyplot as plt
import tensorflow as tf


class Batch:
    def __init__(self) -> None:
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.start_time = time.time()
        self.end_time = self.start_time

    def result(self) -> None:
        self.end_time = time.time()


class Epoch:
    def __init__(self) -> None:
        self.loss: List[float] = []
        self.calc_time: List[float] = []

    def result(self, batch: Batch) -> None:
        self.loss.append(batch.loss.result().numpy())
        self.calc_time.append(batch.end_time - batch.start_time)

    def get_latest(self) -> str:
        return f"loss: {self.loss[-1]:.4e}" f", time: {self.calc_time[-1]:.4e} sec"


def show_image(history: Epoch, filepath: str = None) -> None:
    plt.figure()
    plt.subplot(2, 1, 1)
    _show_loss_image(history.loss)

    plt.subplot(2, 1, 2)
    _show_history_time(history.calc_time)

    if filepath is not None:
        plt.savefig(filepath)

    plt.show()


def _show_loss_image(loss: List[float]) -> None:
    plt.plot(loss, label="Loss")
    plt.ylabel("Loss")
    plt.title("Training Loss")


def _show_history_time(times: List[float]) -> None:
    plt.plot(times, label="time")
    plt.ylabel("time [sec]")
    plt.title("Calculate Time")
