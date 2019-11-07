# default packages
import time
from typing import List

# thrid party
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
