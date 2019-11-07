# default packages
import pathlib
import pickle
import time
from logging import getLogger
from typing import List

# thrid party
import matplotlib.pyplot as plt
import tensorflow as tf

logger = getLogger(__name__)


class Batch:
    """バッチごとの履歴を管理するクラス
    """

    def __init__(self) -> None:
        """Initialize
        """
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.start_time = time.time()
        self.end_time = self.start_time

    def result(self) -> None:
        """バッチ単位の計算を完了するときに呼び出す
        """
        self.end_time = time.time()


class Epoch:
    """エポック単位の履歴を管理するクラス
    """

    def __init__(self) -> None:
        """Initialize
        """
        self.loss: List[float] = []
        self.calc_time: List[float] = []

    def result(self, batch: Batch) -> None:
        """1エポックを完了するときに呼び出す

        Args:
            batch (Batch): バッチ単位の履歴情報
        """
        self.loss.append(batch.loss.result().numpy())
        self.calc_time.append(batch.end_time - batch.start_time)

    def get_latest(self) -> str:
        """最新の履歴情報を文字列として取得する

        Returns:
            str: 最新履歴の文字列
        """
        return f"loss: {self.loss[-1]:.4e}" f", time: {self.calc_time[-1]:.4e} sec"


def restore(filepath: str) -> Epoch:
    """エポック単位の履歴情報をファイルから復元する

    Args:
        filepath (str): 復元に利用するファイル

    Returns:
        Epoch: 復元結果
    """
    if filepath is None:
        logger.info("file path is None.")
        return Epoch()

    if pathlib.Path(filepath).exists() is False:
        logger.info(f"cannot find file: {filepath}")
        return Epoch()

    with open(filepath, "rb") as f:
        result = pickle.load(f)

    epoch = Epoch()
    epoch.loss = result["loss"]
    epoch.calc_time = result["calc_time"]

    return epoch


def save(history: Epoch, filepath: str) -> None:
    """エポック単位の履歴を保存する

    Args:
        history (Epoch): 保存するエポック単位の履歴
        filepath (str): 保存するファイルパス
    """
    if filepath is None:
        logger.info("file path is None.")
        return

    with open(filepath, "wb") as f:
        pickle.dump({"loss": history.loss, "calc_time": history.calc_time}, f)


def show_image(history: Epoch, filepath: str = None) -> None:
    """エポック単位の履歴を画像として出力する

    Args:
        history (Epoch): エポック単位の履歴
        filepath (str, optional): ファイルとして保存するときのファイルパス. Defaults to None.
    """
    plt.figure()
    plt.subplot(2, 1, 1)
    _show_loss_image(history.loss)

    plt.subplot(2, 1, 2)
    _show_history_time(history.calc_time)

    if filepath is not None:
        plt.savefig(filepath)

    plt.show()
    plt.close()


def _show_loss_image(loss: List[float]) -> None:
    """損失をグラフ化する

    Args:
        loss (List[float]): 損失の履歴
    """
    plt.plot(loss, label="Loss")
    plt.ylabel("Loss")
    plt.title("Training Loss")


def _show_history_time(times: List[float]) -> None:
    """計算時間をグラフ化する

    Args:
        times (List[float]): 計算時間の履歴
    """
    plt.plot(times, label="time")
    plt.ylabel("time [sec]")
    plt.title("Calculate Time")
