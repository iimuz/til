# default packages
import random
from typing import List, Tuple

# third party
import matplotlib.pyplot as plt


def _axhspan() -> None:
    """水平方向に背景色を変更する
    """
    x, y = _create_data()
    pos_begin = (max(y) - min(y)) * 0.3 + min(y)
    pos_end = (max(y) - min(y)) * 0.7 + min(y)

    _ = plt.figure()
    plt.plot(x, y)
    plt.axhspan(pos_begin, pos_end, color="green", alpha=0.3)
    plt.savefig("data/axhspan.png")

    _clear_plot()


def _axvspan() -> None:
    """垂直方向に背景色を変更する
    """
    x, y = _create_data()
    pos_begin = x[int(len(x) * 0.5)]
    pos_end = x[int(len(x) * 0.8)]

    _ = plt.figure()
    plt.plot(x, y)
    plt.axvspan(pos_begin, pos_end, color="green", alpha=0.3)
    plt.savefig("data/axvspan.png")

    _clear_plot()


def _clear_plot() -> None:
    """matplotlib の figure を消去する
    """
    plt.clf()
    plt.cla()
    plt.close()


def _create_data() -> Tuple[List[int], List[float]]:
    """表示用にランダムで値を生成する

    Returns:
        Tuple[List[int], List[float]]: (x, y)
    """
    LIST_LENGTH = 100
    RANDOM_SEED = 0

    random.seed(RANDOM_SEED)
    x = list(range(LIST_LENGTH))
    y = [random.random() for _ in range(LIST_LENGTH)]

    return x, y


def _main() -> None:
    """スクリプト実行
    """
    _axvspan()
    _axhspan()


if __name__ == "__main__":
    _main()
