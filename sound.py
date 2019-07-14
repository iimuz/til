import pathlib
import pygame.mixer as mixer
import random
import time


class ShuffleList:
    """ランダムに並べた音源リストを管理する
    """

    def __init__(self, sound_dir: str) -> None:
        """初期化

        Args:
            sound_dir (str): 音源ファイルのディレクトリ
        """
        self._sound_list = [str(path) for path in pathlib.Path(sound_dir).glob("*.mp3")]
        random.shuffle(self._sound_list)
        self._count = 0

    def current(self) -> str:
        """音源へのパスを返す

        Returns:
            str: 音源のパス
        """
        return self._sound_list[self._count]

    def next(self) -> None:
        """次の音源へ進める
        """
        self._count += 1
        if self._count >= len(self._sound_list):
            self._count = 0


def execute(filename: str, loop: int, wait_sec: float) -> None:
    """音を鳴らす

    Args:
        filename (str): 音源へのパス
    """
    mixer.music.load(filename)
    mixer.music.play(loop)
    time.sleep(wait_sec)
    mixer.music.stop()


def initialize() -> None:
    """音環境を初期化する
    """
    mixer.init()
