import RPi.GPIO as GPIO
import time


def initialize(index: int) -> None:
    """センサの初期化を行う

    Args:
        index (int): センサ番号
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(index, GPIO.IN)


def wait_until_balancing(index: int, wait_sec: float) -> None:
    """センサが安定(動作していないこと)するまで待機する

    Args:
        index (int): センサ番号
        wait_sec (float): センサ情報を取得する間隔
    """
    while GPIO.input(index) == 1:
        time.sleep(wait_sec)


def wait_until_moving(index: int, wait_sec: float) -> None:
    """センサが動作を感知するまで待機する

    Args:
        index (int): センサ番号
        wait_sec (float): センサ情報を取得する間隔
    """
    while GPIO.input(index) == 0:
        time.sleep(wait_sec)
