import pygame.mixer as mixer
import RPi.GPIO as GPIO
import time


def exec_sound() -> None:
    """ 音を鳴らす
    """
    mixer.music.play(2)
    time.sleep(3)
    mixer.music.stop()


def initialize_sensor() -> None:
    """ センサの初期化を行う
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(4, GPIO.IN)


def initialize_sound() -> None:
    """ 音環境の初期化を行う
    """
    mixer.init()
    mixer.music.load("hoge.mp3")


def run() -> None:
    """ センサの値を検知して音を鳴らす
    """
    try:
        while True:
            time.sleep(0.1)
            if GPIO.input(4) != 0:
                print("start sound.")
                exec_sound()
                print("wait sensor data...")
    except KeyboardInterrupt:
        print("end")
        return


if __name__ == "__main__":
    initialize_sound()
    initialize_sensor()
    run()
