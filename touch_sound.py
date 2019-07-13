import pathlib
import pygame.mixer as mixer
import random
import RPi.GPIO as GPIO
import time
from typing import List


def create_sound_list(sound_dir: str) -> List[str]:
    """ 音声リストを生成する
    """
    sound_list = pathlib.Path(sound_dir).glob("*.mp3")
    sound_list = [str(path) for path in sound_list]
    random.shuffle(sound_list)
    return sound_list


def exec_sound(filename: str) -> None:
    """ 音を鳴らす
    """
    mixer.music.load(filename)
    mixer.music.play(1)
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


def run(sound_list: List[str]) -> None:
    """ センサの値を検知して音を鳴らす
    """
    try:
        sound_count = 0
        is_continuous = False
        while True:
            time.sleep(0.1)
            sensor_data = GPIO.input(4)
            if sensor_data != 0 and is_continuous == False:
                sound_file = sound_list[sound_count]
                print("start sound: " + sound_file)
                exec_sound(sound_file)

                sound_count = sound_count + 1
                if sound_count >= len(sound_list):
                    sound_count = 0
                is_continuous = True
                print("wait sensor data...")
            if is_continuous and sensor_data == 0:
                print("sensor is valid.")
                is_continuous = False
    except KeyboardInterrupt:
        print("end")
        return


if __name__ == "__main__":
    sound_list = create_sound_list("_test/sound")
    initialize_sound()
    initialize_sensor()
    run(sound_list)
