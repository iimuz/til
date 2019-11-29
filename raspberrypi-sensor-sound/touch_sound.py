import sensor
import sound


def run() -> None:
    """センサの感知を受けて音をならす
    """
    SOUND_LOOP_NUM = 1
    SOUND_TIME_SEC = 3.0
    sound.initialize()
    sound_list = sound.ShuffleList("_test/sound")

    SENSOR_INDEX = 4
    WAIT_SEC = 0.1
    sensor.initialize(SENSOR_INDEX)

    try:
        while True:
            print("wait until balancing...")
            sensor.wait_until_balancing(SENSOR_INDEX, WAIT_SEC)

            print("wait until moving...")
            sensor.wait_until_moving(SENSOR_INDEX, WAIT_SEC)

            print("start sound: " + sound_list.current())
            sound.execute(sound_list.current(), SOUND_LOOP_NUM, SOUND_TIME_SEC)
            sound_list.next()
    except KeyboardInterrupt:
        print("end")
        return


if __name__ == "__main__":
    run()
