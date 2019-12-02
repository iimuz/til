# default packages
import pathlib
import unittest
from datetime import datetime

# third party
import pandas as pd
from scipy import io


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.DATA_DIR = pathlib.Path("data")
        self.EXAMPLE_SENSOR = self.DATA_DIR.joinpath(
            "hs_bearing_1/sensor-20130307T015746Z.mat"
        )
        self.EXAMPLE_TACH = self.DATA_DIR.joinpath(
            "hs_bearing_1/tach-20130307T015746Z.mat"
        )

    def test_load(self) -> None:
        """データを読み込んでみる。

        Note:
            データセットは、1日に一度 6 秒間だけ 97656 Hz で観測したデータである。
        """
        var = io.loadmat(str(self.EXAMPLE_SENSOR))
        self.assertEqual(
            b"MATLAB 5.0 MAT-file, Platform: MACI, Created on: Tue May 28 16:09:11 2013",
            var["__header__"],
        )
        self.assertEqual(97656 * 6, len(var["v"]))

    def test_sensor_convert_to_df(self) -> None:
        """センサデータを読み込み pandas.DataFrame 化
        """
        # load data
        var = io.loadmat(str(self.EXAMPLE_SENSOR))
        date_val = datetime.strptime(
            self.EXAMPLE_SENSOR.stem.split("-")[-1], "%Y%m%dT%H%M%SZ"
        )

        # convert to pd.DataFrame
        df = pd.DataFrame(var["v"], columns=["vibration"])
        df["date"] = date_val

        # check
        self.assertEqual(97656 * 6, len(df))

    def test_tach_convert_to_df(self) -> None:
        """tachデータを読み込み pandas.DataFrame 化
        """
        # load data
        var = io.loadmat(str(self.EXAMPLE_TACH))
        date_val = datetime.strptime(
            self.EXAMPLE_TACH.stem.split("-")[-1], "%Y%m%dT%H%M%SZ"
        )

        # convert to pd.DataFrame
        df = pd.DataFrame(var["tach"], columns=["tach"])
        df["date"] = date_val

        # check
        self.assertEqual(2446, len(df))


if __name__ == "__main__":
    unittest.main()
