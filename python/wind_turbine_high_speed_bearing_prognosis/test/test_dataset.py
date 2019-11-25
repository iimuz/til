# default packages
import pathlib
import unittest

# third party
from scipy import io


class TestDataset(unittest.TestCase):
    def test_load(self) -> None:
        DATA_DIR = pathlib.Path("data")
        MATFILE = DATA_DIR.joinpath("hs_bearing_1/sensor-20130307T015746Z.mat")

        var = io.loadmat(str(MATFILE))
        for key, item in var.items():
            print(key, item)

    def test_convert_df(self) -> None:
        pass
