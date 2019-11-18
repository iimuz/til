# default
import unittest

# third party
from scipy import io

# my pckages
import download


class TestDwonload(unittest.TestCase):
    """対象のデータセットをダウンロードするテスト。
    """

    def test_download(self):
        download.get_file()
        var = io.loadmat("data/hs_bearing_1/hs_bearing_1/sensor-20130307T015746Z.mat")
        for key, item in var.items():
            print(key, item)
