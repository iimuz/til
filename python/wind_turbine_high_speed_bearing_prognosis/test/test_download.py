# default
import unittest

# my pckages
import download


class TestDwonload(unittest.TestCase):
    """対象のデータセットをダウンロードするテスト。
    """

    def test_download(self):
        download.get_file()
