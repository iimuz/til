# default
import pathlib
import unittest

# my pckages
import download


class TestDwonload(unittest.TestCase):
    """対象のデータセットをダウンロードするテスト。
    """

    def test_download(self) -> None:
        URL = "http://data-acoustics.com/wp-content/uploads/2014/06/hs_bearing_1.zip"
        EXPAND_DIR = pathlib.Path("data")
        ARCHIVE_FILE = EXPAND_DIR.joinpath(URL.split("/")[-1])

        if ARCHIVE_FILE.exists() is False:
            download.get_file(URL, EXPAND_DIR)
        download.extract(ARCHIVE_FILE, EXPAND_DIR)
