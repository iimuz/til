import unittest
import zipfile


class TestZip(unittest.TestCase):
    def test_zip(self) -> None:
        """指定したファイルを zip ファイルに圧縮する。
        """
        ZIP_FILEPATH = "temp/new.zip"

        # close した時点で書き込まれる。
        with zipfile.ZipFile(
            str(ZIP_FILEPATH), "w", compression=zipfile.ZIP_DEFLATED
        ) as zfile:
            zfile.write("data/file_a.txt")
            zfile.write("data/file_b.txt")

    def test_unzip(self) -> None:
        """zipファイル中の全てのデータを解凍
        """
        ZIP_FILEPATH = "temp/new.zip"
        EXPAND_DIR = "temp/expand"

        with zipfile.ZipFile(str(ZIP_FILEPATH)) as zfile:
            zfile.extractall(EXPAND_DIR)
