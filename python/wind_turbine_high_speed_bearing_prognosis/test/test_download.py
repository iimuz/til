# default
import pathlib
import unittest

# my pckages
import download


class TestDwonload(unittest.TestCase):
    """対象のデータセットをダウンロードするテスト。
    """

    def test_download(self) -> None:
        """データを一つ分ロードし解凍する。
        """
        NUM_ARCHIVE = 5
        URL_BASE = "http://data-acoustics.com/wp-content/uploads/2014/06/"
        EXPAND_DIR = pathlib.Path("data")

        for idx in range(1, NUM_ARCHIVE + 1):
            url = URL_BASE + f"hs_bearing_{idx}.zip"
            archive_file = EXPAND_DIR.joinpath(url.split("/")[-1])
            if archive_file.exists() is False:
                download.get_file(url, str(EXPAND_DIR))
            download.extract(str(archive_file), str(EXPAND_DIR))


if __name__ == "__main__":
    unittest.main()
