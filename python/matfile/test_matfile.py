# default package
import pathlib
import unittest

# thrid party
from scipy import io

# my packages
import download


class TestMatfile(unittest.TestCase):
    def test_matfile(self):
        url = "http://data-acoustics.com/wp-content/uploads/2014/06/hs_bearing_1.zip"
        extract_dir = pathlib.Path("data")
        archive_path = extract_dir.joinpath(url.split("/")[-1])
        matfile = extract_dir.joinpath(archive_path.stem).joinpath(
            "sensor-20130307T015746Z.mat"
        )

        # download mat file
        if archive_path.exists() is False:
            download.get_file(url, str(extract_dir))
        if matfile.exists() is False:
            download.unzip(str(archive_path), str(extract_dir))

        # load mat file
        var = io.loadmat(str(matfile))
        for key, item in var.items():
            print(key, item)
