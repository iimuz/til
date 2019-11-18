# default packages
import pathlib
import zipfile
from urllib import request

# third party
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`.

    Args:
        tqdm (tqdm): tqdm
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """ update function
        Args:
            b (int, optional): Number of blocks transferred. Defaults to 1.
            bsize (int, optional): Size of each block (in tqdm units). Defaults to 1.
            tsize ([type], optional): Total size (in tqdm units). Defaults to None.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_file() -> None:
    URL = "http://data-acoustics.com/wp-content/uploads/2014/06/hs_bearing_1.zip"
    DOWNLOAD = pathlib.Path("data/hs_bearing_1.zip")
    EXTRACT_DIR = pathlib.Path("data/hs_bearing_1")

    if DOWNLOAD.exists() is False:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=URL.split("/")[-1]
        ) as t:
            request.urlretrieve(
                URL, filename=str(DOWNLOAD), reporthook=t.update_to, data=None
            )

    with zipfile.ZipFile(str(DOWNLOAD)) as zfile:
        zfile.extractall(str(EXTRACT_DIR))
