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


def get_file(url: str, download_dir: str) -> None:
    ARCHIVE_FILE = pathlib.Path(download_dir).joinpath(url.split("/")[-1])
    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=ARCHIVE_FILE.name) as t:
        request.urlretrieve(
            url, filename=str(ARCHIVE_FILE), reporthook=t.update_to, data=None
        )


def extract(filepath: str, expand_dir: str) -> None:
    with zipfile.ZipFile(filepath) as zfile:
        zfile.extractall(expand_dir)
