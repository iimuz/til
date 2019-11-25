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
    """URLからファイルをダウンロードします

    Args:
        url (str): ダウンロードするファイルのURL
        download_dir (str): ダウンロードしたファイルを保存する先
    """
    filename = url.split("/")[-1]
    download = pathlib.Path(download_dir).joinpath(filename)

    if download.exists() is False:
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            request.urlretrieve(
                url, filename=str(download), reporthook=t.update_to, data=None
            )


def unzip(filepath: str, extract_dir: str) -> None:
    """zipファイルを解凍します

    Args:
        filepath (str): zipファイルのパス
        extract_dir (str): 解凍したファイルを保存するディレクトリ
    """
    with zipfile.ZipFile(filepath) as zfile:
        zfile.extractall(extract_dir)
