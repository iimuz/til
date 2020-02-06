# default packages
import os
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


def _main() -> None:
    """urlretriveの進捗表示にtqdmを利用する。

    Note:
        - 参考文献: tqdm – PythonとCLIの高速で拡張できるプログレスバー: `https://githubja.com/tqdm/tqdm`
    """
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=URL.split("/")[-1]) as t:
        request.urlretrieve(URL, filename=os.devnull, reporthook=t.update_to, data=None)


if __name__ == "__main__":
    _main()
