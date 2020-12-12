"""Food101データセット."""
# default packages
import logging
import tarfile
import urllib.request as request

# third party packages
import tqdm.autonotebook as tqdm

# my packages
import src.data.dataset as dataset
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class Food101(dataset.BaseDataset):
    """ローカルのFood101データセットを管理するクラス."""

    def create_dataset(self, pbar: tqdm.tqdm, **kwargs) -> None:
        tarpath = self.path.joinpath("food-101.tar.gz")
        if not tarpath.exists():
            url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
            with ut.TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=url) as pbar:
                request.urlretrieve(
                    url,
                    filename=tarpath,
                    reporthook=pbar.update_to,
                    data=None,
                )

        expand_paqth = self.path.joinpath("food-101")
        if not expand_paqth.exists():
            with tarfile.open(tarpath) as tar:
                if any(
                    [
                        path.name[0] == "/" or path.name[0:2] == ".."
                        for path in tar.getmembers()
                    ]
                ):
                    # 本当はnot foundではないので修正が必要
                    raise FileNotFoundError
                tar.extractall(self.path)

    def load_dataset(self, pbar: tqdm.tqdm, **kwargs) -> None:
        pass


def main() -> None:
    """データセットのダウンロードを実行するスクリプト."""
    Food101().create()


if __name__ == "__main__":
    try:
        ut.init_root_logger(logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
