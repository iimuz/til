"""Food101データセット."""
# default packages
import json
import logging
import tarfile
import urllib.request as request

# third party packages
import pandas as pd
import tqdm.autonotebook as tqdm

# my packages
import src.data.dataset as dataset
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class Food101(dataset.BaseDataset):
    """ローカルのFood101データセットを管理するクラス."""

    def __init__(self, mode: dataset.Mode) -> None:
        super().__init__(mode=mode)

        self.primary_keys = ["name"]

        self._path_expand = self.path.joinpath("food-101")
        self._path_images = self._path_expand.joinpath("images")
        self._path_meta = self._path_expand.joinpath("meta")
        self._path_classes = self._path_meta.joinpath("classes.txt")
        self._path_test = self._path_meta.joinpath("test.json")
        self._path_train = self._path_meta.joinpath("train.json")

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
        path_list = (
            self._path_train if self.mode == dataset.Mode.TRAIN else self._path_test
        )
        with open(path_list) as f:
            filelist = json.load(f)
        self.data_ = pd.DataFrame(
            [
                {
                    "name": path,
                    "class": key,
                }
                for key, files in filelist.items()
                for path in files
            ]
        )
        self.data_["path"] = self.data_["name"].apply(
            lambda x: self._path_images.joinpath(x).resolve()
        )

        with open(self._path_classes) as f:
            self.classes_ = set([v.rstrip() for v in f.readlines()])


def main() -> None:
    """データセットのダウンロードを実行するスクリプト."""
    target = Food101(dataset.Mode.TRAIN).create().load()
    _logger.info(target.data_)


if __name__ == "__main__":
    try:
        ut.init_root_logger(logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
