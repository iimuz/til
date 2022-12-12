"""MVTecAd Dataset."""
# default packages
import dataclasses as dc
import enum
import logging
import pathlib
import shutil
import sys
import tarfile
import typing as t
import urllib.request as request

# third party packages
import pandas as pd

# my packages
import src.data.dataset as ds
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class Kind(enum.Enum):
    HAZELNUT = "hazelnut"

    @classmethod
    def value_of(cls, name: str) -> "Kind":
        """設定値の文字列から Enum 値を返す.

        Raises:
            ValueError: 指定した文字列が設定値にない場合

        Returns:
            [type]: Enum の値
        """
        for e in Kind:
            if e.value == name:
                return e

        raise ValueError(f"invalid value: {name}")


class MVTecAd(ds.Dataset):
    def __init__(self, kind: Kind) -> None:
        super().__init__()
        archive, datadir = _get_archive_file_name(kind)
        self.archive_file = self.path.joinpath(archive)
        self.datadir = self.path.joinpath(datadir)
        self.train_list = self.path.joinpath(f"{datadir}_train.csv")
        self.valid_list = self.path.joinpath(f"{datadir}_valid.csv")
        self.test_list = self.path.joinpath(f"{datadir}_test.csv")

    def save_dataset(self, reprocess: bool) -> None:
        if reprocess:
            _logger.info("=== reporcess mode. delete existing data.")
            self.archive_file.unlink()
            shutil.rmtree(self.datadir)
            self.train_list.unlink()
            self.valid_list.unlink()
            self.test_list.unlink()

        self.path.mkdir(exist_ok=True)

        if not self.datadir.exists():
            if not self.archive_file.exists():
                _logger.info("=== download zip file.")
                _download(self.archive_file)

            _logger.info("=== extract all.")
            with tarfile.open(self.archive_file, "r") as tar:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, self.path)

        if not self.train_list.exists() and not self.valid_list.exists():
            _logger.info("=== create train and valid file list.")
            filelist = sorted(
                [p.relative_to(self.path) for p in self.datadir.glob("train/**/*.png")]
            )
            train_ratio = 0.8
            train_num = int(len(filelist) * train_ratio)

            if not self.train_list.exists():
                train_list = pd.DataFrame({"filepath": filelist[:train_num]})
                train_list.to_csv(self.train_list, index=False)

            if not self.valid_list.exists():
                valid_list = pd.DataFrame({"filepath": filelist[train_num:]})
                valid_list.to_csv(self.valid_list, index=False)

        if not self.test_list.exists():
            _logger.info("=== create test file list.")
            filelist = sorted(
                [p.relative_to(self.path) for p in self.datadir.glob("test/**/*.png")]
            )
            test_list = pd.DataFrame({"filepath": filelist})
            test_list.to_csv(self.test_list, index=False)

    def load_dataset(self) -> None:
        self.train = pd.read_csv(self.train_list)
        self.valid = pd.read_csv(self.valid_list)
        self.test = pd.read_csv(self.test_list)


@dc.dataclass
class Config:
    kind: str = "hazelnut"


def main(config: Config) -> None:
    """MVTecAD データセットをダウンロードし、学習及びテスト用のファイルリストを生成する."""
    kind = Kind.value_of(config.kind)
    mvtec = MVTecAd(kind)
    mvtec.save()


def _download(filepath: pathlib.Path) -> None:
    mvtec_ad = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection"
    url = mvtec_ad + "/" + filepath.name
    _logger.info(mvtec_ad)
    _logger.info(url)

    with ut.TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=filepath.name) as pbar:
        request.urlretrieve(
            url, filename=filepath, reporthook=pbar.update_to, data=None
        )


def _get_archive_file_name(kind: Kind) -> t.Tuple[str, str]:
    series = {Kind.HAZELNUT: ("hazelnut.tar.xz", "hazelnut")}

    if kind not in series.keys():
        raise NotImplementedError(f"unknown type: {kind}")

    return series[kind]


if __name__ == "__main__":
    try:
        ut.init_root_logger()

        if len(sys.argv) == 1:
            config = Config()
        elif len(sys.argv) == 2:
            config = Config()
        else:
            raise Exception(
                "input arguments error."
                " usage: python path/to/script.py"
                " or python path/to/script.py path/to/config.yml"
            )

        main(config)
    except Exception as e:
        _logger.exception(e)
