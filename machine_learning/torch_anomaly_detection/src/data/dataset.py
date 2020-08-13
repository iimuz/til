"""データセットのダウンロードなどファイルの処理を実施するモジュール."""
# default package
import dataclasses
import enum
import logging
import pathlib
import tarfile
import traceback
import typing as t
import urllib.request as request

# third party packages
import numpy as np
import PIL.Image as Image
import torch.utils.data as torch_data
import tqdm as tqdm_std

# my packaegs
import src.data.directories as directories
import src.data.log_utils as log_utils
import src.data.utils as utils

# logger
logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    """データセットの出力モード."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class ImageDataset(torch_data.Dataset):
    """MVTecAd  用データセット"""

    def __init__(
        self, filelist_path: str, transform: t.Optional[t.Any], mode: Mode = Mode.TRAIN,
    ) -> None:
        self.filelist = filelist_path
        self.transform = transform
        self.mode = mode

        self.type = self.mode.value

    def __getitem__(self, idx):
        img = Image.open(self.filelist[idx])

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = np.array(img).transpose(0, 3, 1, 2)

        return img

    def __len__(self) -> int:
        return len(self.filelist)


@dataclasses.dataclass
class Config:
    """設定値ファイルの読み込み用クラス."""

    dataset_name: str = "MVTec_Hazelnut"
    download_dir: t.Optional[str] = None


class DatasetName(enum.Enum):
    """データセットを指定するための設定値."""

    MVTEC_HAZELNUT = "MVTec_Hazelnut"

    @classmethod
    def value_of(cls, name: str) -> "DatasetName":
        """設定値の文字列から Enum 値を返す.

        Raises:
            ValueError: 指定した文字列が設定値にない場合

        Returns:
            [type]: Enum の値
        """
        for e in DatasetName:
            if e.value == name:
                return e

        raise ValueError(f"invalid value: {name}")


class TqdmUpTo(tqdm_std.tqdm):
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


def download(config: Config) -> pathlib.Path:
    """データセットをダウンロードして、展開する.

    Args:
        config (Config): ダウンロードに必要な設定値

    Returns:
        pathlib.Path: ダウンロードしたフォルダ
    """
    url = get_dataset_url(DatasetName.value_of(config.dataset_name))

    filename = url.split("/")[-1]
    download_dir = (
        pathlib.Path(config.download_dir)
        if config.download_dir is not None
        else directories.get_raw()
    )
    filepath = download_dir.joinpath(filename)
    if filepath.exists():
        logger.info(f"file already exists: {filepath}")
    else:
        filepath.parent.mkdir(exist_ok=True, parents=True)
        with TqdmUpTo(
            unit="B", unit_scale=True, miniters=1, desc=filepath.name
        ) as pbar:
            request.urlretrieve(
                url, filename=filepath, reporthook=pbar.update_to, data=None
            )

    with tarfile.open(filepath, "r") as tar:
        tar.extractall(download_dir)

    extracted_dir = download_dir.joinpath(filepath.name.split(".")[0])
    return extracted_dir


def get_dataset_url(name: DatasetName) -> str:
    """データセットをダウンロードするための URL を取得する.

    Args:
        name (DatasetName): データセット名

    Returns:
        str: データセットの URL
    """
    mvtec_ad = "ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection"
    url = ""

    if name == DatasetName.MVTEC_HAZELNUT:
        url = mvtec_ad + "/hazelnut.tar.xz"

    return url


def _main() -> None:
    """実行用スクリプト."""
    log_utils.init_root_logger()

    config = utils.load_config_from_input_args(lambda x: Config(**x))
    if config is None:
        logger.error("config error.")
        return

    filepath = download(config)
    logger.info(f"download path: {filepath}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
