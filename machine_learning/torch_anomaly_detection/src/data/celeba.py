"""CelebA Dataset.

Notes:
    - `http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`
"""
# default packages
import logging
import pathlib
import shutil
import typing as t
import zipfile

# third party packages
import pandas as pd
import requests
import tqdm as tqdm_std

# my packages
import src.data.dataset as dataset
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class Celeba(dataset.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.archive_file = self.path.joinpath("img_align_celeba.zip")
        self.datadir = self.path.joinpath("img_align_celeba")
        self.train_list = self.path.joinpath("train.csv")
        self.valid_list = self.path.joinpath("valid.csv")

    def save_dataset(self, reprocess: bool) -> None:
        if reprocess:
            _logger.info("=== reporcess mode. delete existing data.")
            shutil.rmtree(self.path)

        self.path.mkdir(exist_ok=True)

        _logger.info("=== download zip file.")
        if not self.archive_file.exists():
            _download(self.archive_file)

        _logger.info("=== unzip.")
        if not self.datadir.exists():
            with zipfile.ZipFile(str(self.archive_file)) as z:
                z.extractall(str(self.path))

        _logger.info("=== create train and valid file list.")
        if not self.train_list.exists() and not self.valid_list.exists():
            filelist = sorted(
                [p.relative_to(self.path) for p in self.path.glob("**/*.jpg")]
            )
            train_ratio = 0.8
            train_num = int(len(filelist) * train_ratio)

            if not self.train_list.exists():
                train_list = pd.DataFrame({"filepath": filelist[:train_num]})
                train_list.to_csv(self.train_list, index=False)

            if not self.valid_list.exists():
                valid_list = pd.DataFrame({"filepath": filelist[train_num:]})
                valid_list.to_csv(self.valid_list, index=False)

    def load_dataset(self) -> None:
        self.train = pd.read_csv(self.train_list)
        self.valid = pd.read_csv(self.valid_list)


def _download(filepath: pathlib.Path, chunksize: int = 32768) -> None:
    """Download CelebA Dataset.

    Args:
        filepath (pathlib.Path): ダウンロードしたファイルを置くファイルパス.
        chunksize (int, optional): ダウンロードのチャンクサイズ. Defaults to 32768.

    Notes:
        - reference:
            `https://gist.github.com/charlesreid1/4f3d676b33b95fce83af08e4ec261822`
    """
    URL = "https://docs.google.com/uc?export=download"
    ID = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"

    with requests.Session() as session:
        params: t.Dict[str, t.Any] = dict(id=ID)
        response = session.get(URL, params=params, stream=True)

        params["confirm"] = _get_confirm_token(response)
        response = session.get(URL, params=params, stream=True)

        _save_response_content(response, filepath, chunksize)


def _get_confirm_token(response: requests.Response) -> t.Optional[str]:
    """トークンを生成する.

    Args:
        response (requests.Response): 取得する先のレスポンス.

    Returns:
        t.Optional[str]: トークン.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(
    response: requests.Response, filepath: pathlib.Path, chunksize: int = 32768,
) -> None:
    """レスポンス内容をファイルとして保存する.

    Args:
        response (requests.Response): レスポンス.
        filepath (pathlib.Path): 保存先のファイルパス.
        chunksize (int, optional): ダウンロードするチャンクサイズ. Defaults to 32768.
    """
    with open(str(filepath), "wb") as f:
        for chunk in tqdm_std.tqdm(response.iter_content(chunksize)):
            if chunk:
                f.write(chunk)


def main() -> None:
    """Celeba データセットをダウンロードし、学習及びテスト用のファイルリストを生成する."""
    celeba = Celeba()
    celeba.save()


if __name__ == "__main__":
    try:
        ut.init_root_logger()
        main()
    except Exception as e:
        _logger.exception(e)
