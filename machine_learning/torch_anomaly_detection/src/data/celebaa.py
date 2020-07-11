"""CelebA Dataset.

Notes:
    - `http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`
"""
# default packages
import enum
import logging
import pathlib
import traceback
import typing as t
import zipfile

# third party packages
import numpy as np
import PIL.Image as Image
import requests
import torch.utils.data as torch_data
import tqdm as tqdm_std

# my packages
import src.data.directories as directories
import src.data.log_utils as log_utils

# logger
logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    """データセットの出力モード."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Dataset(torch_data.Dataset):
    def __init__(
        self,
        filelist: t.List[pathlib.Path],
        transform: t.Optional[t.Any],
        mode: Mode = Mode.TRAIN,
    ) -> None:
        super(Dataset, self).__init__()

        self.filelist = filelist
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


def download(filepath: pathlib.Path, chunksize: int = 32768) -> None:
    """Download CelebaA Dataset.

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

        params["confirm"] = get_confirm_token(response)
        response = session.get(URL, params=params, stream=True)

        save_response_content(response, filepath, chunksize)


def extract_all(zippath: pathlib.Path, outdir: pathlib.Path) -> None:
    with zipfile.ZipFile(str(zippath)) as z:
        z.extractall(str(outdir))


def get_confirm_token(response: requests.Response) -> t.Optional[str]:
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


def save_response_content(
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


def _main() -> None:
    log_utils.init_root_logger()

    filepath = directories.get_raw().joinpath("img_align_celeba.zip")
    extractpath = directories.get_raw()
    if filepath.exists() is False:
        download(filepath)
    extract_all(filepath, extractpath)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
