"""CelebA Dataset.

Notes:
    - `http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`
"""
# default packages
import csv
import logging
import pathlib
import traceback
import typing as t
import zipfile

# third party packages
import requests
import tqdm as tqdm_std

# my packages
import src.data.directories as directories
import src.data.log_utils as log_utils

# logger
logger = logging.getLogger(__name__)


def download(filepath: pathlib.Path, chunksize: int = 32768) -> None:
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

        params["confirm"] = get_confirm_token(response)
        response = session.get(URL, params=params, stream=True)

        save_response_content(response, filepath, chunksize)


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


def main() -> None:
    """CelebA データセットをダウンロードし、学習及びテスト用のファイルリストを生成する."""
    log_utils.init_root_logger()

    logger.info("=== download and extract files.")
    filepath = directories.get_raw().joinpath("img_align_celeba.zip")
    if filepath.exists() is False:
        download(filepath)

    logger.info("=== unzip.")
    extractpath = directories.get_raw()
    with zipfile.ZipFile(str(filepath)) as z:
        z.extractall(str(extractpath))

    logger.info("=== create train and valid file list.")
    filelist = sorted(
        [p.relative_to(extractpath) for p in extractpath.glob("**/*.jpg")]
    )
    train_num = int(len(filelist) * 0.8)
    train_list = filelist[:train_num]
    valid_list = filelist[train_num:]

    train_path = directories.get_interim().joinpath("celeba_train.csv")
    with open(str(train_path), "w") as ft:
        writer = csv.writer(ft)
        writer.writerows([[p] for p in train_list])

    valid_path = directories.get_interim().joinpath("celeba_valid.csv")
    with open(str(valid_path), "w") as fv:
        writer = csv.writer(fv)
        writer.writerows([[p] for p in valid_list])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
