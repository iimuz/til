"""Dataset downloader."""
# default packages
import argparse
import dataclasses
import logging
import pathlib
import zipfile
from logging import getLogger
from urllib import request

# my packages
from tqdmupto import TqdmUpTo

# logger
logger = getLogger(__name__)


@dataclasses.dataclass
class Config:
    name: str = "dataset-name"
    save_dir: str = "/path/to/save/dataset/directory"


def download(name: str, download_dir: pathlib.Path) -> pathlib.Path:
    base_url = "http://www.timeseriesclassification.com/Downloads/"

    filename = f"{name}.zip"
    target_url = f"{base_url}{filename}"
    download_dir.mkdir(exist_ok=True)
    download_path = download_dir.joinpath(filename)

    if download_path.exists():
        logger.info(f"file already exits: {download_path}")
        return download_path

    with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        request.urlretrieve(
            target_url, filename=str(download_path), reporthook=t.update_to
        )

    return download_path


def unzip(filepath: pathlib.Path) -> pathlib.Path:
    if filepath.exists() is False:
        logger.error(f"file does not exists: {filepath}")

    unzip_dir = filepath.parent.joinpath(filepath.stem)
    unzip_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(str(filepath)) as zfile:
        zfile.extractall(str(unzip_dir))

    return unzip_dir


def _argparse() -> Config:
    parser = argparse.ArgumentParser(description="Download dataset.")
    parser.add_argument("name", help="dataset name", default="CBF")
    parser.add_argument("--save-dir", help="save directory", default="_data/raw")
    args = parser.parse_args()

    config = Config(**vars(args))

    return config


def _main() -> None:
    logging.basicConfig(level=logging.INFO)

    config = _argparse()
    logger.info(f"dataset name: {config.name}")
    logger.info(f"download dir: {config.save_dir}")
    download_path = download(config.name, pathlib.Path(config.save_dir))
    unzip(download_path)


if __name__ == "__main__":
    _main()
