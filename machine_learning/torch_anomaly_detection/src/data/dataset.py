"""データセットのダウンロードなどファイルの処理を実施するモジュール."""
# default package
import abc
import dataclasses
import enum
import logging
import pathlib
import tarfile
import typing as t
import urllib.request as request

# third party packages
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch.utils.data as torch_data
import tqdm as tqdm_std

# my packaegs
import src.data.directories as directories
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class Dataset(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.path = directories.get_raw().joinpath(self.name)
        self.train = pd.DataFrame()
        self.valid = pd.DataFrame()
        self.test = pd.DataFrame()

    def load(self) -> "Dataset":
        self.load_dataset()

        return self

    @abc.abstractmethod
    def load_dataset(self) -> None:
        raise NotImplementedError

    def save(self, reprocess: bool = False) -> "Dataset":
        self.save_dataset(reprocess)

        return self

    @abc.abstractmethod
    def save_dataset(self, reprocess: bool) -> None:
        raise NotImplementedError


def _main() -> None:
    """実行用スクリプト."""
    ut.init_root_logger()

    config = utils.load_config_from_input_args(lambda x: Config(**x))
    if config is None:
        _logger.error("config error.")
        return

    filepath = download(config)
    _logger.info(f"download path: {filepath}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        _logger.exception(e)
