"""データセットの基本モジュール."""
# default packages
import abc
import enum
import functools
import logging
import pathlib
import typing as t

# third party packages
import tqdm.autonotebook as tqdm

# logger
_logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    """実行モード."""

    TRAIN: str = "train"
    VALID: str = "validation"
    TEST: str = "test"


class BaseDataset(metaclass=abc.ABCMeta):
    """データセットのベースとなるクラス."""

    def __init__(self, mode: Mode = Mode.TRAIN) -> None:
        self.name = self.__class__.__name__
        self.path = pathlib.Path("data/raw").joinpath(self.name)

        self.mode = mode

        self.primary_keys: t.List[str] = list()
        self.info_keys: t.List[str] = list()

    def create(
        self, pbar: tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True), **kwargs
    ) -> "BaseDataset":
        """データセットをダウンロードするなどによりローカルに作成する

        Returns:
            BaseDatase: 作り出したデータセット自身
        """
        self.create_dataset(pbar=pbar, **kwargs)

        return self

    @abc.abstractmethod
    def create_dataset(self, pbar: tqdm.tqdm, **kwargs) -> None:
        """実際にデータセットを作成する関数."""
        raise NotImplementedError

    def load(
        self, pbar: tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True), **kwargs
    ) -> "BaseDataset":
        """ローカルに作成済みのデータセットを読み込むクラス

        Returns:
            BaseDataset: 読み込んだデータセット自身
        """
        self.load_dataset(pbar=pbar, **kwargs)

        return self

    @abc.abstractmethod
    def load_dataset(self, pbar: tqdm.tqdm, **kwargs) -> None:
        """実際にデータセットを作成する関数."""
        raise NotImplementedError
