"""Food101データセット."""
# default packages
import json
import logging
import tarfile
import urllib.request as request
import typing as t

# third party packages
import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchvision.io as torchvision_io
import torchvision.transforms as transforms
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

        self._ratio_train = 0.8
        self._random_state = 42

    def create_dataset(self, pbar: tqdm.tqdm, **kwargs) -> None:
        tarpath = self.path.joinpath("food-101.tar.gz")
        if not tarpath.exists():
            url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
            self.path.mkdir(exist_ok=True)
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
            lambda x: self._path_images.joinpath(f"{x}.jpg").resolve()
        )

        if self.mode != dataset.Mode.TEST:
            ratio = (
                self._ratio_train
                if self.mode == dataset.Mode.TRAIN
                else 1.0 - self._ratio_train
            )
            train, valid = model_selection.train_test_split(
                self.data_,
                train_size=ratio,
                stratify=self.data_["class"],
                random_state=self._random_state,
            )
            self.data_ = train if self.mode == dataset.Mode.TRAIN else valid
        self.data_.sort_values(by=self.primary_keys, inplace=True, ignore_index=True)

        with open(self._path_classes) as f:
            self.classes_ = sorted(list([v.rstrip() for v in f.readlines()]))
        self.label_encoder_ = preprocessing.LabelEncoder().fit(self.classes_)


class Food101WithLabel(torch_data.Dataset):
    """PyTorch用ラベル付きFood101データセット."""

    def __init__(
        self, transforms: t.Optional[nn.Sequential], mode: dataset.Mode
    ) -> None:
        super().__init__()

        self.data = Food101(mode=mode).load()
        self.transforms = transforms

    def __getitem__(self, idx: int) -> t.Tuple[torch.Tensor, int]:
        target = self.data.data_.iloc[0]
        img = torchvision_io.read_image(str(target["path"]))
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.data.label_encoder_.transform([target["class"]])[0]

        return img, label

    def __len__(self) -> int:
        return self.data.data_.shape[0]


class Food101WithLableModule(pl.LightningDataModule):
    """PyTorchLighting用ラベル付きFood101データセット."""

    def __init__(self, batch_size: int = 144, num_workers: int = 4) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = (224, 224)

        self.transforms_train = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.image_size),
            transforms.ConvertImageDtype(torch.float32),
        )
        self.transforms_test = nn.Sequential(
            transforms.Resize(self.image_size),
            transforms.ConvertImageDtype(torch.float32),
        )

    def prepare_data(self, *args, **kwargs):
        Food101(dataset.Mode.TRAIN).create()

    def setup(self, stage: t.Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ = Food101WithLabel(
                self.transforms_train, mode=dataset.Mode.TRAIN
            )
            self.valid_ = Food101WithLabel(
                self.transforms_test, mode=dataset.Mode.VALID
            )

        if stage == "test" or stage is None:
            self.test_ = Food101WithLabel(self.transforms_test, mode=dataset.Mode.TEST)

    def train_dataloader(self, *args, **kwargs) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.train_,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )

    def val_dataloader(self, *args, **kwargs) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.valid_,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )

    def test_dataloader(self, *args, **kwargs) -> torch_data.DataLoader:
        return torch_data.DataLoader(
            self.test_,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )


def main() -> None:
    """データセットのダウンロードを実行するスクリプト."""
    with ut.timer(name="create dataset", print_log=_logger.info):
        Food101(dataset.Mode.TRAIN).create().load()

    batch_size = 64
    num_workers = 4
    device = "cpu"
    transform = nn.Sequential(transforms.RandomCrop(224)).to(torch.device(device))
    dataloader = torch_data.DataLoader(
        Food101WithLabel(transform, mode=dataset.Mode.TRAIN),
        batch_size=batch_size,
        num_workers=num_workers,
    )
    with ut.timer(name="load batch", print_log=_logger.info):
        for batch, labels in dataloader:
            batch.to(torch.device(device))
            break


if __name__ == "__main__":
    try:
        ut.init_root_logger(logging.INFO)
        main()
    except Exception as e:
        _logger.exception(e)
