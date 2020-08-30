"""PyTorch Lightning 用のMVTecADデータロードモジュール."""
# default
import logging
import typing as t

# third party packages
import pytorch_lightning as pl
import torch.utils.data as td
import torchvision.transforms as tv_transforms

# my packages
import src.data.dataset_torch as ds
import src.data.mvtecad as mvtecad
import src.data.mvtecad_torch as mvtecad_torch
import src.data.utils as ut

# logger
_logger = logging.getLogger(__name__)


class DataModuleAE(pl.LightningDataModule):
    def __init__(
        self,
        kind: mvtecad.Kind = mvtecad.Kind.HAZELNUT,
        image_size: t.Tuple[int, int] = (128, 128),
        batch_size: int = 144,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms_train = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.Resize(image_size),
                tv_transforms.ToTensor(),
            ]
        )
        self.transforms_test = tv_transforms.Compose(
            [tv_transforms.Resize(image_size), tv_transforms.ToTensor()]
        )

    def prepare_data(self) -> None:
        mvtec = mvtecad.MVTecAd(self.kind)
        mvtec.save()

    def setup(self, stage: t.Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.dataset_train_ = mvtecad_torch.DatasetAE(
                kind=self.kind, transforms=self.transforms_train, mode=ds.Mode.TRAIN,
            )
            self.dataset_valid_ = mvtecad_torch.DatasetAE(
                kind=self.kind, transforms=self.transforms_test, mode=ds.Mode.VALID,
            )

        if stage == "test" or stage is None:
            self.dataset_test_ = mvtecad_torch.DatasetAE(
                kind=self.kind, transforms=self.transforms_test, mode=ds.Mode.TEST,
            )

    def train_dataloader(self) -> td.DataLoader:
        return td.DataLoader(
            self.dataset_train_,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )

    def val_dataloader(self) -> td.DataLoader:
        return td.DataLoader(
            self.dataset_valid_,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )

    def test_dataloader(self) -> td.DataLoader:
        return td.DataLoader(
            self.dataset_test_,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=ut.worker_init_random,
        )
