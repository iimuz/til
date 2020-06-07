"""学習処理を実行するスクリプト."""
# default packages
import logging
import pathlib
import random
import traceback

# third party packages
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import torch
import torch.utils.data as torch_data

# my packages
import src.data.dataset as dataset
import src.data.directory as directory
import src.data.jena_climate as jena_climate
import src.models.simple_lstm as simple_lstm
import src.models.trainer as trainer

# logger
logger = logging.getLogger(__name__)


def create_dataset(filepath: pathlib.Path) -> np.ndarray:
    df = pd.read_csv(filepath)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    df = df.set_index(["Date Time"])
    df = df.sort_index()

    return df.to_numpy()


def worker_init_random(worker_id: int) -> None:
    random.seed(worker_id)


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # gpu の場合に必要
    # cudnn.deterministic = True  # gpu の場合は True にする必要があるが遅くなる。


def main() -> None:
    """学習処理の実行スクリプト."""
    logging.basicConfig(level=logging.INFO)
    random_seed = 42

    init_random_seed(random_seed)

    data_all = create_dataset(dataset.get_raw_filepath())
    # 全体の2割だけは、本当のテスト用に利用しない.
    data_partial, _ = model_selection.train_test_split(
        data_all, train_size=0.8, shuffle=False,
    )
    data_train, data_valid = model_selection.train_test_split(
        data_all, train_size=0.8, shuffle=False,
    )
    scaler = preprocessing.StandardScaler().fit(data_train)
    data_train = scaler.transform(data_train)
    data_valid = scaler.transform(data_valid)

    input_length = 8
    forecast_length = 1
    dataset_train = jena_climate.Dataset(
        data_train,
        input_length=input_length,
        forecast_length=forecast_length,
        mode=jena_climate.Mode.TRAIN,
    )
    dataset_valid = jena_climate.Dataset(
        data_valid,
        input_length=input_length,
        forecast_length=forecast_length,
        mode=jena_climate.Mode.VALID,
    )

    batch_size = 32
    dataloader_train = torch_data.DataLoader(
        dataset_train,
        batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )
    dataloader_valid = torch_data.DataLoader(
        dataset_valid,
        batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )

    network = simple_lstm.SingleLSTM(input_length, forecast_length)
    model = trainer.ForecastTrainer(network)
    model.set_dataloader(dataloader_train, dataloader_valid)

    cache_dir = directory.get_interim().joinpath("single_lstm")
    pl_trainer = pl.Trainer(
        early_stop_callback=True,
        default_save_path=str(cache_dir),
        fast_dev_run=False,
        min_epochs=10,
        max_epochs=100,
        gpus=[0] if torch.cuda.is_available() else None,
    )
    init_random_seed(random_seed)
    pl_trainer.fit(model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
