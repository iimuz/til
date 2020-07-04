"""学習処理を実行するスクリプト."""
# default packages
import dataclasses
import logging
import pathlib
import random
import sys
import traceback
import typing as t

# third party packages
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import yaml

# my packages
import src.data.dataset as dataset
import src.data.directory as directory
import src.data.jena_climate as jena_climate
import src.models.dilated_cnn as dilated_cnn
import src.models.simple_lstm as simple_lstm
import src.models.trainer as trainer

# logger
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    # dataset parameters
    input_length: int = 128
    forecast_length: int = 64

    # training parameters
    random_seed: int = 0
    batch_size: int = 64
    early_stop: bool = True
    min_epochs: int = 10
    max_epochs: int = 1000

    num_workers: int = 4
    save_top_k: int = 5
    row_log_interval: int = 1000
    progress_bar_refresh_rate: int = 100

    # model paramters
    network_name: str = "SimpleLSTM"
    network_params: t.Dict = dataclasses.field(default_factory=dict)

    # output
    log_dir: str = "simple_lstm"


def create_dataset(filepath: pathlib.Path) -> np.ndarray:
    df = pd.read_csv(filepath)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    df = df.set_index(["Date Time"])
    df = df.sort_index()

    return df.to_numpy()


def get_network(name: str, params: t.Dict) -> t.Optional[nn.Module]:
    if "DilatedCNN" == name:
        return dilated_cnn.DCNN(**params)
    if "DoubleLSTM" == name:
        return simple_lstm.DoubleLSTM(**params)
    if "LayeredDoubleLSTM" == name:
        return simple_lstm.LayeredDoubleLSTM(**params)
    if "SimpleLSTM" == name:
        return simple_lstm.SingleLSTM(**params)

    logger.error(f"unknown network name: {name}")
    return None


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # gpu の場合に必要
    # cudnn.deterministic = True  # gpu の場合は True にする必要があるが遅くなる。


def load_config() -> t.Optional[Config]:
    """入力引数で与えられた設定ファイルを読む."""
    if len(sys.argv) != 2:
        logger.error("input arguments error: python <script_path> <config_path>")
        return None

    config_path = pathlib.Path(sys.argv[1])
    if not config_path.exists():
        logger.error(f"input config path error: {config_path}")
        return None

    with open(str(config_path), "r") as f:
        config = Config(**yaml.load(f, Loader=yaml.SafeLoader))

    return config


def worker_init_random(worker_id: int) -> None:
    random.seed(worker_id)


def main() -> None:
    """学習処理の実行スクリプト."""
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    if config is None:
        logger.error("cannot load config file.")
        return

    init_random_seed(config.random_seed)

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

    dataset_train = jena_climate.Dataset(
        data_train,
        input_length=config.input_length,
        forecast_length=config.forecast_length,
        mode=jena_climate.Mode.TRAIN,
    )
    dataset_valid = jena_climate.Dataset(
        data_valid,
        input_length=config.input_length,
        forecast_length=config.forecast_length,
        mode=jena_climate.Mode.VALID,
    )

    dataloader_train = torch_data.DataLoader(
        dataset_train,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )
    dataloader_valid = torch_data.DataLoader(
        dataset_valid,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_random,
    )

    pl.seed_everything(config.random_seed)
    input_feature = data_train.shape[1]
    output_dims = config.forecast_length * input_feature
    network = get_network(
        config.network_name,
        {
            "input_feature": input_feature,
            "output_size": output_dims,
            **config.network_params,
        },
    )
    if network is None:
        logger.error("network error.")
        return
    model = trainer.ForecastTrainer(network)
    model.set_dataloader(dataloader_train, dataloader_valid)

    cache_dir = directory.get_interim().joinpath(config.log_dir)
    profiler = True  # if use detail profiler, pl_profiler.AdvancedProfiler()
    model_checkpoint = pl_callbacks.ModelCheckpoint(
        filepath=str(cache_dir),
        monitor="val_loss",
        save_top_k=config.save_top_k,
        save_weights_only=False,
        mode="min",
        period=1,
    )
    pl_trainer = pl.Trainer(
        early_stop_callback=config.early_stop,
        default_save_path=str(cache_dir),
        fast_dev_run=False,
        min_epochs=config.min_epochs,
        max_epochs=config.max_epochs,
        gpus=[0] if torch.cuda.is_available() else None,
        row_log_interval=config.row_log_interval,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        profiler=profiler,
        checkpoint_callback=model_checkpoint,
    )
    pl_trainer.fit(model)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
