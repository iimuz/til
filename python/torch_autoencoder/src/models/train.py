"""学習を実行する."""
# default packages
import argparse
import random
from logging import getLogger
from typing import Dict, Optional, Tuple

# third party packages
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer

# my packages
from src.models.aetrainer import AETrainer
from src.models.doublecnn import DoubleCNN
from src.models.doubledense import DoubleDense
from src.models.singledense import SingleDense
from src.models.singlelstm import SingleLSTM

# logger
logger = getLogger(__name__)


def _argparse() -> Dict:
    parser = argparse.ArgumentParser(description="Training autoencoders.")
    parser.add_argument("name", help="autoencoder name", default="SingleDense")
    args = parser.parse_args()

    return vars(args)


def _create_network(
    name: str, sequence_length: int
) -> Tuple[Optional[nn.Module], Optional[Dict]]:
    """ネットワークを生成します."""
    if name == "SingleDense":
        hparams = dict(input_dim=sequence_length, hidden_dim=sequence_length // 2)
        return SingleDense(**hparams), hparams

    if name == "DoubleCNN":
        hparams = dict(input_channel=1)
        return DoubleCNN(**hparams), hparams

    if name == "DoubleDense":
        hparams = dict(input_dim=sequence_length, hidden_dim=sequence_length // 2)
        return DoubleDense(**hparams), hparams

    if name == "SingleLSTM":
        hparams = dict(
            input_size=1,
            hidden_size=32,
            output_size=1,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        return SingleLSTM(**hparams), hparams

    logger.error(f"unknown network name: {name}")
    return None, None


def _init_rand_seed(seed):
    """ランダム値を初期化します."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _main() -> None:
    """学習を実行するメインスクリプトです."""
    import logging

    logging.basicConfig(level=logging.INFO)

    args = _argparse()

    gpus = [0] if torch.cuda.is_available() else None
    _init_rand_seed(0)

    sequence_length = 64
    network, hparams = _create_network(args["name"], sequence_length)
    if network is None or hparams is None:
        logger.error("network or hparams is None.")
        return

    save_path = f"_models/{args['name']}"
    model = AETrainer(
        model=network,
        sequence_length=sequence_length,
        batch_size=64,
        scaler_path="_data/interim/dataset/scaler.pkl",
        train_path="_data/interim/dataset/train.pkl",
        validation_path="_data/interim/dataset/test.pkl",
        learning_rate=1e-3,
        num_workers=4,
        hparams=hparams,
    )
    trainer = Trainer(
        early_stop_callback=True,
        default_save_path=save_path,
        fast_dev_run=False,
        min_epochs=1,
        max_epochs=1000,
        gpus=gpus,
    )
    trainer.fit(model)


if __name__ == "__main__":
    _main()
