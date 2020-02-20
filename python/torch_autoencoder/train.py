import random
from logging import getLogger

import numpy as np
import torch
from pytorch_lightning import Trainer

from aetrainer import AETrainer

from simpleautoencoder import SimpleAutoencoder
from simpledeepautoencoder import SimpleDeepAutoencoder
from simplecnnautoencoder import SimpleCNNAutoencoder
from simplelstm import SimpleLSTM

logger = getLogger(__name__)


def _create_network(name: str, sequence_length: int):
    if name == "SimpleAE":
        hparams = dict(input_dim=sequence_length, hidden_dim=sequence_length // 2)
        return SimpleAutoencoder(**hparams), hparams

    if name == "SimpleCNN":
        hparams = dict(input_channel=1)
        return SimpleCNNAutoencoder(**hparams), hparams

    if name == "SimpleDNN":
        hparams = dict(input_dim=sequence_length, hidden_dim=sequence_length // 2)
        return SimpleDeepAutoencoder(**hparams), hparams

    if name == "SimpleLSTM":
        hparams = dict(
            input_size=sequence_length,
            hidden_size=32,
            output_size=sequence_length,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        return SimpleLSTM(**hparams), hparams


def _init_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    gpus = [0] if torch.cuda.is_available() else None
    _init_rand_seed(0)

    sequence_length = 64
    # name = "SimpleCNN"
    name = "SimpleLSTM"
    network, hparams = _create_network(name, sequence_length)
    save_path = f"_models/{name}"

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
