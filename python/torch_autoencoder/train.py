import random
from logging import getLogger

import numpy as np
import torch
from pytorch_lightning import Trainer

from aetrainer import AETrainer

# from simpleautoencoder import SimpleAutoencoder
# from simpledeepautoencoder import SimpleDeepAutoencoder
# from simplecnnautoencoder import SimpleCNNAutoencoder
from simplelstm import SimpleLSTM

logger = getLogger(__name__)


def _init_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    save_path = "_models/lstm"
    gpus = [0] if torch.cuda.is_available() else None
    _init_rand_seed(0)

    # model = SimpleAutoencoder(10, 5).to(device)
    # model = SimpleDeepAutoencoder(10, 6).to(device)
    # hparams = dict(input_channels=1)
    # network = SimpleCNNAutoencoder(**hparams)
    hparams = dict(input_size=64, hidden_size=32, output_size=64,)
    network = SimpleLSTM(**hparams)

    model = AETrainer(
        model=network,
        sequence_length=64,
        batch_size=64,
        scaler_path="_data/interim/scaler.pkl",
        train_path="_data/interim/train.pkl",
        validation_path="_data/interim/train.pkl",
        learning_rate=1e-3,
        num_workers=2,
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
