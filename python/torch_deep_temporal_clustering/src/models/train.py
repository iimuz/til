"""学習を実行するスクリプト."""
# default packages
import logging
import random

# third party packages
import numpy as np
import torch
from pytorch_lightning import Trainer

# my packages
from src.models.dtc_trainer import DTCTrainer
from src.models.deep_temporal_clustering import DTClustering

# logger
logger = logging.getLogger(__name__)


def _init_rand_seed(seed):
    """ランダム値を初期化します."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _main() -> None:
    logging.basicConfig(level=logging.INFO)

    gpus = [0] if torch.cuda.is_available() else None
    _init_rand_seed(seed=0)
    save_path = "_models/test"

    network = DTClustering()
    model = DTCTrainer(
        network=network,
        train_path="_data/raw/CBF/CBF_TRAIN.ts",
        valid_path="_data/raw/CBF/CBF_TEST.ts",
        batch_size=16,
        workers=0,
    )
    trainer = Trainer(
        early_stop_callback=True,
        default_save_path=save_path,
        fast_dev_run=False,
        min_epochs=1,
        max_epochs=10000,
        gpus=gpus,
    )
    trainer.fit(model)


if __name__ == "__main__":
    _main()
