"""学習を実行するスクリプト."""
# default packages
import logging
import pathlib
import random

# third party packages
import numpy as np
import torch
from pytorch_lightning import Trainer

# my packages
import src.models.deep_temporal_clustering as dtc
from src.data.tsdataset import TSDataset
from src.models.dtc_trainer import DTCTrainer

# logger
logger = logging.getLogger(__name__)


def _init_rand_seed(seed):
    """ランダム値を初期化します."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_centers(
    train_path: pathlib.Path, network: dtc.DTClustering, n_clusters: int, workers: int
) -> np.ndarray:
    dataset = TSDataset(train_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=workers
    )
    train_x = next(iter(loader))[0].permute(0, 3, 1, 2)
    init_centers = dtc.calc_centeroid(train_x, network, n_clusters=n_clusters)

    return init_centers


def _main() -> None:
    logging.basicConfig(level=logging.INFO)

    gpus = [0] if torch.cuda.is_available() else None
    _init_rand_seed(seed=0)
    save_path = "_models/test"
    ckpt_path = pathlib.Path(
        "_models/test/lightning_logs/version_0/checkpoints/_ckpt_epoch_9999.ckpt"
    )
    train_path = pathlib.Path("_data/raw/CBF/CBF_TRAIN.ts")
    valid_path = pathlib.Path("_data/raw/CBF/CBF_TEST.ts")
    batch_size = 16
    n_clusters = 10
    workers = 0

    network = dtc.DTClustering()
    model = DTCTrainer(
        network=network,
        train_path=train_path,
        valid_path=valid_path,
        batch_size=batch_size,
        workers=workers,
    )
    if ckpt_path.exists() is False:
        logger.info("learning...")
        trainer = Trainer(
            early_stop_callback=True,
            default_save_path=save_path,
            fast_dev_run=False,
            min_epochs=1,
            max_epochs=10000,
            gpus=gpus,
        )
        trainer.fit(model)
    else:
        logger.info(f"load checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path))
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.freeze()

    centers = init_centers(train_path, network, n_clusters, workers)
    logger.info(centers.shape)


if __name__ == "__main__":
    _main()
