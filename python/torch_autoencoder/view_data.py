import pathlib
from logging import getLogger

import numpy as np
import pandas as pd

from vwapdataset import VwapDataset


logger = getLogger(__name__)


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    dataset = VwapDataset(filepath=pathlib.Path("_data/train.pkl"))
    for i in range(10):
        logger.info(f"value: {dataset[i]}")


if __name__ == "__main__":
    _main()
