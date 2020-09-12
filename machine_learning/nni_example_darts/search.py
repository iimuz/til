"""ネットワーク探索を実行するスクリプト."""
# default packages
import dataclasses as dc
import logging

# my packages
import datasets as ds

# logger
_logger = logging.getLogger(__name__)


@dc.dataclass
class Config:
    layers: int = 8
    batch_size: int = 64
    epochs: int = 50
    channels: int = 16

    log_frequency: int = 10
    unrolled: bool = False
    visualization: bool = False


def main(config: Config) -> None:
    dataset_train, dataset_valid = ds.get_dataset()


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO)

        _config = Config()
        main(_config)
    except Exception as e:
        _logger.exception(e)
