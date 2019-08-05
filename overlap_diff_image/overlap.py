import numpy as np
import sys
import yaml

from PIL import Image
from logging import getLogger

logger = getLogger(__name__)


def main() -> None:
    """指定したベース画像との差分をオーバーラップした画像を保存する。
    """
    CONFIG_PATH = "config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    logger.debug(f"config: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f.read())

    logger.debug(CONFIG)

    logger.debug(f"load base image: {CONFIG['base_image_path']}")
    img_base = np.asarray(Image.open(CONFIG["base_image_path"])).astype(np.int)

    img_overlap = img_base.copy()
    for diff_name in CONFIG["diff_image_list"]:
        logger.debug(f"load image: {diff_name}")
        img_diff = np.asarray(Image.open(diff_name)).astype(np.int)
        img_overlap += np.abs(img_base - img_diff)
    img_overlap[img_overlap < 0] = 0
    img_overlap[img_overlap > 255] = 255

    logger.debug(f"output image: {CONFIG['output_image_path']}")
    Image.fromarray(np.uint8(img_overlap)).save(CONFIG["output_image_path"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
