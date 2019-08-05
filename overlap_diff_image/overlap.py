import numpy as np

from PIL import Image
from logging import getLogger

logger = getLogger(__name__)


def main() -> None:
    """指定したベース画像との差分をオーバーラップした画像を保存する。
    """
    BASE_IMAGE_NAME = "_data/hoge000.png"
    DIFF_IMAGE_LIST = [
        "_data/hoge001.png",
        "_data/hoge002.png",
        "_data/hoge003.png",
        "_data/hoge004.png",
        "_data/hoge005.png",
    ]
    OUTPUT_IMAGE_NAME = "_data/hoge_overlap.png"

    logger.debug(f"load base image: {BASE_IMAGE_NAME}")
    img_base = np.asarray(Image.open(BASE_IMAGE_NAME)).astype(np.int)

    img_overlap = img_base.copy()
    for diff_name in DIFF_IMAGE_LIST:
        logger.debug(f"load image: {diff_name}")
        img_diff = np.asarray(Image.open(diff_name)).astype(np.int)
        img_overlap += np.abs(img_base - img_diff)
    img_overlap[img_overlap < 0] = 0
    img_overlap[img_overlap > 255] = 255

    logger.debug(f"output image: {OUTPUT_IMAGE_NAME}")
    Image.fromarray(np.uint8(img_overlap)).save(OUTPUT_IMAGE_NAME)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    main()
