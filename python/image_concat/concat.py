import math
import numpy as np
import pathlib
import sys
import yaml

from PIL import Image
from logging import getLogger
from typing import List

logger = getLogger(__name__)


def concat_image(
    filelist: List[pathlib.Path], cols: int, width: int, height: int
) -> np.ndarray:
    """ファイルリストのファイルを読み込み一枚の画像へ変換する

    Args:
        filelist (List[pathlib.Path]): 読み込むファイルパスのリスト
        cols (int): 結合時の横方向の枚数
        width (int): 結合時に変換する画像サイズ
        height (int): 結合時に変換する画像サイズ

    Returns:
        np.ndarray: 結合した一枚の画像
    """
    rows = math.ceil(len(filelist) / cols)
    dst_img = np.zeros((width * rows, height * cols))
    for idx, path in enumerate(filelist):
        img = Image.open(path).resize((width, height), Image.LINEAR)
        row, col = divmod(idx, cols)
        start_height, end_height = row * height, (row + 1) * height
        start_width, end_width = col * width, (col + 1) * width
        dst_img[start_height:end_height, start_width:end_width] = img

    return dst_img


def main() -> None:
    """指定されたフォルダ内の画像を読み込みリサイズ、結合を行い保存する。
    """
    CONFIG_PATH = "config.yaml" if len(sys.argv) < 2 else sys.argv[1]
    logger.info(f"config: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f.read())
    logger.info(f"{CONFIG}")

    filelist = sorted(
        pathlib.Path(CONFIG["image"]["dir"]).glob(CONFIG["image"]["query"])
    )
    img_concat = concat_image(
        filelist,
        CONFIG["matrix"]["columns"],
        CONFIG["image"]["size"]["width"],
        CONFIG["image"]["size"]["height"],
    )
    img_normalized = normalize_image(img_concat)
    img_normalized *= 255

    Image.fromarray(img_normalized.astype(np.uint8)).save(CONFIG["output"])


def normalize_image(img: np.ndarray) -> np.ndarray:
    """画像を min, max を利用して [0, 1] の範囲に正規化する

    Args:
        img (np.ndarray): 正規化対象の画像

    Returns:
        np.ndarray: 正規化した画像
    """
    max_val, min_val = img.max(), img.min()
    img_normalized = (img - min_val) / (max_val - min_val)
    return img_normalized


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
