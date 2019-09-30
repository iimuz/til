import pathlib
from logging import getLogger
from typing import Tuple

import imageio
import numpy as np
from PIL import Image

logger = getLogger(__name__)


def create_images(
    num: int = 10,
    size: Tuple[int, int] = (30, 30),
    save_dir: str = "_data/",
    prefix: str = "images",
    save_ext: str = ".png",
) -> None:
    """斜め方向に白い四角が移動する画像を指定枚数作成します

    Args:
        num (int, optional): 作成する画像枚数. Defaults to 10.
        size (Tuple[int, int], optional): 作成する画像のサイズ. Defaults to (30, 30).
        save_dir (str, optional): 保存先ディレクトリ. Defaults to "_data/".
        prefix (str, optional): 保存するファイルの接頭辞. Defaults to "images".
        save_ext (str, optional): 保存するファイルの拡張子. Defaults to ".png".
    """
    if num > min(size):
        logger.error(f"image size is smaller than num. size = {size}, num = {num}")
        return

    block_size = (size[0] // num, size[0] // num)
    filepath_template = lambda: str(
        pathlib.Path(save_dir).joinpath(f"{prefix}{idx:03d}{save_ext}")
    )
    for idx in range(num):
        image = np.zeros(size, dtype=np.uint8)
        image[
            idx * block_size[0] : (idx + 1) * block_size[0],
            idx * block_size[1] : (idx + 1) * block_size[1],
        ] = 255
        image = Image.fromarray(image)
        image.save(filepath_template())


def create_gif(
    image_dir: str = "_data/",
    search_query: str = "images*.png",
    filepath: str = "_data/anim.gif",
) -> None:
    """gif ファイルを生成します

    Args:
        image_dir (str, optional): gif ファイルとしてまとめる画像があるフォルダ. Defaults to "_data/".
        search_query (str, optional): gif ファイルにまとめる画像を取得するためのクエリ. Defaults to "images*.png".
        filepath (str, optional): gif ファイルを保存するファイル名. Defaults to "_data/anim.gif".
    """
    filenames = pathlib.Path(image_dir).glob(search_query)
    filenames = sorted(filenames)
    with imageio.get_writer(filepath, mode="I") as writer:
        for filename in filenames:
            logger.info(f"load file: {filename}")

            image = imageio.imread(filename)
            writer.append_data(image)


def _main() -> None:
    """サンプルコードのエントリポイント
    """
    import logging

    logging.basicConfig(level=logging.INFO)

    # アニメーションとするためのフレーム画像の生成
    sequence_image_dir = "_data/"
    sequence_image_prefix = "images"
    sequence_image_ext = ".png"
    create_images(
        num=10,
        size=(30, 30),
        save_dir=sequence_image_dir,
        prefix=sequence_image_prefix,
        save_ext=sequence_image_ext,
    )

    # gif の生成
    search_query = f"{sequence_image_prefix}*{sequence_image_ext}"
    create_gif(
        image_dir=sequence_image_dir,
        search_query=search_query,
        filepath="_data/anim.gif",
    )


if __name__ == "__main__":
    _main()
