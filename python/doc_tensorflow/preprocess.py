import numpy as np

from logging import getLogger
from PIL import Image


logger = getLogger(__name__)


def resize(x):
    """ 画像サイズが最低値以下の場合に最低値に拡大する
    """
    width, height = 96, 96

    x_out = []
    for i in range(len(x)):
        img = x[i].reshape(x[i].shape[:-1])
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        x_out.append(np.array(img))

    return np.array(x_out)
