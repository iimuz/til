"""マハラノビス距離を計算するスクリプト(Scipy 利用版).

Notes:
    Scipy の下記関数を利用しています。
    - `https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html`
"""
# default packages
import logging
import traceback

# third party packages
import numpy as np
import scipy.spatial.distance as distance

# my packages
import log_handler

# logger
logger = logging.getLogger(__name__)


def _main() -> None:
    """マハラノビス距離計算のスクリプト."""
    # logger
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(log_handler.stdout_handler())

    # 初期化
    np.random.seed(0)

    # データ設定
    rows, cols = 10, 3
    data = np.random.randn(rows, cols).astype(np.float64)

    # 平均及び共分散行列の計算
    avg_vec = np.mean(data, axis=0).reshape((1, -1))
    cov_mat = np.cov(data - avg_vec, rowvar=False, bias=True)
    cov_inv_mat = np.linalg.inv(cov_mat)

    # Mahalanobis distance
    dist_mat = np.apply_along_axis(
        lambda x: distance.mahalanobis(x, avg_vec, cov_inv_mat), axis=1, arr=data,
    )

    logger.info(dist_mat)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
