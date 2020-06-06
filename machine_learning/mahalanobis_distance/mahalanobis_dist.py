"""マハラノビス距離を計算するスクリプト.

Notes:
    下記の距離を計算するスクリプトします。
    - `https://en.wikipedia.org/wiki/Mahalanobis_distance`
"""
# default packages
import logging
import traceback

# third party packages
import numpy as np

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

    # 中心化
    avg_vec = np.mean(data, axis=0).reshape((1, -1))
    data -= avg_vec

    # 共分散行列の計算
    cov_mat = np.cov(data, rowvar=False, bias=True)

    # Mahalanobis distance
    cov_inv_mat = np.linalg.inv(cov_mat)
    dist_sq_mat = np.apply_along_axis(lambda x: x @ cov_inv_mat @ x.T, axis=1, arr=data)
    dist_mat = np.sqrt(dist_sq_mat)

    logger.info(dist_mat)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
