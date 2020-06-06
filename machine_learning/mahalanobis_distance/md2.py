"""MD^2値を計算するスクリプト.

Notes:
    下記資料の MD 値を計算します。
    - `https://www.jstage.jst.go.jp/article/qes/9/4/9_74/_pdf/-char/ja`
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

    # 共分散行列の計算(今回は標本分散)
    cov_mat = np.cov(data, rowvar=False, bias=True)

    # MD^2
    cov_inv_mat = np.linalg.inv(cov_mat)
    dist_sq_mat = np.apply_along_axis(lambda x: x @ cov_inv_mat @ x.T, axis=1, arr=data)
    dist_mat = dist_sq_mat / data.shape[1]

    logger.info(dist_mat)


if __name__ == "__main__":
    try:
        _main()
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
