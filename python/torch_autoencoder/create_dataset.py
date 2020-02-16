import pathlib
import pickle
from datetime import datetime
from datetime import timedelta
from logging import getLogger
from typing import List
from urllib import request

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from tqdmupto import TqdmUpTo

logger = getLogger(__name__)


def _calc_vwap(df: pd.DataFrame) -> pd.DataFrame:
    def vwap(row):
        nom = np.sum(row.price * row.foreignNotional)
        denomi = np.sum(row.foreignNotional)
        return nom / denomi

    df_vwap = df.groupby(pd.Grouper(freq="1Min")).apply(vwap)
    df_vwap = df_vwap.to_frame(name="vwap")
    return df_vwap


def _download(
    base_url: str, target_list: List[str], download_dir: pathlib.Path
) -> None:
    download_dir.mkdir(exist_ok=True)
    for filename in target_list:
        url = base_url + filename
        filepath = download_dir.joinpath(filename)
        if filepath.exists():
            continue

        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
            request.urlretrieve(url, filename=str(filepath), reporthook=t.update_to)


def _read_dataset(filelist: List[pathlib.Path]) -> pd.DataFrame:
    data = {}
    with tqdm(filelist) as pbar:
        for filepath in pbar:
            pbar.set_description(f"file: {filepath.stem}")
            try:
                data[filepath] = pd.read_csv(filepath)
            except Exception as e:
                logger.error(f"filename: {filepath}\n{e}")
                continue
    df = pd.concat([val for val in data.values()])
    df = df[df.symbol == "XBTUSD"]
    df.timestamp = pd.to_datetime(df.timestamp.str.replace("D", "T"))
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    return df


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    BASE_URL = "https://s3-eu-west-1.amazonaws.com/public.bitmex.com/data/trade/"
    TARGET_LIST = [
        f"{(datetime(2019, 8, 1) + timedelta(days=delta)).strftime('%Y%m%d')}.csv.gz"
        for delta in range(60)
    ]
    DOWNLOAD_DIR = pathlib.Path("_data")
    DATASET_FILE = DOWNLOAD_DIR.joinpath("dataset.pkl")
    TRAIN_FILE = DOWNLOAD_DIR.joinpath("train.pkl")
    TEST_FILE = DOWNLOAD_DIR.joinpath("test.pkl")
    TRAIN_DATE = datetime(2019, 9, 1)
    TRAIN_SC = DOWNLOAD_DIR.joinpath("scaler.pkl")
    CHUNUK_NUM = 5

    # データファイルのダウンロード
    _download(BASE_URL, TARGET_LIST, DOWNLOAD_DIR)

    # 単変量の時系列データへ変換
    if DATASET_FILE.exists() is False:
        df_vwap_dict = {}
        filelist = list(DOWNLOAD_DIR.glob("*.csv.gz"))
        for idx in range(0, len(filelist), CHUNUK_NUM):
            end_idx = min(idx + CHUNUK_NUM, len(filelist))
            df = _read_dataset(filelist[idx:end_idx])
            df_vwap_dict[idx] = _calc_vwap(df)
        df_vwap = pd.concat([df for df in df_vwap_dict.values()])
        df_vwap.to_pickle(str(DATASET_FILE))
    else:
        df_vwap = pd.read_pickle(str(DATASET_FILE))

    # 学習データと検証データを分離
    if TRAIN_FILE.exists() is False:
        df_train = df_vwap.loc[:TRAIN_DATE]
        df_train.to_pickle(str(TRAIN_FILE))
    else:
        df_train = pd.read_pickle(str(TRAIN_FILE))
    if TEST_FILE.exists() is False:
        df_test = df_vwap.loc[TRAIN_DATE:]
        df_test.to_pickle(str(TEST_FILE))
    else:
        df_test = pd.read_pickle(str(TEST_FILE))

    if TRAIN_SC.exists() is False:
        sc = MinMaxScaler().fit(df_train.to_numpy())
        with open(str(TRAIN_SC), "wb") as f:
            pickle.dump(sc, f)
    else:
        with open(str(TRAIN_SC), "rb") as f:
            sc = pickle.load(f)

    logger.info(f"train shape: {df_train.shape}")
    logger.info(f"test shape: {df_test.shape}")
    logger.info(f"scaler (min, max): ({sc.data_min_}, {sc.data_max_})")


if __name__ == "__main__":
    _main()
