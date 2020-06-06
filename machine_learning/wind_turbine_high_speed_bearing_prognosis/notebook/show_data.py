# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # データの確認


# ## 事前設定


# default packages
import logging
import pathlib
import sys
from datetime import datetime
from typing import Generator, Tuple


# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import sklearn.preprocessing as skpreprocessing
import sklearn.decomposition as skdecomposition
from IPython.display import display
from mpl_toolkits.mplot3d import axes3d
from scipy import io

# my packages
sys.path.append(str(pathlib.Path("..").resolve()))

import feature
import rank


# autoreload
# %load_ext autoreload
# %autoreload 2


# logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# settings
DATA_DIR = pathlib.Path("../data")
EXAMPLE_SENSOR = DATA_DIR.joinpath("hs_bearing_1/sensor-20130307T015746Z.mat")
EXAMPLE_TACH = DATA_DIR.joinpath("hs_bearing_1/tach-20130307T015746Z.mat")
SENSOR_FILES = DATA_DIR.glob("hs_bearing_*/sensor-*.mat")

SAMPLING_HZ = 97656
SAMPLING_SEC = 6


# ## データ読み込み


def load_data(filepath: str) -> pd.DataFrame:
    var = io.loadmat(filepath)
    date_val = datetime.strptime(
        pathlib.Path(filepath).stem.split("-")[-1], "%Y%m%dT%H%M%SZ"
    )

    df = pd.DataFrame(var["v"], columns=["vibration"])
    df["date"] = date_val

    return df


def load_some_files(files: Generator) -> pd.DataFrame:
    df = pd.DataFrame()
    for file in sorted(list(files)):
        logger.info(f"load file: {file}")
        df = pd.concat([df, load_data(file)], ignore_index=True)

    return df


df_org = load_some_files(SENSOR_FILES)


# ## データ表示


display(df_org.head())


display(df_org.info())


display(df_org.describe())


def plot_per_day(df: pd.DataFrame) -> None:
    fig = plt.figure()

    for date, data in df.groupby(by=["date"]):
        plt.plot(data["vibration"])

    plt.show()
    plt.close(fig)


plot_per_day(df_org)


def moving_window_mean(x: np.ndarray, window_size: int, step: int) -> None:
    calc = np.array(
        [
            np.fft.fft(x[idx : idx + window_size])
            for idx in range(0, x.shape[0] - window_size, step)
        ]
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(range(calc.shape[1]), range(calc.shape[0]))
    ax.plot_surface(X, Y, calc)

    plt.show()
    plt.clf()


moving_window_mean(df_org["vibration"].to_numpy(), 100000, SAMPLING_HZ * SAMPLING_SEC)


def pkurtosis(x: np.ndarray, window_size: int, step: int) -> None:
    calc = [
        scipy.signal.spectrogram(
            x[idx : idx + window_size],
            SAMPLING_HZ,
            window=("hann"),
            nperseg=256,
            scaling="spectrum",
        )
        for idx in range(0, x.shape[0] - window_size, step)
    ]
    t = calc[0][1]
    freq = calc[0][0]
    spectrogram = np.array([data for _, _, data in calc])

    # 単一の区間におけるスペクトルを表示
    plt.pcolormesh(t, freq, spectrogram[0, :, :])
    plt.show()
    plt.clf()

    # スペクトル尖度の計算と表示
    kurtosis = scipy.stats.kurtosis(spectrogram, axis=2)
    plt.pcolormesh(freq, range(kurtosis.shape[0]), kurtosis)
    plt.show()
    plt.clf()

    # スペクトル尖度の3D surface表示
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(freq, range(kurtosis.shape[0]))
    ax.plot_surface(X, Y, kurtosis)
    plt.show()
    plt.clf()


pkurtosis(
    df_org["vibration"].to_numpy(), SAMPLING_HZ * SAMPLING_SEC, SAMPLING_HZ,
)


def calc_feature(x: np.ndarray) -> None:
    day_sample = SAMPLING_HZ * SAMPLING_SEC
    days = x.shape[0] // day_sample
    df = pd.DataFrame()
    for day in range(days):
        features = feature.calc_all(
            x[day * day_sample : (day + 1) * day_sample], SAMPLING_HZ
        )
        df = df.append({"day": day, **features}, ignore_index=True)
    df = df.set_index(["day"])

    return df


df_feature = calc_feature(df_org["vibration"].to_numpy())


display(df_feature.info())


display(df_feature.describe())


display(df_feature.head())


def smoothing(df: pd.DataFrame) -> pd.DataFrame:
    span = 5
    x = df.to_numpy()
    x_smooth = np.array(
        [
            np.mean(x[max([0, day - span]) : day + 1, :], axis=0)
            for day in range(len(df))
        ]
    )

    df_smooth = pd.DataFrame(x_smooth, columns=df.columns, index=df.index)

    return df_smooth


df_smooth = smoothing(df_feature)


def show(df1: pd.DataFrame, df2: pd.DataFrame, column_name: str) -> None:
    plt.figure()
    plt.plot(df1[column_name])
    plt.plot(df2[column_name])
    plt.show()
    plt.clf()


show(df_feature, df_smooth, "SKMean")


def split_data(
    df: pd.DataFrame, num_of_train: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df.iloc[:num_of_train, :]
    df_valid = df.iloc[num_of_train:, :]

    return df_train, df_valid


df_train, df_valid = split_data(df_smooth, 20)


def feature_importance(df: pd.DataFrame) -> None:
    scores = {name: rank.monotonicity(df[name].to_numpy()) for name in df.columns}
    df = pd.DataFrame(scores, index=[0])

    return df


df_importance = feature_importance(df_train)


df_importance.T.sort_values(by=[0], ascending=False).plot(kind="bar")
selected_name = df_importance.T[df_importance.T > 0.3].dropna().index


def normalize(
    train: pd.DataFrame, valid: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ss = skpreprocessing.StandardScaler()
    ss.fit(train)

    train_normalized = ss.transform(train)
    valid_normalized = ss.transform(valid)

    df_nrm_train = pd.DataFrame(
        train_normalized, index=train.index, columns=train.columns
    )
    df_nrm_valid = pd.DataFrame(
        valid_normalized, index=valid.index, columns=valid.columns
    )

    return df_nrm_train, df_nrm_valid


df_normalized_train, df_normalized_valid = normalize(df_train, df_valid)


display(df_normalized_train.describe())


display(df_normalized_valid.describe())


def show_decomposition(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    pca = skdecomposition.PCA(n_components=2)
    pca.fit(train)

    pca_train = pca.transform(train)
    pca_valid = pca.transform(valid)

    _ = plt.figure()
    plt.plot(
        pca_train[:, 0], pca_train[:, 1], linestyle="None", marker="o", color="blue"
    )
    plt.plot(
        pca_valid[:, 0], pca_valid[:, 1], linestyle="None", marker="o", color="green"
    )
    plt.show()
    plt.clf()


df_selected_train = df_normalized_train[selected_name]
df_selected_valid = df_normalized_valid[selected_name]
show_decomposition(df_selected_train, df_selected_valid)


def show_health(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    pca = skdecomposition.PCA(n_components=2)
    pca.fit(train)

    pca_train = pca.transform(train)
    pca_valid = pca.transform(valid)

    health_index = 0
    health = np.hstack([pca_train[:, health_index], pca_valid[:, health_index]])

    logger.info(health.shape)

    _ = plt.figure()
    plt.plot(health)
    plt.show()
    plt.clf()


show_health(df_selected_train, df_selected_valid)
