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
from typing import Generator


# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from IPython.display import display
from mpl_toolkits.mplot3d import axes3d
from scipy import io

# my packages
sys.path.append(str(pathlib.Path("..").resolve()))

import feature


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
            x[idx : idx + window_size], SAMPLING_HZ, window=("hann"), nperseg=256, scaling="spectrum",
        )
        for idx in range(0, x.shape[0] - window_size, step)
    ]
    
    t = calc[0][1]
    freq = calc[0][0]

    plt.pcolormesh(t, freq, calc[0][2])
    plt.show()
    plt.clf()
    
    calc = np.array([data for _, _, data in calc])
    calc = scipy.stats.kurtosis(calc, axis=2)
    # calc = np.sum(calc, axis=2)

    plt.pcolormesh(freq, range(calc.shape[0]), calc)
    plt.show()
    plt.clf()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(freq, range(calc.shape[0]))
    ax.plot_surface(X, Y, calc)

    plt.show()
    plt.clf()


pkurtosis(
    df_org["vibration"].to_numpy(), SAMPLING_HZ * SAMPLING_SEC, SAMPLING_HZ,
)


def calc_feature(x: np.ndarray) -> None:
    features = feature.calc_all(x[: SAMPLING_HZ * SAMPLING_SEC])

    logger.info(features)


calc_feature(df_org["vibration"].to_numpy())
