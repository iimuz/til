# default package
from typing import Dict


# third party
import numpy as np
import scipy.signal
import scipy.stats


def calc_all(data: np.ndarray) -> Dict:
    _, _, spectrogram = scipy.signal.spectrogram(data, 97656, return_onesided=False)
    spectrogram = scipy.stats.kurtosis(spectrogram ** 2, axis=0)
    print(spectrogram.shape)

    features = {
        "Mean": np.mean(data),
        "Std": np.std(data),
        "Skewness": scipy.stats.skew(data),
        "Kurtosis": scipy.stats.kurtosis(data),
        "Peak2Peak": peak2peak(data),
        "CrestFactor": crest_factor(data),
        "ShapeFactor": shape_factor(data),
        "ImpulseFactor": impulse_factor(data),
        "MarginFactor": margin_factor(data),
        "Energy": energy(data),
        "SKMean": np.mean(spectrogram),
    }

    return features


def crest_factor(data: np.ndarray) -> float:
    max_val = np.max(data)
    rms = np.sqrt(np.mean(data ** 2))
    return max_val / rms


def energy(data: np.ndarray) -> float:
    return np.sum(data ** 2)


def impulse_factor(data: np.ndarray) -> float:
    max_val = np.max(data)
    mean_abs = np.mean(np.abs(data))
    return max_val / mean_abs


def margin_factor(data: np.ndarray) -> float:
    max_val = np.max(data)
    mean_abs = np.mean(np.abs(data))
    return max_val / (mean_abs ** 2)


def peak2peak(data: np.ndarray) -> float:
    min_val = np.min(data)
    max_val = np.max(data)
    return max_val - min_val


def shape_factor(data: np.ndarray) -> float:
    rms = np.sqrt(np.mean(data ** 2))
    mean_abs = np.mean(np.abs(data))
    return rms / mean_abs
