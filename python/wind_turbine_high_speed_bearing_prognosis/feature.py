# default package
from logging import getLogger
from typing import Dict


# third party
import numpy as np
import scipy.signal
import scipy.stats


# logger
logger = getLogger(__name__)


def calc_all(data: np.ndarray, fs: float) -> Dict:
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
    }
    features.update(calc_spectrum(data, fs))

    return features


def calc_spectrum(data: np.ndarray, fs: float) -> Dict:
    freq, t, spectrum = scipy.signal.spectrogram(
        data, fs, window=("hann"), nperseg=256, scaling="spectrum"
    )
    kspectrum = scipy.stats.kurtosis(spectrum, axis=1)

    features = {
        "SKMean": np.mean(kspectrum),
        "SKStd": np.std(kspectrum),
        "SKSkewness": scipy.stats.skew(kspectrum),
        "SKKurtosis": scipy.stats.kurtosis(kspectrum),
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
