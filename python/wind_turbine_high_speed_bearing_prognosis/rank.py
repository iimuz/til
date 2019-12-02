# default pckages
from logging import getLogger

# third party
import numpy as np


# logger
logger = getLogger(__name__)


def monotonicity(data: np.ndarray) -> float:
    num_of_data = data.shape[0]
    diff = data[1:] - data[:-1]
    num_of_positive = (diff >= 0.0).sum()
    num_of_negative = diff.shape[0] - num_of_positive

    score = abs(num_of_positive - num_of_negative) / (num_of_data - 1)

    return score
