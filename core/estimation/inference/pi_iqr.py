# stats/inference/pi_iqr.py

import numpy as np
from scipy.stats import norm


def pi_iqr(
    *,
    data,
    alpha,
):
    """
    Prediction interval based on IQR.
    """
    q1, q3 = np.quantile(data, [0.25, 0.75])
    iqr = q3 - q1

    delta = 0.5 * (norm.ppf(1 - alpha / 2) / norm.ppf(0.75) - 1)

    return (
        q1 - delta * iqr,
        q3 + delta * iqr,
    )
