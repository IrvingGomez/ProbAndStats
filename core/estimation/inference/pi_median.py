# stats/inference/pi_median.py

import numpy as np
from scipy.stats import norm

from .estimators import estimate_median, estimate_sigma


def pi_median(
    *,
    data,
    alpha,
    estimator,
    sigma_estimator,
):
    """
    Asymptotic prediction interval based on the chosen median estimator.

    scale = sqrt(σ² + πσ² / (2n)) = σ * sqrt(1 + π/(2n))

    Median and σ are computed with user-chosen estimators.
    """
    data = np.asarray(data)
    n = len(data)

    median_hat = estimate_median(data, estimator)

    sigma_hat = estimate_sigma(
        data=data,
        estimator=sigma_estimator,
    )

    scale = np.sqrt(sigma_hat**2 + np.pi * sigma_hat**2 / (2 * n))

    return (
        norm.ppf(alpha / 2, median_hat, scale),
        norm.ppf(1 - alpha / 2, median_hat, scale),
    )
