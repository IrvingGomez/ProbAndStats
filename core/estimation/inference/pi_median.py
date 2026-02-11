# stats/inference/pi_median.py

import numpy as np
from scipy.stats import norm

from .estimators import estimate_sigma


def pi_median(
    *,
    data,
    alpha,
    sigma_estimator,
):
    """
    Asymptotic prediction interval based on the sample median.

    scale = sqrt(σ² + πσ² / (2n)) = σ * sqrt(1 + π/(2n))

    σ is computed with the user-chosen deviation estimator.
    """
    data = np.asarray(data)
    n = len(data)

    median_hat = np.median(data)

    sigma_hat = estimate_sigma(
        data=data,
        estimator=sigma_estimator,
    )

    scale = np.sqrt(sigma_hat**2 + np.pi * sigma_hat**2 / (2 * n))

    return (
        norm.ppf(alpha / 2, median_hat, scale),
        norm.ppf(1 - alpha / 2, median_hat, scale),
    )
