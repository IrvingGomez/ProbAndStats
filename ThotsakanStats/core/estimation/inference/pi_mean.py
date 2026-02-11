# stats/inference/pi_mean.py

import numpy as np
from scipy.stats import norm, t

from .estimators import estimate_mean, estimate_sigma


def pi_mean(
    *,
    data,
    alpha,
    estimator,
    dist,
    sigma_estimator,
    trim_param=None,
    winsor_limits=None,
    weights=None,
):
    """
    Prediction interval for a new observation based on the mean estimator.

    Var(X_new − μ̂) = σ² (1 + 1/n)

    - μ̂: user-chosen mean estimator.
    - σ: user-chosen deviation estimator.
    """
    data = np.asarray(data)
    n = len(data)

    mu_hat = estimate_mean(
        data,
        estimator,
        trim_param=trim_param,
        winsor_limits=winsor_limits,
        weights=weights,
    )

    sigma_hat = estimate_sigma(
        data=data,
        estimator=sigma_estimator,
    )

    scale = sigma_hat * np.sqrt(1 + 1 / n)

    if dist == "t":
        return (
            t.ppf(alpha / 2, n - 1, loc=mu_hat, scale=scale),
            t.ppf(1 - alpha / 2, n - 1, loc=mu_hat, scale=scale),
        )

    return (
        norm.ppf(alpha / 2, loc=mu_hat, scale=scale),
        norm.ppf(1 - alpha / 2, loc=mu_hat, scale=scale),
    )
