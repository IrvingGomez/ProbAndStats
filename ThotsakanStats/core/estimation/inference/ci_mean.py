# stats/inference/ci_mean.py

import numpy as np
from scipy.stats import norm, t

from .estimators import estimate_mean, estimate_sigma


def ci_mean_analytic(
    *,
    data,
    estimator,
    alpha,
    dist,
    sigma_estimator,
    trim_param=None,
    winsor_limits=None,
    weights=None,
):
    """
    Analytic confidence interval for the mean.

    - Mean is computed with the user-chosen mean estimator.
    - σ is computed with the user-chosen deviation estimator.
    """
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

    scale = sigma_hat / np.sqrt(n)

    if dist == "t":
        return (
            t.ppf(alpha / 2, n - 1, loc=mu_hat, scale=scale),
            t.ppf(1 - alpha / 2, n - 1, loc=mu_hat, scale=scale),
        )

    return (
        norm.ppf(alpha / 2, loc=mu_hat, scale=scale),
        norm.ppf(1 - alpha / 2, loc=mu_hat, scale=scale),
    )


def ci_mean_bootstrap(
    *,
    data,
    estimator,
    alpha,
    B,
    trim_param=None,
    winsor_limits=None,
    weights=None,
):
    """
    Bootstrap CI for the mean using the user-chosen mean estimator.
    """
    data = np.asarray(data)
    n = len(data)

    boot_stats = np.empty(B)

    for b in range(B):
        idx = np.random.choice(n, size=n, replace=True)
        boot_data = data[idx]

        boot_weights = None
        if weights is not None:
            boot_weights = np.asarray(weights)[idx]

        boot_stats[b] = estimate_mean(
            boot_data,
            estimator,
            trim_param=trim_param,
            winsor_limits=winsor_limits,
            weights=boot_weights,
        )

    return np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
