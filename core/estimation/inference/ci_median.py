# stats/inference/ci_median.py

import numpy as np
from scipy.stats import norm

from .estimators import estimate_median, estimate_sigma


def ci_median_analytic(
    *,
    data,
    alpha,
    estimator,
    sigma_estimator,
):
    """
    Asymptotic CI for the median under Normality.

    Uses:
        Var(median) ≈ (π / 2) * σ² / n

    Median and σ are computed with user-chosen estimators.
    """
    n = len(data)
    median_hat = estimate_median(data, estimator)

    sigma_hat = estimate_sigma(
        data=data,
        estimator=sigma_estimator,
    )

    scale = sigma_hat * np.sqrt(np.pi / (2 * n))

    return (
        norm.ppf(alpha / 2, median_hat, scale),
        norm.ppf(1 - alpha / 2, median_hat, scale),
    )


def ci_median_bootstrap(
    *,
    data,
    alpha,
    estimator,
    B,
):
    n = len(data)

    boot_stats = np.array([
        estimate_median(np.random.choice(data, size=n, replace=True), estimator)
        for _ in range(B)
    ])

    return np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
