# stats/inference/ci_median.py

import numpy as np
from scipy.stats import norm

from .estimators import estimate_sigma


def ci_median_analytic(
    *,
    data,
    alpha,
    sigma_estimator,
):
    """
    Asymptotic CI for the median under Normality.

    Uses:
        Var(median) ≈ (π / 2) * σ² / n

    σ is computed with the user-chosen deviation estimator.
    """
    n = len(data)
    median_hat = np.median(data)

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
    B,
):
    n = len(data)

    boot_stats = np.array([
        np.median(np.random.choice(data, size=n, replace=True))
        for _ in range(B)
    ])

    return np.quantile(boot_stats, [alpha / 2, 1 - alpha / 2])
