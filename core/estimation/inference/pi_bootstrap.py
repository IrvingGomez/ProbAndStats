# stats/inference/pi_bootstrap.py

import numpy as np


def pi_bootstrap(
    *,
    data,
    alpha,
    B,
):
    """
    Bootstrap prediction interval:
    average of bootstrap quantiles.
    """
    n = len(data)

    boot_intervals = np.array([
        np.quantile(
            np.random.choice(data, size=n, replace=True),
            [alpha / 2, 1 - alpha / 2],
        )
        for _ in range(B)
    ])

    return boot_intervals.mean(axis=0)
