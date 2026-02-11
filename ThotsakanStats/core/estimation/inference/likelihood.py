# stats/inference/likelihood.py

import numpy as np


def relative_log_likelihood(
    *,
    data,
    mu,
    sigma,
    sigma_hat,
):
    """
    Relative log-likelihood for Normal model.

    ℓ(μ,σ) − ℓ(μ̂,σ̂)
    """
    n = len(data)
    xbar = np.mean(data)

    return n * (
        np.log(sigma_hat / sigma)
        + 0.5 * (
            1
            - (np.mean(data**2) - 2 * mu * xbar + mu**2) / sigma**2
        )
    )


def relative_likelihood(
    *,
    data,
    mu,
    sigma,
    sigma_hat,
):
    return np.exp(
        relative_log_likelihood(
            data=data,
            mu=mu,
            sigma=sigma,
            sigma_hat=sigma_hat,
        )
    )
