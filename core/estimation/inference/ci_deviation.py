import numpy as np
from scipy.stats import chi2, norm
from scipy.stats import iqr, median_abs_deviation


# ============================================================
# Analytic Confidence Intervals
# ============================================================

def ci_deviation_analytic(
    *,
    data,
    alpha,
    estimator,
):

    n = len(data)

    if estimator == "Deviation (1 ddof)":
        return _ci_std_chi2(data, alpha)

    if estimator == "Range (bias corrected)":
        return _ci_sigma_from_range(data, alpha)

    if estimator == "IQR (bias corrected)":
        return _ci_iqr_asymptotic(data, alpha)

    if estimator == "MAD (bias corrected)":
        return _ci_mad_asymptotic(data, alpha)

    if estimator == "AAD (bias corrected)":
        return _ci_aad_asymptotic(data, alpha)

    raise ValueError(f"Unknown deviation estimator: {estimator}")


# -------------------------------
# χ² CI for σ (std, ddof=1)
# -------------------------------

def _ci_std_chi2(data, alpha):
    n = len(data)
    s = np.std(data, ddof=1)

    num = s * np.sqrt(n - 1)
    lo = num / np.sqrt(chi2.ppf(1 - alpha / 2, n - 1))
    hi = num / np.sqrt(chi2.ppf(alpha / 2, n - 1))

    return lo, hi


# -------------------------------
# Range-based CI (bias corrected)
# -------------------------------

def _ci_sigma_from_range(data, alpha):
    n = len(data)
    R = np.max(data) - np.min(data)

    d2_n = d2(n)
    d3_n = d3(n)

    z = norm.ppf(1 - alpha / 2)

    denom_lo = d2_n + z * d3_n
    denom_hi = d2_n - z * d3_n

    if denom_hi <= 0:
        raise ValueError("Invalid configuration: denominator ≤ 0")

    return R / denom_lo, R / denom_hi


# -------------------------------
# IQR (asymptotic)
# -------------------------------

def _ci_iqr_asymptotic(data, alpha):
    n = len(data)
    IQR = iqr(data)

    w = 2 * norm.ppf(0.75)
    k = np.sqrt(np.pi / (2 * np.exp(-norm.ppf(0.75) ** 2)))
    z = norm.ppf(1 - alpha / 2)

    return (
        IQR / (w + z * k / np.sqrt(n)),
        IQR / (w - z * k / np.sqrt(n)),
    )


# -------------------------------
# MAD (asymptotic)
# -------------------------------

def _ci_mad_asymptotic(data, alpha):
    n = len(data)
    MAD = median_abs_deviation(data)

    w = norm.ppf(0.75)
    k = np.sqrt(np.pi / (8 * np.exp(-norm.ppf(0.75) ** 2)))
    z = norm.ppf(1 - alpha / 2)

    return (
        MAD / (w + z * k / np.sqrt(n)),
        MAD / (w - z * k / np.sqrt(n)),
    )


# -------------------------------
# AAD (asymptotic)
# -------------------------------

def _ci_aad_asymptotic(data, alpha):
    n = len(data)
    AAD = np.mean(np.abs(data - np.mean(data)))

    w = np.sqrt(2 / np.pi)
    k = np.sqrt(1 - 2 / np.pi)
    z = norm.ppf(1 - alpha / 2)

    return (
        AAD / (w + z * k / np.sqrt(n)),
        AAD / (w - z * k / np.sqrt(n)),
    )


# ============================================================
# Bootstrap Confidence Intervals
# ============================================================

def ci_deviation_bootstrap(
    *,
    data,
    alpha,
    B,
    estimator,
):
    """
    Bootstrap CI for deviation estimators.
    """

    n = len(data)

    if estimator == "Deviation (1 ddof)":
        boot = np.array([
            np.std(np.random.choice(data, n, replace=True), ddof=1)
            for _ in range(B)
        ])
        return np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    if estimator == "Range (bias corrected)":
        boot = np.array([
            np.max(b := np.random.choice(data, n, replace=True)) - np.min(b)
            for _ in range(B)
        ])
        return np.quantile(boot, [alpha / 2, 1 - alpha / 2]) / d2(n)

    if estimator == "IQR (bias corrected)":
        boot = np.array([
            iqr(np.random.choice(data, n, replace=True))
            for _ in range(B)
        ])
        return np.quantile(boot, [alpha / 2, 1 - alpha / 2]) / (
            2 * norm.ppf(0.75)
        )

    if estimator == "MAD (bias corrected)":
        boot = np.array([
            median_abs_deviation(np.random.choice(data, n, replace=True))
            for _ in range(B)
        ])
        return np.quantile(boot, [alpha / 2, 1 - alpha / 2]) / norm.ppf(0.75)

    if estimator == "AAD (bias corrected)":
        boot = np.array([
            np.mean(np.abs(
                (b := np.random.choice(data, n, replace=True))
                - np.mean(b)
            ))
            for _ in range(B)
        ])
        return np.quantile(boot, [alpha / 2, 1 - alpha / 2]) * np.sqrt(np.pi / 2)

    raise ValueError(f"Unknown deviation estimator: {estimator}")


# ============================================================
# Bias-correction constants (monolith-compatible)
# ============================================================

def d2(n):
    table = {
        2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
        7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078,
        11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
        15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640,
        19: 3.689, 20: 3.735, 21: 3.778, 22: 3.819,
        23: 3.858, 24: 3.895, 25: 3.931,
    }
    return table[n]


def d3(n):
    table = {
        2: 0.852, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848,
        7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797,
        11: 0.787, 12: 0.778, 13: 0.770, 14: 0.763,
        15: 0.756, 16: 0.750, 17: 0.744, 18: 0.739,
        19: 0.734, 20: 0.729, 21: 0.724, 22: 0.720,
        23: 0.716, 24: 0.712, 25: 0.708,
    }
    return table[n]
