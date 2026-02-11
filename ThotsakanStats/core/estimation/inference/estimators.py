# stats/inference/estimators.py

import numpy as np
from scipy.stats import trim_mean, iqr, median_abs_deviation, norm
from scipy.stats.mstats import gmean, hmean, winsorize


# ---------------
# Mean estimators
# ---------------

def estimate_mean(
    data,
    estimator,
    *,
    trim_param=None,
    winsor_limits=None,
    weights=None,
):
    data = np.asarray(data)

    if estimator == "Sample Mean":
        return np.mean(data)

    if estimator == "Geometric Mean":
        if np.any(data <= 0):
            raise ValueError("Geometric mean requires positive data")
        return gmean(data)

    if estimator == "Harmonic Mean":
        if np.any(data <= 0):
            raise ValueError("Harmonic mean requires positive data")
        return hmean(data)

    if estimator == "Trimmed Mean":
        if trim_param is None:
            raise ValueError("trim_param must be provided")

        try:
            trim_param = float(trim_param)
        except Exception:
            raise ValueError("trim_param must be a numeric value")

        if not (0 < trim_param < 0.5):
            raise ValueError("trim_param must be in (0, 0.5)")

        return trim_mean(data, trim_param)

    if estimator == "Interquartile Mean":
        return trim_mean(data, 0.25)

    if estimator == "Winsorized Mean":
        if winsor_limits is None:
            raise ValueError("winsor_limits must be provided")

        # --------------------------------------------------
        # Parse winsor limits
        # --------------------------------------------------
        if isinstance(winsor_limits, str):
            parts = [p.strip() for p in winsor_limits.split(",") if p.strip()]
            try:
                parts = [float(p) for p in parts]
            except ValueError:
                raise ValueError(
                    "winsor_limits must be numeric (e.g. '0.1' or '0.05,0.2')"
                )

            if len(parts) == 1:
                limits = parts[0]
            elif len(parts) == 2:
                limits = (parts[0], parts[1])
            else:
                raise ValueError(
                    "winsor_limits must have one or two values"
                )

        elif isinstance(winsor_limits, (list, tuple)):
            if len(winsor_limits) != 2:
                raise ValueError(
                    "winsor_limits list/tuple must have exactly two values"
                )
            limits = (float(winsor_limits[0]), float(winsor_limits[1]))

        else:
            limits = float(winsor_limits)

        # --------------------------------------------------
        # Validate bounds
        # --------------------------------------------------
        if isinstance(limits, tuple):
            if not (0 <= limits[0] < 0.5 and 0 <= limits[1] < 0.5):
                raise ValueError("winsor_limits must be in [0, 0.5)")
        else:
            if not (0 <= limits < 0.5):
                raise ValueError("winsor_limits must be in [0, 0.5)")

        # --------------------------------------------------
        # Compute winsorized mean
        # --------------------------------------------------
        wins_data = winsorize(data, limits=limits)
        return np.mean(wins_data)

    if estimator == "Weighted Mean":
        if weights is None:
            raise ValueError("weights must be provided for weighted mean")

        weights = np.asarray(weights)

        if len(weights) != len(data):
            raise ValueError("weights must have same length as data")

        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

        return np.average(data, weights=weights)

    raise ValueError(f"Unknown mean estimator: {estimator}")


# --------------------
# Deviation estimators
# --------------------

def estimate_sigma(
    data,
    estimator,
):
    """
    Return a bias-corrected estimate of σ based on the chosen deviation
    estimator name.
    """
    data = np.asarray(data)
    n = len(data)

    if n < 2:
        raise ValueError("At least two observations are required to estimate deviation.")

    # 1) Classical sample standard deviation (ddof=1)
    if estimator == "Deviation (1 ddof)":
        return np.std(data, ddof=1)

    # 2) Range-based estimator, bias-corrected by d2(n)
    if estimator == "Range (bias corrected)":
        R = np.max(data) - np.min(data)
        return R / d2(n)

    # 3) IQR-based estimator: σ ≈ IQR / (2 Φ⁻¹(0.75))
    if estimator == "IQR (bias corrected)":
        IQR = iqr(data)
        return IQR / (2 * norm.ppf(0.75))

    # 4) MAD-based estimator: σ ≈ MAD / Φ⁻¹(0.75)
    if estimator == "MAD (bias corrected)":
        MAD = median_abs_deviation(data)
        return MAD / norm.ppf(0.75)

    # 5) AAD-based estimator: σ ≈ AAD * sqrt(π/2)
    if estimator == "AAD (bias corrected)":
        AAD = np.mean(np.abs(data - np.mean(data)))
        return AAD * np.sqrt(np.pi / 2)

    raise ValueError(f"Unknown deviation estimator: {estimator}")


def d2(n: int) -> float:
    """
    Bias-correction constant for the range-based σ estimator.
    Same table as in ci_deviation.py.
    """
    table = {
        2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
        7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078,
        11: 3.173, 12: 3.258, 13: 3.336, 14: 3.407,
        15: 3.472, 16: 3.532, 17: 3.588, 18: 3.640,
        19: 3.689, 20: 3.735, 21: 3.778, 22: 3.819,
        23: 3.858, 24: 3.895, 25: 3.931,
    }
    if n not in table:
        raise ValueError("Range-based estimator only supported for 2 ≤ n ≤ 25.")
    return table[n]
