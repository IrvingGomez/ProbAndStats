import pandas as pd

from core.estimation.inference.ci import (
    ci_mean_analytic,
    ci_median_analytic,
    ci_deviation_analytic,
    ci_mean_bootstrap,
    ci_median_bootstrap,
    ci_deviation_bootstrap,
)

from core.estimation.inference.pi import (
    pi_mean,
    pi_median,
    pi_iqr,
    pi_bootstrap,
)

from core.estimation.inference.confidence_regions import confidence_regions
from core.estimation.inference.estimator_options import available_estimators


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def select_distribution(mean_estimator: str, sigma_estimator: str) -> str:
    if mean_estimator == "Sample Mean" and sigma_estimator == "Deviation (1 ddof)":
        return "t"
    return "norm"


def validate_deviation_estimator(*, sigma_estimator: str, n: int):
    if sigma_estimator == "Range (bias corrected)" and n > 25:
        raise ValueError(
            "Range-based confidence intervals require n ≤ 25. "
            "Use another estimator or bootstrap."
        )




def get_available_estimators(data):
    mean_choices, deviation_choices = available_estimators(data)
    return {
        "mean_estimators": mean_choices,
        "deviation_estimators": deviation_choices,
    }

# ---------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------

def run_confidence_intervals(
    *,
    data,
    alpha,
    mean_estimator,
    median_estimator,
    sigma_estimator,
    trim_param=None,
    winsor_limits=None,
    weights=None,
    bootstrap_mean=False,
    bootstrap_median=False,
    bootstrap_deviation=False,
    bootstrap_samples=1000,
):
    n = len(data)

    validate_deviation_estimator(
        sigma_estimator=sigma_estimator,
        n=n,
    )

    dist = select_distribution(mean_estimator, sigma_estimator)

    # ---------------- Mean ----------------
    if bootstrap_mean:
        mean_ci = ci_mean_bootstrap(
            data=data,
            estimator=mean_estimator,
            alpha=alpha,
            B=bootstrap_samples,
            trim_param=trim_param,
            winsor_limits=winsor_limits,
            weights=weights,
        )
    else:
        mean_ci = ci_mean_analytic(
            data=data,
            estimator=mean_estimator,
            alpha=alpha,
            dist=dist,
            sigma_estimator=sigma_estimator,
            trim_param=trim_param,
            winsor_limits=winsor_limits,
            weights=weights,
        )

    # ---------------- Median ----------------
    if bootstrap_median:
        median_ci = ci_median_bootstrap(
            data=data,
            alpha=alpha,
            estimator=median_estimator,
            B=bootstrap_samples,
        )
    else:
        median_ci = ci_median_analytic(
            data=data,
            alpha=alpha,
            estimator=median_estimator,
            sigma_estimator=sigma_estimator,
        )

    # ---------------- Deviation ----------------
    if bootstrap_deviation:
        sigma_ci = ci_deviation_bootstrap(
            data=data,
            alpha=alpha,
            B=bootstrap_samples,
            estimator=sigma_estimator,
        )
    else:
        sigma_ci = ci_deviation_analytic(
            data=data,
            alpha=alpha,
            estimator=sigma_estimator,
        )

    table = pd.DataFrame(
        [
            ["Confidence", "Mean", *mean_ci],
            ["Confidence", "Median", *median_ci],
            ["Confidence", "Deviation", *sigma_ci],
        ],
        columns=["Interval Type", "Statistic", "Lower", "Upper"],
    )

    return table, mean_ci, sigma_ci, median_ci


# ---------------------------------------------------------------------
# Prediction Intervals
# ---------------------------------------------------------------------

def run_prediction_intervals(
    *,
    data,
    alpha,
    mean_estimator,
    median_estimator,
    sigma_estimator,
    trim_param=None,
    winsor_limits=None,
    weights=None,
    bootstrap=False,
    bootstrap_samples=1000,
):
    dist = select_distribution(mean_estimator, sigma_estimator)

    rows = []

    # Mean-based PI
    mean_pi = pi_mean(
        data=data,
        alpha=alpha,
        estimator=mean_estimator,
        dist=dist,
        sigma_estimator=sigma_estimator,
        trim_param=trim_param,
        winsor_limits=winsor_limits,
        weights=weights,
    )
    rows.append(["Prediction", "Mean", *mean_pi])

    # Median-based PI (uses same deviation estimator)
    median_pi = pi_median(
        data=data,
        alpha=alpha,
        estimator=median_estimator,
        sigma_estimator=sigma_estimator,
    )
    rows.append(["Prediction", "Median", *median_pi])

    # IQR-based PI
    iqr_pi = pi_iqr(
        data=data,
        alpha=alpha,
    )
    rows.append(["Prediction", "IQR", *iqr_pi])

    # Optional bootstrap PI
    if bootstrap:
        boot_pi = pi_bootstrap(
            data=data,
            alpha=alpha,
            B=bootstrap_samples,
        )
        rows.append(["Prediction", "Bootstrap", *boot_pi])

    return pd.DataFrame(
        rows,
        columns=["Interval Type", "Statistic", "Lower", "Upper"],
    )

# ---------------------------------------------------------------------
# Confidence Regions
# ---------------------------------------------------------------------

def run_confidence_regions(
    *,
    data,
    alpha,
    mean_estimator,
    median_estimator,
    sigma_estimator,
    trim_param,
    winsor_limits,
    weights,
    bootstrap_mean,
    bootstrap_median,
    bootstrap_deviation,
    bootstrap_samples,
    mu_ci_source,
    probs,
    eps_mu,
    eps_sigma,
    add_ci_box,
):
    """
    Use the CI machinery to compute CIs for mean, median and deviation,
    then choose which CI to use for μ (mean-based or median-based) and
    pass that CI plus the σ CI into the likelihood-based confidence
    regions function.
    """

    ci_table, mean_ci, sigma_ci, median_ci = run_confidence_intervals(
        data=data,
        alpha=alpha,
        mean_estimator=mean_estimator,
        median_estimator=median_estimator,
        sigma_estimator=sigma_estimator,
        trim_param=trim_param,
        winsor_limits=winsor_limits,
        weights=weights,
        bootstrap_mean=bootstrap_mean,
        bootstrap_median=bootstrap_median,
        bootstrap_deviation=bootstrap_deviation,
        bootstrap_samples=bootstrap_samples,
    )

    if mu_ci_source == "Median-based CI":
        mu_ci = median_ci
    else:
        # default: mean-based CI
        mu_ci = mean_ci

    fig = confidence_regions(
        data=data,
        mean_ci=mu_ci,
        sigma_ci=sigma_ci,
        probs=probs,
        eps_mu=eps_mu,
        eps_sigma=eps_sigma,
        add_ci_box=add_ci_box,
    )

    return fig


# ---------------------------------------------------------------------
# Combined Runner (used by UI)
# ---------------------------------------------------------------------

def run_intervals(
    *,
    data,
    alpha,
    mean_estimator,
    median_estimator,
    sigma_estimator,
    bootstrap_mean,
    bootstrap_median,
    bootstrap_deviation,
    bootstrap_samples,
):
    ci_table, mean_ci, sigma_ci, _ = run_confidence_intervals(
        data=data,
        alpha=alpha,
        mean_estimator=mean_estimator,
        median_estimator=median_estimator,
        sigma_estimator=sigma_estimator,
        bootstrap_mean=bootstrap_mean,
        bootstrap_median=bootstrap_median,
        bootstrap_deviation=bootstrap_deviation,
        bootstrap_samples=bootstrap_samples,
    )

    pi_table = run_prediction_intervals(
        data=data,
        alpha=alpha,
        mean_estimator=mean_estimator,
        median_estimator=median_estimator,
        sigma_estimator=sigma_estimator,
        bootstrap=bootstrap_mean,
        bootstrap_samples=bootstrap_samples,
    )

    combined = pd.concat([ci_table, pi_table], ignore_index=True)

    return ci_table, pi_table, combined
