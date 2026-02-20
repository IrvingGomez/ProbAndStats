from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from core.estimation.inference.estimators import (
    estimate_mean,
    estimate_median,
    estimate_sigma,
)
from core.estimation.inference.ci import (
    ci_mean_analytic,
    ci_mean_bootstrap,
    ci_median_analytic,
    ci_median_bootstrap,
)
from core.estimation.inference.pi import (
    pi_mean,
    pi_median,
    pi_iqr,
    pi_bootstrap,
)
from core.estimation.graphical_analysis import (
    plot_histogram_with_overlays,
    plot_ecdf,
)


# ---------------------------------------------------------------------
# Utilities (aligned with inference_controller)
# ---------------------------------------------------------------------
def select_distribution(mean_estimator: str, sigma_estimator: str) -> str:
    if (
        mean_estimator == "Sample Mean"
        and sigma_estimator == "Deviation (1 ddof)"
    ):
        return "t"
    return "norm"


def validate_deviation_estimator(*, sigma_estimator: str, n: int):
    if sigma_estimator == "Range (bias corrected)" and n > 25:
        raise ValueError(
            "Range-based confidence intervals require n ≤ 25. "
            "Use another estimator or bootstrap."
        )


def _prepare_series(
    df: pd.DataFrame,
    column: str,
    weights_col: Optional[str],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if df is None:
        raise ValueError("No data loaded. Please load a dataset first.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe.")

    series = df[column].dropna()
    if series.empty:
        raise ValueError(f"Column '{column}' has no non-missing values.")

    weights = None
    if weights_col is not None:
        if weights_col not in df.columns:
            raise ValueError(
                f"Weights column '{weights_col}' not found in the dataframe."
            )
        weights_series = df[weights_col].reindex(series.index).dropna()
        common_idx = series.index.intersection(weights_series.index)
        series = series.loc[common_idx]
        weights_series = weights_series.loc[common_idx]
        weights = weights_series.to_numpy()

    return series.to_numpy(), weights


def run_graphical_analysis(
    *,
    df: pd.DataFrame,
    column: str,
    graph_type: str,
    # Histogram / PMF controls
    add_kde: bool,
    add_data: bool,
    add_normal: bool,
    add_ci: bool,
    ci_choice: str,
    add_pi: bool,
    pi_choice: str,
    # Estimators
    mean_estimator: str,
    median_estimator: str,
    sigma_estimator: str,
    trim_param,
    winsor_limits,
    weights_col: Optional[str],
    # Normal μ source
    normal_mu_source: str,
    # Bootstrap options
    bootstrap_mean: bool,
    bootstrap_median: bool,
    bootstrap_sigma: bool,
    bootstrap_prediction: bool,
    bootstrap_samples: int,
    # CI/PI confidence level
    ci_pi_conf_level: float,
    # ECDF controls
    ecdf_add_conf: bool,
    ecdf_conf_level: float,
    ecdf_add_normal: bool,
):
    data, weights = _prepare_series(df, column, weights_col)

    if not (0.0 < ci_pi_conf_level < 1.0):
        raise ValueError("Confidence level for CI/PI must be in (0, 1).")

    if graph_type in ("Histogram", "Empirical Probability Mass Function"):
        return _run_hist_or_pmf(
            data=data,
            var_name=column,
            graph_type=graph_type,
            add_kde=add_kde,
            add_data=add_data,
            add_normal=add_normal,
            add_ci=add_ci,
            ci_choice=ci_choice,
            add_pi=add_pi,
            pi_choice=pi_choice,
            mean_estimator=mean_estimator,
            median_estimator=median_estimator,
            sigma_estimator=sigma_estimator,
            trim_param=trim_param,
            winsor_limits=winsor_limits,
            weights=weights,
            normal_mu_source=normal_mu_source,
            bootstrap_mean=bootstrap_mean,
            bootstrap_median=bootstrap_median,
            bootstrap_sigma=bootstrap_sigma,
            bootstrap_prediction=bootstrap_prediction,
            bootstrap_samples=bootstrap_samples,
            ci_pi_conf_level=ci_pi_conf_level,
        )

    if graph_type == "Empirical Cumulative Distribution Function (ECDF)":
        return _run_ecdf(
            data=data,
            var_name=column,
            ecdf_add_conf=ecdf_add_conf,
            ecdf_conf_level=ecdf_conf_level,
            ecdf_add_normal=ecdf_add_normal,
            mean_estimator=mean_estimator,
            median_estimator=median_estimator,
            sigma_estimator=sigma_estimator,
            trim_param=trim_param,
            winsor_limits=winsor_limits,
            weights=weights,
            normal_mu_source=normal_mu_source,
        )

    raise ValueError(f"Unknown graph type: {graph_type}")


def _run_hist_or_pmf(
    *,
    data: np.ndarray,
    var_name: str,
    graph_type: str,
    add_kde: bool,
    add_data: bool,
    add_normal: bool,
    add_ci: bool,
    ci_choice: str,
    add_pi: bool,
    pi_choice: str,
    mean_estimator: str,
    median_estimator: str,
    sigma_estimator: str,
    trim_param,
    winsor_limits,
    weights: Optional[np.ndarray],
    normal_mu_source: str,
    bootstrap_mean: bool,
    bootstrap_median: bool,
    bootstrap_sigma: bool,
    bootstrap_prediction: bool,
    bootstrap_samples: int,
    ci_pi_conf_level: float,
):
    alpha = 1.0 - ci_pi_conf_level

    n = len(data)
    validate_deviation_estimator(
        sigma_estimator=sigma_estimator,
        n=n,
    )

    ci_mean_interval = None
    ci_median_interval = None
    pi_interval = None
    hat_mu = None
    hat_sigma = None

    need_intervals = add_ci or add_pi or add_normal

    if need_intervals:
        # --- Parameters for Normal overlay ---
        if add_normal:
            if normal_mu_source == "Mean-based CI":
                hat_mu = estimate_mean(
                    data,
                    mean_estimator,
                    trim_param=trim_param,
                    winsor_limits=winsor_limits,
                    weights=weights,
                )
            else:
                hat_mu = estimate_median(data, median_estimator)

            hat_sigma = estimate_sigma(
                data=data,
                estimator=sigma_estimator,
            )

        # --- Confidence intervals ---
        if add_ci:
            dist = select_distribution(mean_estimator, sigma_estimator)

            # CI for mean
            if bootstrap_mean:
                ci_mean_interval = ci_mean_bootstrap(
                    data=data,
                    estimator=mean_estimator,
                    alpha=alpha,
                    trim_param=trim_param,
                    winsor_limits=winsor_limits,
                    weights=weights,
                    B=bootstrap_samples,
                )
            else:
                ci_mean_interval = ci_mean_analytic(
                    data=data,
                    estimator=mean_estimator,
                    alpha=alpha,
                    dist=dist,
                    sigma_estimator=sigma_estimator,
                    trim_param=trim_param,
                    winsor_limits=winsor_limits,
                    weights=weights,
                )

            # CI for median
            if bootstrap_median:
                ci_median_interval = ci_median_bootstrap(
                    data=data,
                    alpha=alpha,
                    estimator=median_estimator,
                    B=bootstrap_samples,
                )
            else:
                ci_median_interval = ci_median_analytic(
                    data=data,
                    alpha=alpha,
                    estimator=median_estimator,
                    sigma_estimator=sigma_estimator,
                )

            # Respect user choice (Mean / Median / Both)
            if ci_choice == "Mean":
                ci_median_interval = None
            elif ci_choice == "Median":
                ci_mean_interval = None

        # --- Prediction intervals ---
        if add_pi:
            dist = select_distribution(mean_estimator, sigma_estimator)
            if pi_choice == "Mean":
                pi_interval = pi_mean(
                    data=data,
                    alpha=alpha,
                    estimator=mean_estimator,
                    dist=dist,
                    sigma_estimator=sigma_estimator,
                    trim_param=trim_param,
                    winsor_limits=winsor_limits,
                    weights=weights,
                )
            elif pi_choice == "Median":
                pi_interval = pi_median(
                    data=data,
                    alpha=alpha,
                    estimator=median_estimator,
                    sigma_estimator=sigma_estimator,
                )
            elif pi_choice == "IQR":
                pi_interval = pi_iqr(
                    data=data,
                    alpha=alpha,
                )
            elif pi_choice == "Bootstrap":
                if not bootstrap_prediction:
                    raise ValueError(
                        "To use the Bootstrap prediction interval, enable the "
                        "'Bootstrap Prediction' option in the estimator settings."
                    )
                pi_interval = pi_bootstrap(
                    data=data,
                    alpha=alpha,
                    B=bootstrap_samples,
                )
            else:
                raise ValueError(
                    f"Unknown prediction-interval choice: {pi_choice}"
                )

    fig = plot_histogram_with_overlays(
        data=data,
        graph_type=graph_type,
        var_name=var_name,
        add_kde=add_kde,
        add_data=add_data,
        add_normal=add_normal,
        hat_mu=hat_mu,
        hat_sigma=hat_sigma,
        ci_mean_interval=ci_mean_interval,
        ci_median_interval=ci_median_interval,
        pi_interval=pi_interval,
    )

    return fig


def _run_ecdf(
    *,
    data: np.ndarray,
    var_name: str,
    ecdf_add_conf: bool,
    ecdf_conf_level: float,
    ecdf_add_normal: bool,
    mean_estimator: str,
    median_estimator: str,
    sigma_estimator: str,
    trim_param,
    winsor_limits,
    weights: Optional[np.ndarray],
    normal_mu_source: str,
):
    if not (0.0 < ecdf_conf_level < 1.0):
        raise ValueError("ECDF confidence level must be in (0, 1).")

    alpha = 1.0 - ecdf_conf_level

    n = len(data)
    validate_deviation_estimator(
        sigma_estimator=sigma_estimator,
        n=n,
    )

    hat_mu = None
    hat_sigma = None

    if ecdf_add_normal:
        if normal_mu_source == "Mean-based CI":
            hat_mu = estimate_mean(
                data,
                mean_estimator,
                trim_param=trim_param,
                winsor_limits=winsor_limits,
                weights=weights,
            )
        else:
            hat_mu = estimate_median(data, median_estimator)

        hat_sigma = estimate_sigma(
            data=data,
            estimator=sigma_estimator,
        )

    fig = plot_ecdf(
        data=data,
        var_name=var_name,
        alpha=alpha,
        add_conf_band=ecdf_add_conf,
        add_normal=ecdf_add_normal,
        hat_mu=hat_mu,
        hat_sigma=hat_sigma,
    )
    return fig
