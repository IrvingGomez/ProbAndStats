from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.linear_regression import run_linear_regression as _run_linear_regression


def _select_working_dataframe(
    df: Optional[pd.DataFrame],
    filtered_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Use the filtered dataframe if it is non-empty; otherwise fall back to the
    original dataframe. This mirrors the behaviour used in other tabs.
    """
    if df is None:
        raise ValueError("No dataset loaded.")

    if filtered_df is not None and not filtered_df.empty:
        return filtered_df

    if df.empty:
        raise ValueError("The dataset is empty.")

    return df


def _parse_confidence_level(text: str) -> float:
    """
    Parse a confidence level like '0.95' into an alpha value for statsmodels.

    Returns
    -------
    alpha : float
        Significance level (e.g. 0.05 for a 95% confidence level).
    """
    s = str(text).strip()
    if not s:
        raise ValueError("Confidence level is required (e.g. 0.95).")
    try:
        level = float(s)
    except ValueError as exc:
        raise ValueError("Confidence level must be a numeric value between 0 and 1.") from exc

    if not (0 < level < 1):
        raise ValueError("Confidence level must be between 0 and 1 (e.g. 0.95).")

    # statsmodels expects alpha, not the confidence level itself
    return 1.0 - level


def _parse_range(text: str) -> Optional[np.ndarray]:
    """
    Parse a range string like '0, 10' into a numpy array suitable for predictions.

    Returns
    -------
    np.ndarray or None
        If the string is empty or only whitespace, returns None.
        Otherwise returns a 1-D array of 100 evenly spaced values between
        the parsed minimum and maximum.
    """
    s = str(text).strip()
    if not s:
        return None

    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError("Range must have the form 'min, max'.")

    try:
        lo = float(parts[0].strip())
        hi = float(parts[1].strip())
    except ValueError as exc:
        raise ValueError("Range values must be numeric (e.g. '0, 10').") from exc

    if lo >= hi:
        raise ValueError("Range minimum must be strictly less than the maximum.")

    return np.linspace(lo, hi, 100)


def run_linear_regression(
    *,
    df: Optional[pd.DataFrame],
    filtered_df: Optional[pd.DataFrame],
    formula_check: bool,
    formula_text: str,
    formula_latex: str,
    dependent_var: Optional[str],
    independent_vars: List[str],
    alpha_input: str,
    intercept: bool,
    graph_check: bool,
    graph_type: str,
    show_ci: bool,
    show_pi: bool,
    fit_to_obs: bool,
    x_range_text: str,
    round_digits: int = 4,
) -> Tuple[str, pd.DataFrame, Optional[Figure]]:
    """
    High-level controller used by the Linear Regression tab.

    This function takes raw user input from the UI, performs validation and
    parsing, calls the stats layer, and returns a tuple:

        (summary_html, params_df_rounded, figure)

    Any exceptions should be caught in the tab layer and turned into user-
    facing error messages.
    """
    working_df = _select_working_dataframe(df, filtered_df)

    if dependent_var is None or dependent_var == "":
        raise ValueError("Please select a dependent variable.")

    if not independent_vars:
        raise ValueError("Please select at least one independent variable.")

    # For the "Simple Regression" graph we require exactly one independent variable.
    if graph_check and graph_type == "Simple Regression" and len(independent_vars) != 1:
        raise ValueError(
            "The 'Simple Regression' graph is only available when exactly one "
            "independent variable is selected."
        )

    # Parse confidence level
    alpha = _parse_confidence_level(alpha_input)

    # Parse X range only when needed: Simple Regression + graph + not fit_to_obs
    x_vector = None
    if graph_check and graph_type == "Simple Regression" and not fit_to_obs:
        x_vector = _parse_range(x_range_text)

    summary_html, params_df, fig = _run_linear_regression(
        df=working_df,
        formula_check=formula_check,
        formula_text=formula_text,
        formula_latex=formula_latex,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        alpha=alpha,
        intercept=intercept,
        create_graph=graph_check,
        graph_type=graph_type,
        show_ci=show_ci,
        show_pi=show_pi,
        fit_to_obs=fit_to_obs,
        x_vector=x_vector,
    )

    # Rounding happens here, not in the stats layer.
    params_df_rounded = params_df.round(round_digits)

    return summary_html, params_df_rounded, fig
