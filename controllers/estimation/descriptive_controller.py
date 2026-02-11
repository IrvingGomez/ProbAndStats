# ui/controllers/estimation/descriptive_controller.py

import pandas as pd
from core.estimation.descriptive import compute_descriptive_statistics


def run_descriptive_statistics(
    *,
    df: pd.DataFrame,
    column: str,
    quantile_probs: list[float],
    trim_alpha: float | None,
    winsor_limits: tuple[float, float] | None,
    weights_col: str | None,
    round_digits: int,
) -> pd.DataFrame:

    if df is None:
        raise ValueError("No dataset loaded.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    series = df[column].dropna()

    if series.empty:
        raise ValueError("Selected column has no valid data.")

    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("Selected column must be numeric.")
    
    weights = None

    if weights_col:
        if weights_col not in df.columns:
            raise ValueError(f"Weights column '{weights_col}' not found.")

        weights = df.loc[series.index, weights_col]

        if not pd.api.types.is_numeric_dtype(weights):
            raise ValueError("Weights must be numeric.")

        if (weights < 0).any():
            raise ValueError("Weights must be non-negative.")

    stats_df = compute_descriptive_statistics(
        data=series.values,
        quantile_probs=quantile_probs,
        trim_alpha=trim_alpha,
        winsor_limits=winsor_limits,
        weights=weights.values if weights is not None else None,
    )

    #numeric_cols = stats_df.select_dtypes("number").columns
    #stats_df[numeric_cols] = stats_df[numeric_cols].round(round_digits)

    stats_df[["Value", "Bias Corrected"]] = stats_df[["Value", "Bias Corrected"]].round(round_digits)

    return stats_df