from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from core.hypothesis_tests import (
    one_sample_ttest,
    two_sample_ttest,
    variance_test,
    one_way_anova,
)

ROUND = 4


def _round_table(table: pd.DataFrame, decimals: int = ROUND) -> pd.DataFrame:
    """Round only numeric columns of the result table."""
    if table is None:
        return table
    tbl = table.copy()
    num_cols = tbl.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        tbl[num_cols] = tbl[num_cols].round(decimals)
    return tbl

def _ensure_numeric_series(df: pd.DataFrame, column: str) -> np.ndarray:
    if df is None:
        raise ValueError("No dataset loaded.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    series = df[column].dropna()
    if series.empty:
        raise ValueError("No valid data in the selected column.")
    return series.to_numpy()


def _materialize_group(
    df: pd.DataFrame,
    numeric_col: str,
    cat_col: str | None,
    cat_vals: Iterable[str],
) -> np.ndarray:
    if cat_col is None:
        raise ValueError("No categorical column selected.")

    if cat_col not in df.columns:
        raise ValueError(f"Categorical column '{cat_col}' not found in the dataset.")

    # Cast selected values to the actual dtype of the column
    if cat_vals is None:
        values = []
    else:
        values = list(cat_vals)

    if not values:
        raise ValueError(f"No categories selected for column '{cat_col}'.")

    cat_series = pd.Series(values).astype(df[cat_col].dtype)
    mask = df[cat_col].isin(cat_series)
    series = df.loc[mask, numeric_col].dropna()

    if series.empty:
        raise ValueError("One or more groups are empty after filtering.")
    return series.to_numpy()


def run_hypothesis_testing(
    *,
    df: pd.DataFrame | None,
    numeric_col: str,
    hypo_test: str,
    mu0_text: str,
    alternative: str,
    include_graph: bool,
    bootstrap_samples: int,
    cat_col1: str | None,
    cat_vals1: list[str],
    name_group1: str,
    cat_col2: str | None,
    cat_vals2: list[str],
    name_group2: str,
    cat_col3: str | None,
    cat_vals3: list[str],
    plot_type: str,
    correction: bool,
    test_type: str,
) -> Tuple[pd.DataFrame, object | None]:
    """
    High-level dispatcher used by the Hypothesis Testing tab.

    Returns:
        (result_table, figure_or_none)
    """
    if df is None:
        raise ValueError("No dataset loaded.")

    # Common numeric data check
    _ = _ensure_numeric_series(df, numeric_col)

    # ------------------------------------------------------------
    # One-sample t-test
    # ------------------------------------------------------------
    if hypo_test == "One sample Student's t-test":
        if not mu0_text.strip():
            raise ValueError("μ₀ must be specified for the one-sample t-test.")
        try:
            mu0 = float(mu0_text)
        except Exception:
            raise ValueError("μ₀ must be a numeric value.")

        sample = df[numeric_col].dropna().to_numpy()

        table, fig = one_sample_ttest(
            sample=sample,
            mu0=mu0,
            alternative=alternative,
            numeric_col=numeric_col,
            bootstrap_samples=bootstrap_samples,
            include_graph=include_graph,
        )
        table = _round_table(table)
        return table, fig

    # ------------------------------------------------------------
    # Two-sample t-test
    # ------------------------------------------------------------
    if hypo_test == "Two samples Student's t-test":
        group1 = _materialize_group(df, numeric_col, cat_col1, cat_vals1)
        group2 = _materialize_group(df, numeric_col, cat_col2, cat_vals2)

        # If names are empty, fall back to defaults
        name1 = name_group1 or "Group 1"
        name2 = name_group2 or "Group 2"

        table, fig = two_sample_ttest(
            group1=group1,
            group2=group2,
            numeric_col=numeric_col,
            name_group1=name1,
            name_group2=name2,
            alternative=alternative,
            correction=correction,
            plot_type=plot_type,
            bootstrap_samples=bootstrap_samples,
            include_graph=include_graph,
        )
        table = _round_table(table)
        return table, fig

    # ------------------------------------------------------------
    # Equal variance between two groups
    # ------------------------------------------------------------
    if hypo_test == "Equal variance between two groups":
        group1 = _materialize_group(df, numeric_col, cat_col1, cat_vals1)
        group2 = _materialize_group(df, numeric_col, cat_col2, cat_vals2)

        name1 = name_group1 or "Group 1"
        name2 = name_group2 or "Group 2"

        table, fig = variance_test(
            group1=group1,
            group2=group2,
            name_group1=name1,
            name_group2=name2,
            test_type=test_type,
            include_graph=include_graph,
            bootstrap_samples=bootstrap_samples,
        )
        table = _round_table(table)
        return table, fig

    # ------------------------------------------------------------
    # One-way ANOVA
    # ------------------------------------------------------------
    if hypo_test == "One-way ANOVA":
        if cat_col3 is None:
            raise ValueError("A categorical column must be selected for ANOVA.")

        if cat_col3 not in df.columns:
            raise ValueError(
                f"Categorical column '{cat_col3}' not found in the dataset."
            )

        if not cat_vals3:
            raise ValueError("At least one category must be selected for ANOVA.")

        cat_series = pd.Series(cat_vals3).astype(df[cat_col3].dtype)
        data_group = df[df[cat_col3].isin(cat_series)][[numeric_col, cat_col3]].dropna()

        table, fig = one_way_anova(
            data_group=data_group,
            numeric_col=numeric_col,
            cat_col=cat_col3,
        )
        table = _round_table(table)
        return table, fig

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------
    raise ValueError(f"Unknown hypothesis test: {hypo_test}")
