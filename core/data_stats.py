import pandas as pd
import numpy as np
import gradio as gr
from pathlib import Path

ROUND = 4

def load_dataset(file):
    """
    Load CSV or Excel file.
    Returns:
        df, status_message
    """
    if file is None:
        return None, "No file uploaded."

    try:
        path = Path(file.name)

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            return None, "Unsupported file format."

        return df, f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns."

    except Exception as e:
        return None, f"Error loading file: {e}"


def dataset_summary(df: pd.DataFrame):
    if df is None:
        return None

    summary = (
        df.describe(include="all")
        .transpose()
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    # Add unique counts explicitly
    summary["unique"] = df.nunique(dropna=True).values

    # Desired column order
    desired_order = [
        "variable",
        "count",
        "unique",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    summary = summary[[c for c in desired_order if c in summary.columns]]

    # ---- IMPORTANT PART ----
    # Format numeric columns as strings
    for col in summary.columns:
        if col not in ["variable", "count", "unique"]:
            summary[col] = summary[col].apply(
                lambda x: f"{x:.{ROUND}f}" if isinstance(x, (int, float)) else x
            )

    return summary


def variable_types(df):
    if df is None:
        return None

    return (
        df.dtypes
        .reset_index()
        .rename(columns={"index": "Variable", 0: "Type"})
    )


def column_choices_single(cols: list[str]):
    return gr.update(choices=cols, value=None)


def column_choices_multi(cols: list[str]):
    return gr.update(choices=cols, value=[])


def category_value_choices(df, col):
    if df is None or col is None or col not in df.columns:
        return gr.update(visible=False, choices=[], value=[])

    values = sorted(df[col].dropna().unique().tolist())

    return gr.update(
        visible=True,
        choices=values,
        value=[],   # MUST be a list for multiselect
    )


def infer_column_types(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return sorted(numeric_cols), sorted(categorical_cols)


def apply_category_filters(
    df,
    cat_cols,
    val1,
    val2,
    val3,
):
    if df is None:
        return None, "❌ No data loaded."

    if not cat_cols or all(not v for v in [val1, val2, val3]):
        return df.copy(), "⚠️ No filters selected. Using full dataset."

    filtered_df = df.copy()

    values = [val1, val2, val3]

    for col, selected_vals in zip(cat_cols[:3], values):
        if selected_vals:
            filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

    return filtered_df, f"✅ Filter applied. Rows remaining: {len(filtered_df)}"

def reclassify_as_categorical(state, column):
    if column and column in state.numeric_cols:
        state.numeric_cols.remove(column)
        state.categorical_cols.append(column)
        state.active_filters = {}   # reset filters
        return True, f"Column '{column}' reclassified as categorical."
    return False, f"Column '{column}' is not numeric."


def reclassify_as_numeric(state, column):
    if column and column in state.categorical_cols:
        state.categorical_cols.remove(column)
        state.numeric_cols.append(column)
        state.active_filters = {}   # reset filters
        return True, f"Column '{column}' reclassified as numeric."
    return False, f"Column '{column}' is not categorical."
