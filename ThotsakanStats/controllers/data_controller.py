import gradio as gr

from core.data_stats import (
    load_dataset,
    dataset_summary,
    variable_types,
    infer_column_types,
    apply_category_filters,
    reclassify_as_categorical,
    reclassify_as_numeric,
)


def wire_callbacks(
    *,
    file_input,
    status_output,

    # RAW DATA
    preview_checkbox,
    overview_checkbox,
    csv_preview,
    desc_output,
    dtypes_output,

    # RECLASSIFICATION
    num_to_cat,
    cat_to_num,
    fix_to_categorical_button,
    fix_to_numeric_button,
    fix_dtype_status,

    # FILTERS
    cat_filter_cols,
    cat_val_1,
    cat_val_2,
    cat_val_3,
    apply_filter_button,
    filter_status,

    # FILTERED DATA
    preview_checkbox_filter,
    overview_checkbox_filter,
    csv_preview_filter,
    desc_output_filter,
    dtypes_output_filter,

    state,
):
    # ==================================================
    # File upload
    # ==================================================
    def on_file_upload(file):
        df, status = load_dataset(file)

        if df is None:
            return (
                status,
                None, None, None,
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
            )

        numeric_cols, categorical_cols = infer_column_types(df)

        state.df = df
        state.filtered_df = df
        state.numeric_cols = numeric_cols
        state.categorical_cols = categorical_cols
        state.active_filters = {}
        state.overrides = {"num_to_cat": [], "cat_to_num": []}

        return (
            status,
            df,
            dataset_summary(df),
            variable_types(df),

            # Reclassification dropdowns
            gr.update(choices=numeric_cols, value=None),
            gr.update(choices=categorical_cols, value=None),

            # Filter columns (categorical only)
            gr.update(choices=categorical_cols, value=[]),
        )

    file_input.change(
        on_file_upload,
        inputs=file_input,
        outputs=[
            status_output,
            csv_preview,
            desc_output,
            dtypes_output,
            num_to_cat,
            cat_to_num,
            cat_filter_cols,
        ],
    )

    # ==================================================
    # Category value dropdowns (Filter 1–3)
    # ==================================================
    def update_category_filters(selected_columns):
        df = state.df

        if df is None or not selected_columns:
            return (
                gr.update(visible=False, choices=[], value=[]),
                gr.update(visible=False, choices=[], value=[]),
                gr.update(visible=False, choices=[], value=[]),
            )

        updates = []
        for i in range(3):
            if i < len(selected_columns):
                col = selected_columns[i]
                values = sorted(df[col].dropna().unique().tolist())
                updates.append(
                    gr.update(
                        visible=True,
                        choices=values,
                        value=[],
                    )
                )
            else:
                updates.append(
                    gr.update(visible=False, choices=[], value=[])
                )

        return tuple(updates)

    cat_filter_cols.change(
        update_category_filters,
        inputs=cat_filter_cols,
        outputs=[cat_val_1, cat_val_2, cat_val_3],
    )

    # ==================================================
    # Apply filters
    # ==================================================
    def on_apply_filter(cat_cols, v1, v2, v3):
        filtered_df, status = apply_category_filters(
            state.df,
            cat_cols,
            v1, v2, v3,
        )

        state.filtered_df = filtered_df
        state.active_filters = {
            col: vals
            for col, vals in zip(cat_cols[:3], [v1, v2, v3])
            if vals
        }

        return status

    apply_filter_button.click(
        on_apply_filter,
        inputs=[cat_filter_cols, cat_val_1, cat_val_2, cat_val_3],
        outputs=filter_status,
    )

    # ==================================================
    # RAW preview / summary
    # ==================================================
    preview_checkbox.change(
        lambda x: gr.update(visible=x),
        inputs=preview_checkbox,
        outputs=csv_preview,
    )

    overview_checkbox.change(
        lambda x: (
            gr.update(visible=x),
            gr.update(visible=x),
        ),
        inputs=overview_checkbox,
        outputs=[desc_output, dtypes_output],
    )

    # ==================================================
    # FILTERED preview / summary
    # ==================================================
    preview_checkbox_filter.change(
        lambda x: (
            gr.update(visible=x),
            state.filtered_df if x else None,
        ),
        inputs=preview_checkbox_filter,
        outputs=[csv_preview_filter, csv_preview_filter],
    )

    overview_checkbox_filter.change(
        lambda x: (
            gr.update(visible=x),
            gr.update(visible=x),
            dataset_summary(state.filtered_df) if x else None,
            variable_types(state.filtered_df) if x else None,
        ),
        inputs=overview_checkbox_filter,
        outputs=[
            desc_output_filter,
            dtypes_output_filter,
            desc_output_filter,
            dtypes_output_filter,
        ],
    )

    # ==================================================
    # Reclassification
    # ==================================================
    def on_fix_to_categorical(column):
        _, msg = reclassify_as_categorical(state, column)
        return (
            gr.update(choices=state.categorical_cols, value=[]),
            gr.update(choices=state.numeric_cols, value=None),
            gr.update(choices=state.categorical_cols, value=None),
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def on_fix_to_numeric(column):
        _, msg = reclassify_as_numeric(state, column)
        return (
            gr.update(choices=state.categorical_cols, value=[]),
            gr.update(choices=state.numeric_cols, value=None),
            gr.update(choices=state.categorical_cols, value=None),
            msg,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    fix_to_categorical_button.click(
        on_fix_to_categorical,
        inputs=num_to_cat,
        outputs=[
            cat_filter_cols,
            num_to_cat,
            cat_to_num,
            fix_dtype_status,
            cat_val_1,
            cat_val_2,
            cat_val_3,
        ],
    )

    fix_to_numeric_button.click(
        on_fix_to_numeric,
        inputs=cat_to_num,
        outputs=[
            cat_filter_cols,
            num_to_cat,
            cat_to_num,
            fix_dtype_status,
            cat_val_1,
            cat_val_2,
            cat_val_3,
        ],
    )
