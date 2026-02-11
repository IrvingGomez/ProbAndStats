import gradio as gr

from controllers.data_controller import wire_callbacks


def build(state):
    gr.Markdown("## 📁 Data Management")

    # ==================================================
    # File upload + status + preview toggles
    # ==================================================
    with gr.Row():
        file_input = gr.File(
            label="Upload CSV or Excel",
            file_types=[".csv", ".xlsx"],
            elem_classes=["data-selector"],
        )

        status_output = gr.Textbox(
            label="Status",
            interactive=False,
        )

        preview_checkbox = gr.Checkbox(
            label="Show Preview",
            value=False,
        )

        overview_checkbox = gr.Checkbox(
            label="Show Dataset Summary",
            value=False,
        )

    # --- Original data outputs ---
    csv_preview = gr.Dataframe(
        label="CSV Preview",
        visible=False,
    )

    desc_output = gr.Dataframe(
        label="Descriptive Summary",
        visible=False,
    )

    dtypes_output = gr.Dataframe(
        label="Variable Types",
        visible=False,
    )

    # ==================================================
    # Filter Data accordion
    # ==================================================
    with gr.Accordion("➖ Filter Data", open=False):

        gr.Markdown(
            "ℹ️ **Filtering is limited to a maximum of 3 categorical variables at a time.**"
        )

        with gr.Row():
            cat_col_dropdown = gr.Dropdown(
                label="Select Categorical Columns for Filter",
                interactive=True,
                multiselect=True,
                max_choices=3,
                value=[],
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            cat_val_1 = gr.Dropdown(
                label="Categories for Filter 1",
                interactive=True,
                multiselect=True,
                visible=False,
                value=[],
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            cat_val_2 = gr.Dropdown(
                label="Categories for Filter 2",
                interactive=True,
                multiselect=True,
                visible=False,
                value=[],
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            cat_val_3 = gr.Dropdown(
                label="Categories for Filter 3",
                interactive=True,
                multiselect=True,
                visible=False,
                value=[],
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

        with gr.Row():
            apply_filter_button = gr.Button("🚀 Apply Filter")

            filter_status = gr.Textbox(
                label="Filter Status",
                interactive=False,
            )

            preview_checkbox_filter = gr.Checkbox(
                label="Show Preview",
                value=False,
            )

            overview_checkbox_filter = gr.Checkbox(
                label="Show Dataset Summary",
                value=False,
            )

    # --- Filtered data outputs ---
    csv_preview_filter = gr.Dataframe(
        label="CSV Preview",
        visible=False,
    )

    desc_output_filter = gr.Dataframe(
        label="Descriptive Summary",
        visible=False,
    )

    dtypes_output_filter = gr.Dataframe(
        label="Variable Types",
        visible=False,
    )

    # ==================================================
    # Fix Variable Type accordion
    # ==================================================
    with gr.Accordion("🛠️ Fix Variable Type", open=False):

        with gr.Row():
            num_override_dropdown = gr.Dropdown(
                label="Reclassify Numeric Column as Categorical",
                interactive=True,
                multiselect=False,
                value=None,
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            fix_to_categorical_button = gr.Button(
                "Reclassify as Categorical"
            )

            cat_override_dropdown = gr.Dropdown(
                label="Reclassify Categorical Column as Numeric",
                interactive=True,
                multiselect=False,
                value=None,
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            fix_to_numeric_button = gr.Button(
                "Reclassify as Numeric"
            )

            fix_dtype_status = gr.Textbox(
                label="Status",
                interactive=False,
            )

    wire_callbacks(
        # --------------------------
        # File / state
        # --------------------------
        file_input=file_input,
        status_output=status_output,
        state=state,

        # --------------------------
        # Raw data
        # --------------------------
        preview_checkbox=preview_checkbox,
        overview_checkbox=overview_checkbox,
        csv_preview=csv_preview,
        desc_output=desc_output,
        dtypes_output=dtypes_output,

        # --------------------------
        # Reclassification
        # --------------------------
        num_to_cat=num_override_dropdown,
        cat_to_num=cat_override_dropdown,
        fix_to_categorical_button=fix_to_categorical_button,
        fix_to_numeric_button=fix_to_numeric_button,
        fix_dtype_status=fix_dtype_status,

        # --------------------------
        # Filters
        # --------------------------
        cat_filter_cols=cat_col_dropdown,
        cat_val_1=cat_val_1,
        cat_val_2=cat_val_2,
        cat_val_3=cat_val_3,
        apply_filter_button=apply_filter_button,
        filter_status=filter_status,

        # --------------------------
        # Filtered data outputs
        # --------------------------
        preview_checkbox_filter=preview_checkbox_filter,
        overview_checkbox_filter=overview_checkbox_filter,
        csv_preview_filter=csv_preview_filter,
        desc_output_filter=desc_output_filter,
        dtypes_output_filter=dtypes_output_filter,
    )

