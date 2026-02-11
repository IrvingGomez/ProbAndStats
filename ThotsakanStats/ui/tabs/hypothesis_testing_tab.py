import pandas as pd
import gradio as gr

from controllers.hypothesis_controller import run_hypothesis_testing
from controllers.utils.downloads import dataframe_to_csv, figure_to_png


def build(state):
    gr.Markdown("## 🧪 Hypothesis Testing")

    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")

        numeric_column_dropdown = gr.Dropdown(
            label="Select Numeric Column",
            choices=[],
            interactive=True,
            elem_classes=["data_related"],
            elem_id="custom_dropdown",
        )

        hypo_test_dropdown = gr.Dropdown(
            label="Type of Hypothesis",
            choices=[
                "One sample Student's t-test",
                "Equal variance between two groups",
                "Two samples Student's t-test",
                "One-way ANOVA",
            ],
            value="One sample Student's t-test",
            interactive=True,
        )

        mu0_input = gr.Textbox(
            label="μ₀ (Null Hypothesis Mean)",
            value="",
            visible=True,
        )

        alternative_radio = gr.Radio(
            label="Alternative hypothesis",
            choices=["two-sided", "greater", "less"],
            value="two-sided",
            interactive=True,
            visible=True,
        )

        ttest_correction_check = gr.Checkbox(
            label="Correct for unequal variances (Welch's t-test)",
            value=True,
            visible=False,
        )

        equal_var_dropdown = gr.Dropdown(
            label="Select Variance Test",
            choices=["Bartlett", "Levene"],
            value="Levene",
            visible=False,
        )

    with gr.Row() as ttest_graph_option:
        ttest_graph_check = gr.Checkbox(
            label="Include graph",
            value=True,
            interactive=True,
        )
        ttest_plot_type = gr.Dropdown(
            label="Select Graph",
            choices=["Sample Histogram", "Mean Density"],
            value="Mean Density",
            visible=False,
        )
        ttest_boots_sample = gr.Slider(
            minimum=100,
            maximum=5000,
            value=1000,
            step=100,
            label="Bootstrap Samples",
        )

    # ------------------------------------------------------------
    # Categorical selection
    # ------------------------------------------------------------
    with gr.Group(visible=False) as category_group:

        refresh_categorical_button = gr.Button(
            "🔄 Refresh Categorical Columns",
            elem_id="run_button",
        )

        with gr.Row() as group1:
            cat_column_dropdown_1 = gr.Dropdown(
                label="Categorical Column 1",
                choices=[],
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )
            cat_values_dropdown_1 = gr.Dropdown(
                label="Categories for Column 1",
                multiselect=True,
                choices=[],
                interactive=True,
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )
            name_group1 = gr.Textbox(
                label="Name of Group 1",
                value="Group 1",
                visible=True,
                interactive=True,
            )

        with gr.Row() as group2:
            cat_column_dropdown_2 = gr.Dropdown(
                label="Categorical Column 2",
                choices=[],
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )
            cat_values_dropdown_2 = gr.Dropdown(
                label="Categories for Column 2",
                multiselect=True,
                choices=[],
                interactive=True,
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )
            name_group2 = gr.Textbox(
                label="Name of Group 2",
                value="Group 2",
                visible=True,
                interactive=True,
            )

        with gr.Row() as group_anova:
            cat_column_dropdown_3 = gr.Dropdown(
                label="Categorical Column",
                choices=[],
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )
            cat_values_dropdown_3 = gr.Dropdown(
                label="Categories for Column",
                multiselect=True,
                choices=[],
                interactive=True,
                elem_classes=["data_related"],
                elem_id="custom_dropdown",
            )

    with gr.Column(elem_id="column_centered"):
        run_hypo_test_button = gr.Button(
            value="🚀 Run Hypothesis Testing",
            elem_id="run_button",
        )

    # ============================================================
    # Results + Downloads
    # ============================================================

    with gr.Row(visible=False) as table_download_row:
        table_filename = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. hypothesis_results",
        )
        table_download_button = gr.Button("💾 Download Table (CSV)")
        table_file = gr.File(
            label="Download link will appear here",
            interactive=False,
        )

    output_table = gr.Dataframe(
        interactive=False,
        visible=False,
    )

    with gr.Row(visible=False) as figure_download_row:
        figure_filename = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. hypothesis_figure",
        )
        figure_download_button = gr.Button("🖼️ Download Figure (PNG)")
        figure_file = gr.File(
            label="Download link will appear here",
            interactive=False,
        )

    output_plot = gr.Plot(
        visible=False,
    )

    # ============================================================
    # Helpers
    # ============================================================
    def refresh_numeric_columns():
        numeric_cols = state.numeric_cols or []
        return gr.update(choices=numeric_cols)

    def refresh_categorical_columns():
        cat_cols = state.categorical_cols or []
        if not cat_cols:
            return [gr.update(choices=[], value=None)] * 3 + [
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
            ]

        return [
            gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_1
            gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_2
            gr.update(choices=cat_cols, value=None),  # cat_column_dropdown_3
            gr.update(choices=[], value=[]),  # cat_values_dropdown_1
            gr.update(choices=[], value=[]),  # cat_values_dropdown_2
            gr.update(choices=[], value=[]),  # cat_values_dropdown_3
        ]

    def update_category_options(col: str | None):
        df = state.filtered_df if state.filtered_df is not None else state.df
        if df is None or not col or col not in df.columns:
            return gr.update(choices=[], value=[])
        values = sorted(df[col].dropna().unique())
        values_str = [str(v) for v in values]
        return gr.update(choices=values_str, value=[])

    def update_group_name(cat_vals: list[str], default_label: str):
        if cat_vals:
            return gr.update(value=cat_vals[0])
        return gr.update(value=default_label)

    def toggle_hypo_test(sel: str):
        if sel == "One sample Student's t-test":
            return [
                gr.update(visible=True),   # mu0_input
                gr.update(visible=True),   # alternative_radio
                gr.update(visible=True),   # ttest_graph_option
                gr.update(visible=False),  # ttest_correction_check
                gr.update(visible=False),  # equal_var_dropdown
                gr.update(visible=False),  # category_group
                gr.update(visible=False),  # group1
                gr.update(visible=False),  # group2
                gr.update(visible=False),  # group_anova
            ]
        elif sel == "Equal variance between two groups":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            ]
        elif sel == "Two samples Student's t-test":
            return [
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            ]
        elif sel == "One-way ANOVA":
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            ]
        else:
            # Fallback: hide everything
            return [
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            ]

    def toggle_ttest_plot_type(include_graph: bool, sel: str):
        if include_graph and sel == "Two samples Student's t-test":
            return gr.update(visible=True)
        return gr.update(visible=False)

    def on_run(
        numeric_col: str | None,
        hypo_test: str,
        mu0_text: str,
        alternative: str,
        include_graph: bool,
        bootstrap_samples: int,
        cat_col1: str | None,
        cat_vals1: list[str],
        name_g1: str,
        cat_col2: str | None,
        cat_vals2: list[str],
        name_g2: str,
        cat_col3: str | None,
        cat_vals3: list[str],
        plot_type: str,
        correction_flag: bool,
        variance_test_type: str,
    ):
        df = state.filtered_df if state.filtered_df is not None else state.df

        def _error_result(message: str):
            err = pd.DataFrame([[message]], columns=["Error"])
            state.export_table = err
            state.export_figure = None
            # (table, figure, table_download_row, figure_download_row)
            return (
                gr.update(value=err, visible=True),
                gr.update(value=None, visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
            )

        if df is None:
            return _error_result("No dataset loaded.")

        if not numeric_col:
            return _error_result("No numeric column selected.")

        try:
            table, fig = run_hypothesis_testing(
                df=df,
                numeric_col=numeric_col,
                hypo_test=hypo_test,
                mu0_text=mu0_text,
                alternative=alternative,
                include_graph=include_graph,
                bootstrap_samples=int(bootstrap_samples),
                cat_col1=cat_col1,
                cat_vals1=cat_vals1 or [],
                name_group1=name_g1,
                cat_col2=cat_col2,
                cat_vals2=cat_vals2 or [],
                name_group2=name_g2,
                cat_col3=cat_col3,
                cat_vals3=cat_vals3 or [],
                plot_type=plot_type,
                correction=bool(correction_flag),
                test_type=variance_test_type,
            )
        except Exception as e:
            return _error_result(f"❌ Error: {e}")

        state.export_table = table
        state.export_figure = fig

        return (
            gr.update(value=table, visible=True),
            gr.update(value=fig, visible=fig is not None),
            gr.update(visible=True),
            gr.update(visible=fig is not None),
        )

    def on_download_table(filename: str | None):
        return dataframe_to_csv(state.export_table, filename or "hypothesis_test")

    def on_download_figure(filename: str | None):
        return figure_to_png(state.export_figure, filename or "hypothesis_test_plot")

    # ============================================================
    # Events
    # ============================================================
    refresh_columns_button.click(
        fn=refresh_numeric_columns,
        inputs=[],
        outputs=numeric_column_dropdown,
    )

    hypo_test_dropdown.change(
        fn=toggle_hypo_test,
        inputs=[hypo_test_dropdown],
        outputs=[
            mu0_input,
            alternative_radio,
            ttest_graph_option,
            ttest_correction_check,
            equal_var_dropdown,
            category_group,
            group1,
            group2,
            group_anova,
        ],
    )

    hypo_test_dropdown.change(
        fn=toggle_ttest_plot_type,
        inputs=[ttest_graph_check, hypo_test_dropdown],
        outputs=[ttest_plot_type],
    )

    ttest_graph_check.change(
        fn=lambda check: gr.update(visible=check),
        inputs=[ttest_graph_check],
        outputs=[ttest_boots_sample],
    )

    ttest_graph_check.change(
        fn=toggle_ttest_plot_type,
        inputs=[ttest_graph_check, hypo_test_dropdown],
        outputs=[ttest_plot_type],
    )

    refresh_categorical_button.click(
        fn=refresh_categorical_columns,
        inputs=[],
        outputs=[
            cat_column_dropdown_1,
            cat_column_dropdown_2,
            cat_column_dropdown_3,
            cat_values_dropdown_1,
            cat_values_dropdown_2,
            cat_values_dropdown_3,
        ],
    )

    cat_column_dropdown_1.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_1],
        outputs=[cat_values_dropdown_1],
    )

    cat_column_dropdown_2.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_2],
        outputs=[cat_values_dropdown_2],
    )

    cat_column_dropdown_3.change(
        fn=update_category_options,
        inputs=[cat_column_dropdown_3],
        outputs=[cat_values_dropdown_3],
    )

    cat_values_dropdown_1.change(
        fn=update_group_name,
        inputs=[cat_values_dropdown_1, name_group1],
        outputs=name_group1,
    )

    cat_values_dropdown_2.change(
        fn=update_group_name,
        inputs=[cat_values_dropdown_2, name_group2],
        outputs=name_group2,
    )

    run_hypo_test_button.click(
        fn=on_run,
        inputs=[
            numeric_column_dropdown,
            hypo_test_dropdown,
            mu0_input,
            alternative_radio,
            ttest_graph_check,
            ttest_boots_sample,
            cat_column_dropdown_1,
            cat_values_dropdown_1,
            name_group1,
            cat_column_dropdown_2,
            cat_values_dropdown_2,
            name_group2,
            cat_column_dropdown_3,
            cat_values_dropdown_3,
            ttest_plot_type,
            ttest_correction_check,
            equal_var_dropdown,
        ],
        outputs=[
            output_table,
            output_plot,
            table_download_row,
            figure_download_row,
        ],
    )

    table_download_button.click(
        fn=on_download_table,
        inputs=table_filename,
        outputs=table_file,
    )

    figure_download_button.click(
        fn=on_download_figure,
        inputs=figure_filename,
        outputs=figure_file,
    )
