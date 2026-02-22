import gradio as gr
import pandas as pd

from controllers.estimation.inference_controller import (
    run_confidence_intervals,
    run_prediction_intervals,
    run_confidence_regions,
    get_available_estimators,
)

from controllers.utils.downloads import dataframe_to_csv, figure_to_png
from core.estimation.inference.estimator_options import ALL_MEDIAN_ESTIMATORS


def build(state):

    # ============================================================
    # Dynamic dropdown filtering
    # ============================================================
    def update_estimator_dropdowns(column):
        df = state.filtered_df if state.filtered_df is not None else state.df

        if df is None or column is None or column not in df.columns:
            return gr.update(), gr.update()

        data = df[column].dropna()
        estimators = get_available_estimators(data)

        return (
            gr.update(choices=estimators["mean_estimators"]),
            gr.update(choices=estimators["deviation_estimators"]),
        )

    gr.Markdown("## Statistical Inference")

    with gr.Row():
        refresh_button = gr.Button("🔄 Refresh Numeric Columns")

        column_dropdown = gr.Dropdown(
            label="Select Numeric Column",
            choices=[],
            interactive=True,
            elem_classes=["data-selector"],
            elem_id="custom_dropdown",
        )

        estimation_type = gr.Dropdown(
            label="Type of Estimation",
            choices=[
                "Confidence Intervals",
                "Prediction Intervals",
                "Confidence and Prediction Intervals",
                "Confidence Regions",
            ],
            value="Confidence and Prediction Intervals",
        )

        alpha_input = gr.Textbox(
            label="Confidence level (e.g. 0.95)",
            value="0.95",
        )

    # ------------------------------------------------------------------
    # Confidence Regions specific controls
    # ------------------------------------------------------------------
    with gr.Row(visible=False) as cr_params_row:
        cr_probs = gr.Textbox(
            label="Confidence levels (from lower to higher)",
            value="0.1, 0.5, 0.75, 0.89, 0.95",
        )

        cr_eps_mu = gr.Textbox(
            label="Extra margin for μ",
            value="0.1, 0.1",
        )

        cr_eps_sigma = gr.Textbox(
            label="Extra margin for σ",
            value="0.05, 0.05",
        )

        cr_add_ci_box = gr.Checkbox(
            label="Add CI for μ and σ",
            value=True,
        )

    mu_ci_source = gr.Radio(
        label="CI for μ based on",
        choices=["Mean-based CI", "Median-based CI"],
        value="Mean-based CI",
        visible=False,
    )

    # ============================================================
    # Estimators + Bootstrap (CI controls block)
    # ============================================================
    with gr.Column() as ci_controls:
        with gr.Row():
            mean_select = gr.Dropdown(
                label="Mean Estimator",
                choices=get_available_estimators([])["mean_estimators"],
                value="Sample Mean",
            )

            trim_alpha = gr.Textbox(
                label="Trimmed Mean α",
                value="0.1",
                visible=False,
            )

            winsor_limits = gr.Textbox(
                label="Winsorized Limits (e.g. 0.1, 0.1)",
                value="0.1, 0.1",
                visible=False,
            )

            weights_column = gr.Dropdown(
                label="Weights Column",
                choices=[],
                visible=False,
                elem_classes=["data-selector"],
                elem_id="custom_dropdown",
            )

            median_select = gr.Dropdown(
                label="Median Estimator",
                choices=ALL_MEDIAN_ESTIMATORS,
                value="Sample Median",
            )

            sigma_select = gr.Dropdown(
                label="Deviation Estimator",
                choices=get_available_estimators([])["deviation_estimators"],
                value="Deviation (1 ddof)",
            )

        def toggle_mean_params(mean_est):
            return (
                gr.update(visible=mean_est == "Trimmed Mean"),
                gr.update(visible=mean_est == "Winsorized Mean"),
                gr.update(visible=mean_est == "Weighted Mean"),
            )

        mean_select.change(
            toggle_mean_params,
            inputs=mean_select,
            outputs=[trim_alpha, winsor_limits, weights_column],
        )

        # ============================================================
        # Bootstrap
        # ============================================================
        with gr.Row():
            boots_mean = gr.Checkbox(label="Bootstrap Mean", value=False)
            boots_median = gr.Checkbox(label="Bootstrap Median", value=False)
            boots_sigma = gr.Checkbox(label="Bootstrap Deviation", value=False)
            boots_pi = gr.Checkbox(label="Bootstrap Prediction", value=False)

        bootstrap_samples = gr.Slider(
            100,
            5000,
            value=1000,
            step=100,
            label="Bootstrap Samples",
            visible=False,
        )

        def toggle_prediction_bootstrap_visibility(estimation_type_value):
            return gr.update(
                visible=estimation_type_value
                in (
                    "Prediction Intervals",
                    "Confidence and Prediction Intervals",
                )
            )

        estimation_type.change(
            fn=toggle_prediction_bootstrap_visibility,
            inputs=estimation_type,
            outputs=boots_pi,
        )

        def toggle_bootstrap_slider(bm, bmed, bs, bpi):
            # Show the slider if ANY bootstrap option is selected
            return gr.update(visible=(bm or bmed or bs or bpi))

        for cb in (boots_mean, boots_median, boots_sigma, boots_pi):
            cb.change(
                toggle_bootstrap_slider,
                inputs=[boots_mean, boots_median, boots_sigma, boots_pi],
                outputs=bootstrap_samples,
            )

    # ============================================================
    # Visibility helpers for CR + CI controls
    # ============================================================
    def toggle_alpha_visibility(estimation_type_value):
        # Hide the global confidence level textbox only for Confidence Regions
        return gr.update(visible=estimation_type_value != "Confidence Regions")


    def toggle_ci_controls(estimation_type_value, add_ci_value):
        """
        Show CI controls (estimators + bootstrap row) in all modes
        except when:
          - Type is 'Confidence Regions' AND
          - 'Add CI for μ and σ' is unchecked.
        """
        if estimation_type_value == "Confidence Regions":
            visible = bool(add_ci_value)
        else:
            visible = True
        return gr.update(visible=visible)

    def toggle_cr_controls(estimation_type_value, add_ci_value):
        """
        - cr_params_row is visible iff type is 'Confidence Regions'.
        - mu_ci_source is visible iff type is 'Confidence Regions'
          AND the 'Add CI for μ and σ' checkbox is checked.
        """
        is_cr = estimation_type_value == "Confidence Regions"
        cr_row_visible = is_cr
        mu_ci_visible = is_cr and bool(add_ci_value)
        return (
            gr.update(visible=cr_row_visible),
            gr.update(visible=mu_ci_visible),
        )

    # ============================================================
    # Run + outputs
    # ============================================================
    with gr.Column(elem_id="column_centered"):
        run_button = gr.Button(
            "🚀 Run Statistical Inference",
            elem_id="run_button",
        )

    with gr.Row(visible=False) as download_row:
        filename_input = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. intervals",
        )
        download_button = gr.Button("💾 Download Table as CSV")
        download_file = gr.File(
            label="Download link will appear here",
            interactive=False,
        )

    output_table = gr.Dataframe(
        visible=False,
        interactive=False,
    )

    with gr.Row(visible=False) as fig_download_row:
        fig_filename_input = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. confidence_region",
        )
        fig_download_button = gr.Button("🖼️ Download Figure as PNG")
        fig_download_file = gr.File(
            label="Download link will appear here",
            interactive=False,
        )

    output_plot = gr.Plot(
        visible=False,
    )

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------
    def refresh_columns():
        return (
            gr.update(choices=state.numeric_cols or []),
            gr.update(choices=state.numeric_cols or []),
        )

    def parse_probs(text):
        try:
            vals = [
                float(p.strip())
                for p in text.split(",")
                if p.strip()
            ]
        except ValueError:
            raise gr.Error(
                "Confidence levels must be numeric, comma-separated values."
            )

        if not vals:
            raise gr.Error("Provide at least one confidence level.")

        if any(not (0 < v < 1) for v in vals):
            raise gr.Error("All confidence levels must be in (0, 1).")

        vals = sorted(vals)
        return vals

    def parse_margin_pair(text, label):
        try:
            parts = [
                float(p.strip())
                for p in text.split(",")
                if p.strip()
            ]
        except ValueError:
            raise gr.Error(f"{label} must be numeric, comma-separated values.")

        if len(parts) != 2:
            raise gr.Error(f"{label} must have exactly two values.")

        if any(p < 0 for p in parts):
            raise gr.Error(f"{label} must be non-negative.")

        return parts[0], parts[1]

    def on_run(
        column,
        estimation_type_value,
        alpha_text,
        cr_probs_text,
        cr_eps_mu_text,
        cr_eps_sigma_text,
        cr_add_ci,
        mu_ci_source_value,
        mean_est,
        median_est,
        sigma_est,
        trim_alpha_text,
        winsor_text,
        weights_col,
        bm,
        bmed,
        bs,
        bpi,
        B,
    ):
        df = state.filtered_df if state.filtered_df is not None else state.df
        if df is None:
            raise gr.Error("No dataset loaded.")

        # ------------------------------------------------------------------
        # Parse global confidence level
        # ------------------------------------------------------------------
        try:
            conf_level = float(alpha_text)
            if not (0 < conf_level < 1):
                raise ValueError
            alpha = 1 - conf_level
        except ValueError:
            raise gr.Error("Confidence level must be in (0, 1).")

        data = df[column].dropna()
        weights = df[weights_col] if weights_col else None

        # Weighted mean validation
        if mean_est == "Weighted Mean" and weights is None:
            raise gr.Error(
                "You selected 'Weighted Mean'. Please choose a weights column "
                "in the 'Weights Column' dropdown."
            )

        # ------------------------------------------------------------------
        # Confidence Regions mode
        # ------------------------------------------------------------------
        if estimation_type_value == "Confidence Regions":
            probs = parse_probs(cr_probs_text)
            eps_mu = parse_margin_pair(cr_eps_mu_text, "Extra margin for μ")
            eps_sigma = parse_margin_pair(cr_eps_sigma_text, "Extra margin for σ")

            # Build the confidence regions figure (controller handles CI choice)
            fig = run_confidence_regions(
                data=data,
                alpha=alpha,
                mean_estimator=mean_est,
                median_estimator=median_est,
                sigma_estimator=sigma_est,
                trim_param=trim_alpha_text,
                winsor_limits=winsor_text,
                weights=weights,
                bootstrap_mean=bm,
                bootstrap_median=bmed,
                bootstrap_deviation=bs,
                bootstrap_samples=B,
                mu_ci_source=mu_ci_source_value,
                probs=probs,
                eps_mu=eps_mu,
                eps_sigma=eps_sigma,
                add_ci_box=cr_add_ci,
            )

            state.export_figure = fig

            if cr_add_ci:
                # Compute CI table for display (separate from the figure logic)
                ci_table, mean_ci, sigma_ci, median_ci = run_confidence_intervals(
                    data=data,
                    alpha=alpha,
                    mean_estimator=mean_est,
                    median_estimator=median_est,
                    sigma_estimator=sigma_est,
                    trim_param=trim_alpha_text,
                    winsor_limits=winsor_text,
                    weights=weights,
                    bootstrap_mean=bm,
                    bootstrap_median=bmed,
                    bootstrap_deviation=bs,
                    bootstrap_samples=B,
                )
                ci_table_rounded = ci_table.round(4)
                state.export_table = ci_table_rounded

                return (
                    gr.update(value=ci_table_rounded, visible=True),  # output_table
                    gr.update(visible=True),                          # download_row
                    gr.update(value=fig, visible=True),               # output_plot
                    gr.update(visible=True),                          # fig_download_row
                )
            else:
                state.export_table = None

                return (
                    gr.update(visible=False),                         # output_table
                    gr.update(visible=False),                         # download_row
                    gr.update(value=fig, visible=True),               # output_plot
                    gr.update(visible=True),                          # fig_download_row
                )

        # ------------------------------------------------------------------
        # CI / PI modes
        # ------------------------------------------------------------------
        tables = []

        if estimation_type_value in (
            "Confidence Intervals",
            "Confidence and Prediction Intervals",
        ):
            ci, _, _, _ = run_confidence_intervals(
                data=data,
                alpha=alpha,
                mean_estimator=mean_est,
                median_estimator=median_est,
                sigma_estimator=sigma_est,
                trim_param=trim_alpha_text,
                winsor_limits=winsor_text,
                weights=weights,
                bootstrap_mean=bm,
                bootstrap_median=bmed,
                bootstrap_deviation=bs,
                bootstrap_samples=B,
            )
            tables.append(ci)

        if estimation_type_value in (
            "Prediction Intervals",
            "Confidence and Prediction Intervals",
        ):
            pi = run_prediction_intervals(
                data=data,
                alpha=alpha,
                mean_estimator=mean_est,
                median_estimator=median_est,
                sigma_estimator=sigma_est,
                trim_param=trim_alpha_text,
                winsor_limits=winsor_text,
                weights=weights,
                bootstrap=bpi,
                bootstrap_samples=B,
            )
            tables.append(pi)

        final_table = pd.concat(tables, ignore_index=True).round(4)
        state.export_table = final_table
        state.export_figure = None

        return (
            gr.update(value=final_table, visible=True),  # output_table
            gr.update(visible=True),                      # download_row
            gr.update(visible=False),                     # output_plot
            gr.update(visible=False),                     # fig_download_row
        )

    def on_download(name):
        return dataframe_to_csv(state.export_table, name)

    def on_download_figure(name):
        base = (name or "confidence_regions").strip() or "confidence_regions"
        return figure_to_png(state.export_figure, base)

    # ============================================================
    # Events
    # ============================================================
    refresh_button.click(
        refresh_columns,
        outputs=[column_dropdown, weights_column],
    )

    column_dropdown.change(
        fn=update_estimator_dropdowns,
        inputs=column_dropdown,
        outputs=[mean_select, sigma_select],
    )

    # CR controls + CI controls visibility
    estimation_type.change(
        fn=toggle_cr_controls,
        inputs=[estimation_type, cr_add_ci_box],
        outputs=[cr_params_row, mu_ci_source],
    )
    estimation_type.change(
        fn=toggle_ci_controls,
        inputs=[estimation_type, cr_add_ci_box],
        outputs=ci_controls,
    )
    estimation_type.change(
        fn=toggle_alpha_visibility,
        inputs=estimation_type,
        outputs=alpha_input,
    )

    cr_add_ci_box.change(
        fn=toggle_cr_controls,
        inputs=[estimation_type, cr_add_ci_box],
        outputs=[cr_params_row, mu_ci_source],
    )
    cr_add_ci_box.change(
        fn=toggle_ci_controls,
        inputs=[estimation_type, cr_add_ci_box],
        outputs=ci_controls,
    )

    run_button.click(
        on_run,
        inputs=[
            column_dropdown,
            estimation_type,
            alpha_input,
            cr_probs,
            cr_eps_mu,
            cr_eps_sigma,
            cr_add_ci_box,
            mu_ci_source,
            mean_select,
            median_select,
            sigma_select,
            trim_alpha,
            winsor_limits,
            weights_column,
            boots_mean,
            boots_median,
            boots_sigma,
            boots_pi,
            bootstrap_samples,
        ],
        outputs=[output_table, download_row, output_plot, fig_download_row],
    )

    download_button.click(
        on_download,
        inputs=filename_input,
        outputs=download_file,
    )

    fig_download_button.click(
        on_download_figure,
        inputs=fig_filename_input,
        outputs=fig_download_file,
    )
