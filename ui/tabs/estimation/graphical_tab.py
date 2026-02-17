import gradio as gr

from controllers.estimation.graphical_controller import run_graphical_analysis
from controllers.utils.downloads import figure_to_png


def build(state):

    ALL_MEAN_ESTIMATORS = [
        "Sample Mean",
        "Geometric Mean",
        "Harmonic Mean",
        "Interquartile Mean",
        "Trimmed Mean",
        "Winsorized Mean",
        "Weighted Mean",
    ]

    ALL_DEVIATION_ESTIMATORS = [
        "Deviation (1 ddof)",
        "Range (bias corrected)",
        "IQR (bias corrected)",
        "MAD (bias corrected)",
        "AAD (bias corrected)",
    ]

    # ============================================================
    # Dynamic dropdown filtering (depends on selected column)
    # ============================================================
    def update_estimator_dropdowns(column):
        df = state.filtered_df if state.filtered_df is not None else state.df

        if df is None or column is None or column not in df.columns:
            return gr.update(), gr.update()

        data = df[column].dropna()

        mean_choices = ALL_MEAN_ESTIMATORS.copy()
        if (data <= 0).any():
            mean_choices = [
                m
                for m in mean_choices
                if m not in ("Geometric Mean", "Harmonic Mean")
            ]

        deviation_choices = ALL_DEVIATION_ESTIMATORS.copy()
        if len(data) > 25:
            deviation_choices = [
                d
                for d in deviation_choices
                if d != "Range (bias corrected)"
            ]

        return (
            gr.update(choices=mean_choices),
            gr.update(choices=deviation_choices),
        )

    gr.Markdown("## Graphical Analysis")

    # -----------------------------------------------------------
    # Top controls: column + graph type
    # -----------------------------------------------------------
    with gr.Row():
        refresh_button = gr.Button("🔄 Refresh Numeric Columns")

        column_dropdown = gr.Dropdown(
            label="Select Numeric Column",
            choices=[],
            interactive=True,
            elem_classes=["data-selector"],
            elem_id="custom_dropdown",
        )

        graph_type_dropdown = gr.Dropdown(
            label="Select Graph",
            choices=[
                "Histogram",
                "Empirical Probability Mass Function",
                "Empirical Cumulative Distribution Function (ECDF)",
            ],
            value="Histogram",
            interactive=True,
        )

    # -----------------------------------------------------------
    # Histogram / PMF options
    # -----------------------------------------------------------
    with gr.Row() as histo_main_row:
        histo_add_kde = gr.Checkbox(
            label="Add KDE", value=True, interactive=True
        )
        histo_add_data = gr.Checkbox(
            label="Show data", value=False, interactive=True
        )
        histo_add_normal = gr.Checkbox(
            label="Add Normal Density", value=False, interactive=True
        )
        histo_add_ci = gr.Checkbox(
            label="Add Confidence Interval", value=False, interactive=True
        )
        histo_add_pi = gr.Checkbox(
            label="Add Prediction Interval", value=False, interactive=True
        )

    # Interval type choices (CI / PI)
    with gr.Row(visible=False) as interval_choice_row:
        histo_choose_ci = gr.Radio(
            label="Confidence Interval",
            choices=["Mean", "Median", "Both"],
            value="Both",
            interactive=True,
        )
        histo_choose_pi = gr.Radio(
            label="Prediction Interval",
            choices=["Mean", "Median", "IQR", "Bootstrap"],
            value="Mean",
            interactive=True,
        )

        ci_pi_conf_level = gr.Textbox(
            label="Confidence level (e.g. 0.95)",
            value="0.95",
            interactive=True,
        )

    # -----------------------------------------------------------
    # ECDF-specific options
    # -----------------------------------------------------------
    with gr.Row(visible=False) as ecdf_row:
        ecdf_add_conf = gr.Checkbox(
            label="Add CI for the ECDF",
            value=True,
            interactive=True,
        )
        ecdf_conf_level = gr.Textbox(
            label="Confidence level (e.g. 0.95)",
            value="0.95",
            interactive=True,
            visible=True,
        )
        ecdf_add_normal = gr.Checkbox(
            label="Add Normal CDF", value=False, interactive=True
        )

    # -----------------------------------------------------------
    # Estimators + bootstrap
    # -----------------------------------------------------------
    with gr.Row(visible=False) as estimator_row:
        mean_select = gr.Dropdown(
            label="Mean Estimator",
            choices=ALL_MEAN_ESTIMATORS,
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
            choices=["Sample Median"],
            value="Sample Median",
        )

        sigma_select = gr.Dropdown(
            label="Deviation Estimator",
            choices=ALL_DEVIATION_ESTIMATORS,
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

    with gr.Row(visible=False) as bootstrap_row:
        boots_mean = gr.Checkbox(label="Bootstrap Mean", value=False)
        boots_median = gr.Checkbox(label="Bootstrap Median", value=False)
        boots_sigma = gr.Checkbox(label="Bootstrap Deviation", value=False)
        boots_pi = gr.Checkbox(label="Bootstrap Prediction", value=False)

    with gr.Row(visible=False) as bootstrap_samples_row:
        bootstrap_samples = gr.Slider(
            label="Bootstrap samples",
            minimum=100,
            maximum=5000,
            step=100,
            value=1000,
        )

    with gr.Row(visible=False) as normal_mu_row:
        normal_mu_source = gr.Radio(
            label="Normal μ based on",
            choices=["Mean-based CI", "Median-based CI"],
            value="Mean-based CI",
            interactive=True,
        )

    # -----------------------------------------------------------
    # Run button and outputs
    # -----------------------------------------------------------
    with gr.Column(elem_id="column_centered"):
        run_button = gr.Button(
            "🚀 Run Graphical Analysis",
            elem_id="run_button",
        )

    with gr.Row(visible=False) as download_row:
        filename_input = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. histogram",
        )
        download_button = gr.Button("🖼️ Download Figure as PNG")
        download_file = gr.File(label="Download link will appear here")

    output_plot = gr.Plot(visible=False)


    # -----------------------------------------------------------
    # UI logic helpers
    # -----------------------------------------------------------
    def refresh_columns():
        numeric_cols = state.numeric_cols or []
        return (
            gr.update(choices=numeric_cols),  # column_dropdown
            gr.update(choices=numeric_cols),  # weights_column
        )

    def toggle_graph_type(graph_type):
        is_hist = graph_type in [
            "Histogram",
            "Empirical Probability Mass Function",
        ]
        return (
            gr.update(visible=is_hist),      # histo_main_row
            gr.update(visible=not is_hist),  # ecdf_row
        )

    def update_estimator_block(
        graph_type, add_normal, add_ci, add_pi, ecdf_add_normal_val
    ):
        is_hist = graph_type in [
            "Histogram",
            "Empirical Probability Mass Function",
        ]
        is_ecdf = (
            graph_type == "Empirical Cumulative Distribution Function (ECDF)"
        )

        # When Histogram / PMF
        if is_hist:
            any_flag = add_normal or add_ci or add_pi
            intervals_flag = add_ci or add_pi
            estimator_visible = any_flag
            bootstrap_visible = any_flag
            interval_row_visible = intervals_flag
            choose_ci_visible = add_ci
            choose_pi_visible = add_pi
            normal_mu_visible = add_normal
            ci_pi_visible = intervals_flag
            boots_pi_visible = add_pi
        # When ECDF
        elif is_ecdf:
            any_flag = ecdf_add_normal_val
            estimator_visible = any_flag
            bootstrap_visible = any_flag
            interval_row_visible = False
            choose_ci_visible = False
            choose_pi_visible = False
            normal_mu_visible = ecdf_add_normal_val
            ci_pi_visible = False
            boots_pi_visible = False
        else:
            # Fallback – hide everything
            estimator_visible = False
            bootstrap_visible = False
            interval_row_visible = False
            choose_ci_visible = False
            choose_pi_visible = False
            normal_mu_visible = False
            ci_pi_visible = False
            boots_pi_visible = False

        return (
            gr.update(visible=estimator_visible),   # estimator_row
            gr.update(visible=bootstrap_visible),   # bootstrap_row
            gr.update(visible=interval_row_visible),  # interval_choice_row
            gr.update(visible=choose_ci_visible),   # histo_choose_ci
            gr.update(visible=choose_pi_visible),   # histo_choose_pi
            gr.update(visible=normal_mu_visible),   # normal_mu_row
            gr.update(visible=boots_pi_visible),    # boots_pi
        )

    def toggle_bootstrap_samples(
        boots_mean_val,
        boots_median_val,
        boots_sigma_val,
        boots_pi_val,
    ):
        show = any(
            [boots_mean_val, boots_median_val, boots_sigma_val, boots_pi_val]
        )
        return gr.update(visible=show)

    # -----------------------------------------------------------
    # Callbacks wiring
    # -----------------------------------------------------------
    refresh_button.click(
        refresh_columns,
        outputs=[column_dropdown, weights_column],
    )

    column_dropdown.change(
        fn=update_estimator_dropdowns,
        inputs=column_dropdown,
        outputs=[mean_select, sigma_select],
    )

    graph_type_dropdown.change(
        fn=toggle_graph_type,
        inputs=graph_type_dropdown,
        outputs=[histo_main_row, ecdf_row],
    )

    ecdf_add_conf.change(
        fn=lambda check: gr.update(visible=check),
        inputs=ecdf_add_conf,
        outputs=ecdf_conf_level,
    )

    # Any change in these controls updates estimator block visibility
    for comp in (
        graph_type_dropdown,
        histo_add_normal,
        histo_add_ci,
        histo_add_pi,
        ecdf_add_normal,
    ):
        comp.change(
            fn=update_estimator_block,
            inputs=[
                graph_type_dropdown,
                histo_add_normal,
                histo_add_ci,
                histo_add_pi,
                ecdf_add_normal,
            ],
            outputs=[
                estimator_row,
                bootstrap_row,
                interval_choice_row,
                histo_choose_ci,
                histo_choose_pi,
                normal_mu_row,
                boots_pi,
            ],
        )

    # Bootstrap sample slider visibility
    for comp in (boots_mean, boots_median, boots_sigma, boots_pi):
        comp.change(
            fn=toggle_bootstrap_samples,
            inputs=[boots_mean, boots_median, boots_sigma, boots_pi],
            outputs=bootstrap_samples_row,
        )

    # -----------------------------------------------------------
    # Run + download logic
    # -----------------------------------------------------------
    def on_run(
        column,
        graph_type,
        histo_kde,
        histo_data,
        histo_ci,
        histo_ci_choice,
        histo_pi,
        histo_pi_choice,
        ci_pi_conf_level_text,
        histo_normal,
        mean_est,
        trim_alpha_text,
        winsor_text,
        weights_col,
        median_est,
        sigma_est,
        normal_mu_src,
        boots_mean_val,
        boots_median_val,
        boots_sigma_val,
        boots_pi_val,
        boot_samples_val,
        ecdf_add_conf_val,
        ecdf_conf_level_text,
        ecdf_add_normal_val,
    ):
        df = state.filtered_df if state.filtered_df is not None else state.df
        if df is None:
            raise gr.Error(
                "No data loaded. Please load data in the Data tab first."
            )

        if column is None:
            raise gr.Error("Please select a numeric column.")

        # Parse CI/PI confidence level
        try:
            ci_pi_level = float(ci_pi_conf_level_text)
        except Exception:
            raise gr.Error(
                "Confidence level for CI/PI must be numeric, e.g. 0.95."
            )

        # Parse ECDF confidence level
        try:
            ecdf_level = float(ecdf_conf_level_text)
        except Exception:
            raise gr.Error(
                "ECDF confidence level must be numeric, e.g. 0.95."
            )

        try:
            fig = run_graphical_analysis(
                df=df,
                column=column,
                graph_type=graph_type,
                add_kde=histo_kde,
                add_data=histo_data,
                add_normal=histo_normal,
                add_ci=histo_ci,
                ci_choice=histo_ci_choice,
                add_pi=histo_pi,
                pi_choice=histo_pi_choice,
                mean_estimator=mean_est,
                median_estimator=median_est,
                sigma_estimator=sigma_est,
                trim_param=trim_alpha_text,
                winsor_limits=winsor_text,
                weights_col=weights_col,
                normal_mu_source=normal_mu_src,
                bootstrap_mean=boots_mean_val,
                bootstrap_median=boots_median_val,
                bootstrap_sigma=boots_sigma_val,
                bootstrap_prediction=boots_pi_val,
                bootstrap_samples=int(boot_samples_val),
                ci_pi_conf_level=ci_pi_level,
                ecdf_add_conf=ecdf_add_conf_val,
                ecdf_conf_level=ecdf_level,
                ecdf_add_normal=ecdf_add_normal_val,
            )
        except ValueError as e:
            raise gr.Error(str(e))

        # Make the plot component visible and set the figure.
        return (
            gr.update(value=fig, visible=True),  # output_plot
            gr.update(visible=True),             # download_row
            None,                                # download_file
        )


    run_button.click(
        fn=on_run,
        inputs=[
            column_dropdown,
            graph_type_dropdown,
            histo_add_kde,
            histo_add_data,
            histo_add_ci,
            histo_choose_ci,
            histo_add_pi,
            histo_choose_pi,
            ci_pi_conf_level,
            histo_add_normal,
            mean_select,
            trim_alpha,
            winsor_limits,
            weights_column,
            median_select,
            sigma_select,
            normal_mu_source,
            boots_mean,
            boots_median,
            boots_sigma,
            boots_pi,
            bootstrap_samples,
            ecdf_add_conf,
            ecdf_conf_level,
            ecdf_add_normal,
        ],
        outputs=[output_plot, download_row, download_file],
    )

    def on_download(fig, filename):
        if fig is None:
            return None
        name = (filename or "graphical_analysis").strip()
        base = name or "graphical_analysis"
        return figure_to_png(fig, base)

    download_button.click(
        fn=on_download,
        inputs=[output_plot, filename_input],
        outputs=download_file,
    )
