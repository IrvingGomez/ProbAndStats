from __future__ import annotations

from typing import List

import gradio as gr

from state.app_state import AppState
from controllers.linear_regression_controller import run_linear_regression


def build(state: AppState) -> None:
    """
    Build the Linear Regression tab.

    This recreates the behaviour and layout of the monolithic implementation,
    while delegating statistics to the `stats.linear_regression` module and
    wiring / validation to `regression_controller`.
    """
    gr.Markdown("## Linear Regression")

    # Use the numeric columns discovered by the Data tab
    numeric_cols = state.numeric_cols or []

    # ------------------------------------------------------------------
    # Top row: column selection
    # ------------------------------------------------------------------
    with gr.Row(elem_id="row_centered"):
        refresh_columns_button = gr.Button("🔄 Refresh Numeric Columns")
        dependent_dropdown = gr.Dropdown(
            label="Dependent Variable",
            choices=numeric_cols,
            interactive=True,
            elem_classes="data_related",
            elem_id="custom_dropdown",
        )
        independent_dropdown = gr.Dropdown(
            label="Independent Variable(s)",
            multiselect=True,
            choices=numeric_cols,
            interactive=True,
            elem_classes="data_related",
            elem_id="custom_dropdown",
        )

    # ------------------------------------------------------------------
    # Formula controls
    # ------------------------------------------------------------------
    with gr.Row():
        formula_check = gr.Checkbox(
            label="Would you like to write down the regression formula?",
            value=False,
            interactive=True,
        )
        formula_text = gr.Textbox(
            label="Write the formula",
            placeholder="Y ~ X + np.sin(X) + I((X-5)**2)",
            interactive=True,
            visible=False,
        )
        formula_latex = gr.Textbox(
            label="Write the formula in LaTeX (Optional)",
            placeholder=r"Y = X + \sin(X) + (X-5)^2",
            interactive=True,
            visible=False,
        )

    # ------------------------------------------------------------------
    # Global options
    # ------------------------------------------------------------------
    with gr.Row():
        alpha_input = gr.Textbox(
            label="Confidence level (e.g. 0.95)",
            value=0.95,
            interactive=True,
        )
        intercept_check = gr.Checkbox(
            label="Include intercept",
            value=True,
            interactive=True,
        )
        graph_check_reg = gr.Checkbox(
            label="Create graph",
            value=True,
            interactive=True,
        )

    # ------------------------------------------------------------------
    # Graph options
    # ------------------------------------------------------------------
    with gr.Row() as graph_options:
        graph_dropdown = gr.Dropdown(
            label="Graph",
            choices=["Simple Regression", "Observed vs Predicted"],
            value="Simple Regression",
            interactive=True,
        )
        show_ci_check = gr.Checkbox(
            label="Include CI",
            value=True,
            interactive=True,
        )
        show_pi_check = gr.Checkbox(
            label="Include PI",
            value=True,
            interactive=True,
        )
        fit_to_obs_check = gr.Checkbox(
            label="Fit to observations",
            value=True,
            interactive=True,
        )
        x_vect_input = gr.Textbox(
            label="Minimum and maximum of dependent variable ",
            value="",
            visible=False,
            interactive=True,
        )

    # ------------------------------------------------------------------
    # Run button
    # ------------------------------------------------------------------
    with gr.Column(elem_id="column_centered"):
        run_regression_button = gr.Button(
            value="🚀 Run Linear Regression",
            elem_id="run_button",
        )

    # ------------------------------------------------------------------
    # Results blocks
    # ------------------------------------------------------------------
    with gr.Row(visible=False) as output_table_row:
        output_table = gr.HTML(label="Regression Summary")

    with gr.Row(visible=False) as output_plot_row:
        output_plot = gr.Plot(label="Regression Plot")

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------
    def on_toggle_formula(check: bool):
        # When the user chooses to write a formula, we show the formula
        # text inputs and hide the intercept checkbox (the formula
        # decides the intercept).
        return (
            gr.update(visible=check),          # formula_text
            gr.update(visible=check),          # formula_latex
            gr.update(visible=not check, value=not check),  # intercept_check
        )

    def on_update_graph_choices(independent_vars: List[str]):
        # With a single predictor we can show both graphs; otherwise only
        # the "Observed vs Predicted" graph is available.
        if len(independent_vars) == 1:
            return gr.update(
                choices=["Simple Regression", "Observed vs Predicted"],
                value="Simple Regression",
            )
        return gr.update(
            choices=["Observed vs Predicted"],
            value="Observed vs Predicted",
        )

    def on_toggle_graph_options(graph_type: str, fit_to_obs: bool):
        # For the Simple Regression graph we allow CI/PI and fit_to_obs;
        # for Observed vs Predicted no interval / range options are used.
        if graph_type == "Simple Regression":
            return (
                gr.update(visible=True),                    # show_ci_check
                gr.update(visible=True),                    # show_pi_check
                gr.update(visible=True, value=fit_to_obs),  # fit_to_obs_check
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=True),
            )

    def on_toggle_graph_block(check: bool):
        return gr.update(visible=check)

    def on_toggle_range_input(fit_to_obs: bool):
        # Range input only makes sense when not fitting strictly to observed X.
        return gr.update(visible=not fit_to_obs)

    def on_refresh_columns():
        cols = state.numeric_cols or []
        return (
            gr.update(choices=cols),
            gr.update(choices=cols),
        )

    # ------------------------------------------------------------------
    # Run callback
    # ------------------------------------------------------------------
    def on_run(
        formula_check_val,
        formula_text_val,
        formula_latex_val,
        dep_var_val,
        indep_vars_val,
        alpha_input_val,
        intercept_val,
        graph_check_val,
        graph_type_val,
        show_ci_val,
        show_pi_val,
        fit_to_obs_val,
        x_range_text_val,
    ):
        # Ensure we have a dataset
        if state.df is None:
            return (
                gr.update(visible=True),
                gr.update(visible=True, value="<b>Error:</b> No dataset loaded."),
                gr.update(visible=False),
                gr.update(visible=False, value=None),
            )

        try:
            summary_html, params_df, fig = run_linear_regression(
                df=state.df,
                filtered_df=state.filtered_df,
                formula_check=bool(formula_check_val),
                formula_text=str(formula_text_val or ""),
                formula_latex=str(formula_latex_val or ""),
                dependent_var=dep_var_val,
                independent_vars=list(indep_vars_val or []),
                alpha_input=str(alpha_input_val),
                intercept=bool(intercept_val),
                graph_check=bool(graph_check_val),
                graph_type=str(graph_type_val),
                show_ci=bool(show_ci_val),
                show_pi=bool(show_pi_val),
                fit_to_obs=bool(fit_to_obs_val),
                x_range_text=str(x_range_text_val or ""),
                round_digits=4,
            )
        except Exception as e:  # noqa: BLE001
            # Surface a friendly HTML error message
            error_html = f"<b>Regression failed:</b> {e}"
            return (
                gr.update(visible=True),
                gr.update(visible=True, value=error_html),
                gr.update(visible=False),
                gr.update(visible=False, value=None),
            )

        # Store the coefficient table for potential downloads elsewhere
        state.export_table = params_df

        return (
            gr.update(visible=True),
            gr.update(visible=True, value=summary_html),
            gr.update(visible=fig is not None),
            gr.update(visible=fig is not None, value=fig),
        )

    # ------------------------------------------------------------------
    # Wire up callbacks
    # ------------------------------------------------------------------
    formula_check.change(
        on_toggle_formula,
        inputs=formula_check,
        outputs=[formula_text, formula_latex, intercept_check],
    )

    refresh_columns_button.click(
        fn=on_refresh_columns,
        inputs=[],
        outputs=[dependent_dropdown, independent_dropdown],
    )

    independent_dropdown.change(
        on_update_graph_choices,
        inputs=independent_dropdown,
        outputs=graph_dropdown,
    )

    graph_check_reg.change(
        on_toggle_graph_block,
        inputs=graph_check_reg,
        outputs=graph_options,
    )

    graph_dropdown.change(
        on_toggle_graph_options,
        inputs=[graph_dropdown, fit_to_obs_check],
        outputs=[show_ci_check, show_pi_check, fit_to_obs_check],
    )

    fit_to_obs_check.change(
        on_toggle_range_input,
        inputs=fit_to_obs_check,
        outputs=x_vect_input,
    )

    run_regression_button.click(
        on_run,
        inputs=[
            formula_check,
            formula_text,
            formula_latex,
            dependent_dropdown,
            independent_dropdown,
            alpha_input,
            intercept_check,
            graph_check_reg,
            graph_dropdown,
            show_ci_check,
            show_pi_check,
            fit_to_obs_check,
            x_vect_input,
        ],
        outputs=[output_table_row, output_table, output_plot_row, output_plot],
    )
