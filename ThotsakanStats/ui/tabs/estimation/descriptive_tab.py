# ui/tabs/estimation/descriptive_tab.py

import gradio as gr
from controllers.estimation.descriptive_controller import run_descriptive_statistics
from controllers.utils.downloads import dataframe_to_csv


def build(state):
    gr.Markdown("## 🧮 Descriptive Statistics")

    # ---------------------------
    # Controls
    # ---------------------------
    with gr.Row():
        refresh_button = gr.Button("🔄 Refresh Numeric Columns")

        column_dropdown = gr.Dropdown(
            label="Select Numeric Variable",
            choices=[],
            interactive=True,
            elem_classes=["data-selector"],
            elem_id="custom_dropdown",
        )

        weights_dropdown = gr.Dropdown(
            label="Weights Column (optional)",
            choices=["— None —"],
            value="— None —",
            interactive=True,
            elem_classes=["data-selector"],
            elem_id="custom_dropdown",
        )

    with gr.Row():
        quantiles_input = gr.Textbox(
            label="Quantiles",
            value="0.25, 0.5, 0.75",
        )

        trim_input = gr.Textbox(
            label="Trimmed Mean α",
            value="0.1",
        )

        winsor_input = gr.Textbox(
            label="Winsorized Limits",
            value="0.1, 0.1",
        )

    with gr.Column(elem_id="column_centered"):
        run_button = gr.Button("🚀 Run Descriptive Statistics", elem_id="run_button")

    with gr.Row(visible=False) as download_row:
        filename_input = gr.Textbox(
            label="Filename (without extension)",
            placeholder="e.g. descriptive_stats",
        )
        download_button = gr.Button("💾 Download Table as CSV")
        download_file = gr.File(
            label="Download link will appear here",
            interactive=False,
        )

    output_table = gr.Dataframe(
        label="Descriptive Statistics",
        interactive=False,
    )

    # ---------------------------
    # Helpers
    # ---------------------------
    def refresh_columns():
        numeric_cols = state.numeric_cols or []
        return (
            gr.update(choices=numeric_cols),
            gr.update(choices=["— None —"] + numeric_cols, value="— None —"),
        )

    def parse_quantiles(text):
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def parse_trim(text):
        return None if not text.strip() else float(text)

    def parse_winsor(text):
        if not text.strip():
            return None
        vals = [float(x.strip()) for x in text.split(",")]
        if len(vals) != 2:
            raise ValueError("Winsor limits must be two values.")
        return tuple(vals)

    def on_run(column, weights_col, q_text, trim_text, winsor_text):
        df = state.filtered_df if state.filtered_df is not None else state.df

        if df is None:
            return [], gr.update(visible=False), gr.update(visible=False)

        if weights_col == "— None —":
            weights_col = None

        stats_df = run_descriptive_statistics(
            df=df,
            column=column,
            quantile_probs=parse_quantiles(q_text),
            trim_alpha=parse_trim(trim_text),
            winsor_limits=parse_winsor(winsor_text),
            weights_col=weights_col,
            round_digits=4,
        )

        # cache for download
        state.export_table = stats_df

        return (
            stats_df,
            gr.update(visible=True),   # show download row
            gr.update(visible=True),   # show file output
        )

    def on_download(filename):
        return dataframe_to_csv(state.export_table, filename)

    # ---------------------------
    # Events
    # ---------------------------
    download_button.click(
        fn=on_download,
        inputs=filename_input,
        outputs=download_file,
    )

    refresh_button.click(
        fn=refresh_columns,
        inputs=[],
        outputs=[column_dropdown, weights_dropdown],
    )

    run_button.click(
        fn=on_run,
        inputs=[
            column_dropdown,
            weights_dropdown,
            quantiles_input,
            trim_input,
            winsor_input,
        ],
        outputs=[
            output_table,
            download_row,
            download_file,
        ],
    )
