import gradio as gr

from state.app_state import AppState
from ui.styles import DATA_SELECTOR_CSS
from ui.assets import LOGOS

from ui.tabs.home_tab import build_home_tab
from ui.tabs.data_tab import build as build_data_tab
from ui.tabs.estimation.descriptive_tab import build as build_descriptive_tab
from ui.tabs.estimation.inference_tab import build as build_inference_tab
from ui.tabs.estimation.graphical_tab import build as build_graphical_tab
from ui.tabs.hypothesis_testing_tab import build as build_hypothesis_tab
from ui.tabs.linear_regression_tab import build as build_linear_regression_tab


def build_layout():
    """
    Global application layout.

    Responsibilities:
    - Instantiate AppState once
    - Apply theme and CSS globally
    - Render persistent header (logos + title)
    - Define main navigation tabs
    """

    state = AppState()

    with gr.Blocks(
        title="Thotsakan Statistics",
        theme=gr.themes.Soft(),
        css=DATA_SELECTOR_CSS,
    ) as demo:

        # ==================================================
        # Global header (always visible)
        # ==================================================
        with gr.Row(equal_height=True):
            gr.Image(LOGOS["himmapan"], height=80, show_label=False, show_download_button=False, show_fullscreen_button=False)
            gr.Image(LOGOS["thotsakan"], height=80, show_label=False, show_download_button=False, show_fullscreen_button=False)
            gr.Image(LOGOS["cmkl"], height=80, show_label=False, show_download_button=False, show_fullscreen_button=False)
            gr.Image(LOGOS["aice"], height=80, show_label=False, show_download_button=False, show_fullscreen_button=False)

        gr.Markdown(
            """
            # Thotsakan Statistics  
            *Probability and Statistics Interactive Laboratory*
            """
        )

        # ==================================================
        # Main application tabs
        # ==================================================
        with gr.Tabs():

            # -------------------------
            # Home
            # -------------------------
            with gr.Tab("🏠 Home"):
                build_home_tab(
                    logos=LOGOS,
                    repo_url="https://github.com/IrvingGomez/ThotsakanStatistics",
                    maiyarap_repo_url=None,  # or set when available
                )

            # -------------------------
            # Data
            # -------------------------
            with gr.Tab("📁 Data"):
                build_data_tab(state)

            # -------------------------
            # Probability
            # -------------------------
            with gr.Tab("🎲 Probability"):
                with gr.Tabs():
                    with gr.Tab("📜 Common Distributions"):
                        gr.Markdown("🚧 Building.")

                    with gr.Tab("✍️ Custom Distribution"):
                        gr.Markdown("🚧 Building.")

                    with gr.Tab("🤏 Approximations"):
                        gr.Markdown("🚧 Building.")

            # -------------------------
            # Estimation
            # -------------------------
            with gr.Tab("📐 Estimation"):
                with gr.Tabs():
                    with gr.Tab("🧮 Descriptive Statistics"):
                        build_descriptive_tab(state)

                    with gr.Tab("💭 Statistical Inference"):
                        build_inference_tab(state)

                    with gr.Tab("📊 Graphical Analysis"):
                        build_graphical_tab(state)
                        
            # -------------------------
            # Hypothesis Testing
            # -------------------------
            with gr.Tab("🧪 Hypothesis Testing"):
                build_hypothesis_tab(state)

            # -------------------------
            # Linear Regression
            # -------------------------
            with gr.Tab("📈 Linear Regression"):
                build_linear_regression_tab(state)

    return demo
