# ui/tabs/home_tab.py
from __future__ import annotations

import gradio as gr


def build_home_tab(
    *,
    logos: dict[str, str],
    repo_url: str | None = None,
    maiyarap_repo_url: str | None = None,
) -> None:
    """
    Build the Home tab (welcome/landing page) UI.

    This tab is intentionally non-interactive: no models, no callbacks.
    It introduces Himmapan Lab, the vision, and the lab products.

    Parameters
    ----------
    logos:
        Dict of logo asset paths. Expected keys (if available):
        - "himmapan"
        - "thotsakan"
        - "maiyarap"
        - "cmkl"
        - "aice"
    repo_url:
        Repository URL for Thotsakan Statistics (optional).
    maiyarap_repo_url:
        Repository URL for Maiyarap Equations (optional).
    """

    # ---------- HERO ----------
    with gr.Column():
        gr.Markdown(
            """
# 👋 Welcome

### **Himmapan Lab**
*Interactive mathematical tools built by students, for students.*

We design software that connects theory with experimentation —
so learning feels **hands-on**, **visual**, and **fun**.
            """.strip()
        )

    gr.Markdown("---")

    # ---------- VISION ----------
    with gr.Column():
        gr.Markdown(
            """
## 🚀 Our Vision

- **Teach with interaction:** turn abstract concepts into experiments you can run.
- **Be transparent:** methods are implemented clearly so students can learn from the code.
- **Grow an ecosystem:** build multiple apps under one consistent engineering framework.
            """.strip()
        )

    gr.Markdown("---")

    # ---------- PRODUCTS ----------
    gr.Markdown("## 🧩 Products")

    with gr.Row(equal_height=True):
        # Thotsakan Statistics Card
        with gr.Column():
            if "thotsakan" in logos:
                gr.Image(
                    logos["thotsakan"],
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=120,
                    container=False,
                )
            gr.Markdown(
                """
### Thotsakan Statistics
**Probability & Statistics Interactive Laboratory**

Explore data, test ideas, and connect statistical theory with computational experimentation.
                """.strip()
            )
            if repo_url:
                gr.Markdown(f"🔗 Repository: {repo_url}")
            gr.HTML("</div>")

        # Maiyarap Equations Card
        with gr.Column():
            if "maiyarap" in logos:
                gr.Image(
                    logos["maiyarap"],
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=120,
                    container=False,
                )
            gr.Markdown(
                """
### Maiyarap Equations
**Differential Equations Interactive Laboratory**

Explore dynamic systems, visualize solutions, and connect mathematical models with computational experimentation.

*Coming soon.*
                """.strip()
            )
            if maiyarap_repo_url:
                gr.Markdown(f"🔗 Repository: {maiyarap_repo_url}")
            gr.HTML("</div>")

    gr.Markdown("---")

    # ---------- ABOUT THIS APP ----------
    with gr.Column():
        gr.Markdown(
            """
## 📌 About this application

Use the tabs at the top to begin:

- **Data**: load datasets and explore their structure  
- **Probability**: explore distributions and model randomness  
- **Estimation**: summarize data and quantify uncertainty  
- **Hypothesis Testing**: evaluate claims and compare groups  
- **Linear Regression**: model relationships and interpret evidence  

Start exploring with a dataset from `datasets/practice/`.
            """.strip()
        )

    gr.Markdown("---")

    # ---------- FOOTER / AFFILIATION ----------
    with gr.Row():
        # Keep this subtle; show logos if available.
        with gr.Column(scale=1, min_width=120):
            if "himmapan" in logos:
                gr.Image(
                    logos["himmapan"],
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=80,
                    container=False
                )

        with gr.Column(scale=1, min_width=120):
            if "cmkl" in logos:
                gr.Image(
                    logos["cmkl"],
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=80,
                    container=False
                )

        with gr.Column(scale=1, min_width=120):
            if "aice" in logos:
                gr.Image(
                    logos["aice"],
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    height=80,
                    container=False
                )

    gr.Markdown(
        """
<p style="opacity:0.75; margin-top: 6px;">
Built under <b>Himmapan Lab</b> for engineering education.

Version 5.0.0, February 2026.
</p>
        """.strip()
    )
