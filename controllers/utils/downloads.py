import re
import tempfile
import gradio as gr

def sanitize_filename(name: str, default: str):
    if not name or not name.strip():
        return default
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean if clean else default


def dataframe_to_csv(df, filename):
    if df is None:
        gr.Warning("❌ No table available to download.")
        return None

    base = sanitize_filename(filename, "descriptive_statistics")

    with tempfile.NamedTemporaryFile(
        delete=False,
        mode="w",
        suffix=".csv",
        prefix=base + "_",
        encoding="utf-8",
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        return tmp.name

def figure_to_png(fig, filename: str):
    if fig is None:
        return None

    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png",
        prefix=filename + "_"
    )
    fig.savefig(tmp.name, dpi=200, bbox_inches="tight")
    return tmp.name