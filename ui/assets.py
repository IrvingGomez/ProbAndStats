from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

def load_logo(name: str):
    return Image.open(
        BASE_DIR / "assets" / "logos" / name
    )

LOGOS = {
    "thotsakan": load_logo("ThotsakanStats.png"),
    "maiyarap": load_logo("MaiyarapEq.png"),
    "himmapan": load_logo("HimmapanLab.png"),
    "cmkl": load_logo("CmklLogo.png"),
    "aice": load_logo("AiceLogo.png"),
}
