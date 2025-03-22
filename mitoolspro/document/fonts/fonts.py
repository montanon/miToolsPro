from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONTS_DIR = Path(__file__).parent

FONT_FILES = {
    "arial": {
        "normal": "arialmt.ttf",
        "bold": "Arial Bold.ttf",
        "italic": "Arial Italic.ttf",
        "bold-italic": "Arial Bold Italic.ttf",
    }
}

FONT_MAPPING = {
    "arial": {
        "normal": "Arial",
        "bold": "Arial-Bold",
        "italic": "Arial-Italic",
        "bold-italic": "Arial-BoldItalic",
    }
}


def register_fonts(fontfamily: str, font_dir: Path = FONTS_DIR) -> None:
    if not font_dir.exists():
        return

    for style, filename in FONT_FILES[fontfamily].items():
        font_path = font_dir / filename
        if font_path.exists():
            font_name = FONT_MAPPING[fontfamily][style]
            pdfmetrics.registerFont(TTFont(font_name, font_path))


def select_font(fontfamily: str, fontname: str) -> str:
    name = fontname.lower()
    if "bold" in name and ("italic" in name or "oblique" in name):
        return FONT_MAPPING[fontfamily]["bold-italic"]
    elif "bold" in name:
        return FONT_MAPPING[fontfamily]["bold"]
    elif "italic" in name or "oblique" in name:
        return FONT_MAPPING[fontfamily]["italic"]
    else:
        return FONT_MAPPING[fontfamily]["normal"]
