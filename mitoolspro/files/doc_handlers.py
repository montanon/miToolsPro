from pathlib import Path
from typing import List, Union

from docx import Document


def read_docx_file(file_path: Union[str, Path], indent: str = "-") -> List[str]:
    document = Document(file_path)
    paragraphs = []
    previous_style = None
    previous_punctuation = False
    n_tabs = 0
    for n, para in enumerate(document.paragraphs):
        if para.text:
            text = para.text.strip()
            formatted_text = ""
            if previous_style is None:
                previous_style = para.style.name
            if (
                para.style.name != previous_style
                or text.endswith(":")
                and not previous_style.find("Heading")
            ):
                if previous_style == "Normal" and para.style.name == "List Paragraph":
                    n_tabs += 1
                elif previous_style == "List Paragraph" and para.style.name == "Normal":
                    n_tabs -= 1
                if text.endswith(":"):
                    n_tabs += 1
            previous_style = para.style.name
            style_name = para.style.name
            previous_punctuation = text.endswith(":")
            if "Heading" in style_name:
                formatted_text += f"{'#' * int(style_name[-1])} "
            highlighted = False
            for run in para.runs:
                run_text = run.text
                if run.bold:
                    run_text = f"**{run_text}**"  # Mark bold text
                if run.italic:
                    run_text = f"*{run_text}*"  # Mark italic text
                if run.font.highlight_color:
                    highlighted = True
                formatted_text += run_text
            if highlighted:
                formatted_text = f"<highlight>{formatted_text}<\highlight>"
            indentation = "\t" * n_tabs + indent
            paragraphs.append(
                f"{indentation if n_tabs != 0 else ''}{formatted_text} -- ({para.style.name}, {previous_style}, {previous_punctuation})"
            )
    return paragraphs
