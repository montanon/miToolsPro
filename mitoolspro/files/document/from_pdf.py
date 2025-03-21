from math import isclose
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTAnno, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal

from mitoolspro.files.document.document_structure import (
    Box,
    Char,
    Document,
    Line,
    Page,
    Run,
)


def _parse_line_from_lt_line(line_obj: LTTextLineHorizontal) -> Line:
    line = Line(line_obj.x0, line_obj.y0, line_obj.x1, line_obj.y1)
    current_run = None
    for char_obj in line_obj:
        if isinstance(char_obj, LTChar):
            cfont = char_obj.fontname
            csize = char_obj.size
            cchar = char_obj.get_text()
            if (
                current_run is None
                or current_run.fontname != cfont
                or not isclose(current_run.size, csize, abs_tol=0.01)
            ):
                if current_run:
                    line.add_run(current_run)
                current_run = Run(cfont, csize)
            current_run.append_char(
                Char(
                    cchar,
                    cfont,
                    csize,
                    char_obj.x0,
                    char_obj.y0,
                    char_obj.x1,
                    char_obj.y1,
                )
            )
        elif isinstance(char_obj, LTAnno) and current_run:
            space = char_obj.get_text()
            current_run.append_char(
                Char(space, current_run.fontname, current_run.size, 0, 0, 0, 0)
            )
    if current_run:
        line.add_run(current_run)
    return line


def parse_pdf_to_structure(pdf_path: Path) -> Document:
    doc = Document()
    try:
        page_layouts = extract_pages(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")

    for page_layout in page_layouts:
        page = Page(page_layout.width, page_layout.height)
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                box = Box(element.x0, element.y0, element.x1, element.y1)
                for line_obj in element:
                    if isinstance(line_obj, LTTextLineHorizontal):
                        line = _parse_line_from_lt_line(line_obj)
                        box.add_line(line)
                page.add_box(box)
        doc.add_page(page)
    return doc
