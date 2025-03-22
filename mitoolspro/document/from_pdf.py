from math import isclose
from pathlib import Path

import fitz
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTAnno, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal

from mitoolspro.document.document_structure import (
    CHAR_SIZE_TOLERANCE,
    BBox,
    Box,
    Char,
    Document,
    Image,
    Line,
    Page,
    Run,
)


def _extract_image_boxes(
    pdf_path: Path, image_extension: str = "png"
) -> list[list[Box]]:
    doc = fitz.open(pdf_path)
    all_boxes = []
    for page_index, page in enumerate(doc):
        boxes = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 1:  # image block
                bbox = BBox(*block["bbox"])
                xref = block["image"]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    img_bytes = pix.tobytes("png")
                except Exception:
                    continue  # skip problematic images
                image = Image(
                    bbox=bbox,
                    stream=img_bytes,
                    name=f"image_page{page_index}_xref{xref}.{image_extension}",
                    mimetype=f"image/{image_extension}",
                )
                box = Box(bbox)
                box.add_image(image)
                boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes


def _parse_line_from_lt_line(line_obj: LTTextLineHorizontal) -> Line:
    line = Line(BBox(line_obj.x0, line_obj.y0, line_obj.x1, line_obj.y1))
    current_run = None
    for char_obj in line_obj:
        if isinstance(char_obj, LTChar):
            cfont = char_obj.fontname
            csize = char_obj.size
            cchar = char_obj.get_text()
            if (
                current_run is None
                or current_run.fontname != cfont
                or not isclose(current_run.size, csize, abs_tol=CHAR_SIZE_TOLERANCE)
            ):
                if current_run:
                    line.add_run(current_run)
                current_run = Run(cfont, csize)
            current_run.append_char(
                Char(
                    cchar,
                    cfont,
                    csize,
                    BBox(char_obj.x0, char_obj.y0, char_obj.x1, char_obj.y1),
                )
            )
        elif isinstance(char_obj, LTAnno) and current_run:
            space = char_obj.get_text()
            current_run.append_char(
                Char(space, current_run.fontname, current_run.size, BBox(0, 0, 0, 0))
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
    try:
        image_boxes_per_page = _extract_image_boxes(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to extract images from PDF: {e}")

    for page_index, page_layout in enumerate(page_layouts):
        page = Page(page_layout.width, page_layout.height)
        boxes = []

        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                bbox = BBox(element.x0, element.y0, element.x1, element.y1)
                box = Box(bbox)
                for line_obj in element:
                    if isinstance(line_obj, LTTextLineHorizontal):
                        line = _parse_line_from_lt_line(line_obj)
                        box.add_line(line)
                boxes.append(box)

        boxes.extend(image_boxes_per_page[page_index])

        boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

        for box in boxes:
            page.add_box(box)

        doc.add_page(page)

    return doc
