from math import isclose
from pathlib import Path
from typing import Dict, List, Tuple, Union

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


def extract_images_from_pdf(
    pdf_path: Path, image_extension: str = "png"
) -> List[List[Box]]:
    def to_bbox(coords: Union[Tuple[float, float, float, float], fitz.Rect]) -> BBox:
        if isinstance(coords, fitz.Rect):
            return BBox(coords.x0, coords.y0, coords.x1, coords.y1)
        return BBox(*coords)

    def build_image_box(bbox: BBox, img_bytes: bytes, name: str, mimetype: str) -> Box:
        image = Image(bbox=bbox, stream=img_bytes, name=name, mimetype=mimetype)
        box = Box(bbox)
        box.add_image(image)
        return box

    all_boxes: List[List[Box]] = []
    doc = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(doc):
            boxes: List[Box] = []
            found_xrefs: Dict[int, int] = {}
            image_positions: Dict[Union[int, bytes], List[Tuple[int, Tuple]]] = {}
            processed_blocks: set = set()

            # Get image info and block structure
            try:
                xref_to_info = {img[0]: img for img in page.get_images(full=True)}
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                all_boxes.append([])
                continue

            # Collect block image positions
            for block_idx, block in enumerate(blocks):
                if block["type"] == 1:  # image block
                    xref = block["image"]
                    bbox = block["bbox"]
                    image_positions.setdefault(xref, []).append((block_idx, bbox))

            # Process block images
            for block_idx, block in enumerate(blocks):
                if block["type"] != 1 or block_idx in processed_blocks:
                    continue

                xref = block["image"]
                bbox = to_bbox(block["bbox"])

                try:
                    if isinstance(xref, bytes):
                        name = (
                            f"image_page{page_index}_block{block_idx}.{image_extension}"
                        )
                        box = build_image_box(
                            bbox=bbox,
                            img_bytes=xref,
                            name=name,
                            mimetype=f"image/{image_extension}",
                        )
                        boxes.append(box)
                        processed_blocks.add(block_idx)
                        continue

                    if xref not in xref_to_info:
                        continue

                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_bytes = pix.tobytes(image_extension)
                    count = found_xrefs.get(xref, 0)
                    name = (
                        f"image_page{page_index}_xref{xref}_{count}.{image_extension}"
                    )

                    box = build_image_box(
                        bbox=bbox,
                        img_bytes=img_bytes,
                        name=name,
                        mimetype=f"image/{image_extension}",
                    )
                    boxes.append(box)
                    found_xrefs[xref] = count + 1
                    processed_blocks.add(block_idx)

                except Exception:
                    continue  # skip problematic images

            # Process any remaining xrefs not picked up by block method
            for xref, positions in image_positions.items():
                for block_idx, bbox_coords in positions:
                    if block_idx in processed_blocks:
                        continue

                    bbox = to_bbox(bbox_coords)

                    try:
                        if isinstance(xref, bytes):
                            name = f"image_page{page_index}_block{block_idx}.{image_extension}"
                            box = build_image_box(
                                bbox=bbox,
                                img_bytes=xref,
                                name=name,
                                mimetype=f"image/{image_extension}",
                            )
                            boxes.append(box)
                            processed_blocks.add(block_idx)
                            continue

                        if xref not in xref_to_info:
                            continue

                        pix = fitz.Pixmap(doc, xref)
                        if pix.n - pix.alpha > 3:
                            pix = fitz.Pixmap(fitz.csRGB, pix)

                        img_bytes = pix.tobytes(image_extension)
                        count = found_xrefs.get(xref, 0)
                        name = f"image_page{page_index}_xref{xref}_{count}.{image_extension}"

                        box = build_image_box(
                            bbox=bbox,
                            img_bytes=img_bytes,
                            name=name,
                            mimetype=f"image/{image_extension}",
                        )
                        boxes.append(box)
                        found_xrefs[xref] = count + 1
                        processed_blocks.add(block_idx)

                    except Exception:
                        continue

            all_boxes.append(boxes)

    finally:
        doc.close()

    return all_boxes


def extract_lines_from_pdf(line_obj: LTTextLineHorizontal) -> Line:
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


def pdf_to_document(pdf_path: Path) -> Document:
    doc = Document()
    try:
        page_layouts = extract_pages(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {e}")
    try:
        image_boxes_per_page = extract_images_from_pdf(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to extract images from PDF: {e}")

    def merge_overlapping_boxes(boxes: list[Box]) -> list[Box]:
        if not boxes:
            return boxes

        # Sort boxes by top edge (y1) descending, then left edge (x0)
        boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

        merged = []
        current_box = boxes[0]

        for next_box in boxes[1:]:
            if current_box.bbox.overlaps(next_box.bbox):
                # If boxes overlap and are of the same type (both text or both image)
                if bool(current_box.get_all_images()) == bool(
                    next_box.get_all_images()
                ):
                    # Merge boxes by extending the bbox
                    new_bbox = BBox(
                        min(current_box.bbox.x0, next_box.bbox.x0),
                        min(current_box.bbox.y0, next_box.bbox.y0),
                        max(current_box.bbox.x1, next_box.bbox.x1),
                        max(current_box.bbox.y1, next_box.bbox.y1),
                    )
                    merged_box = Box(new_bbox)

                    # Copy content from both boxes
                    for line in current_box.get_all_lines():
                        merged_box.add_line(line)
                    for line in next_box.get_all_lines():
                        merged_box.add_line(line)
                    for image in current_box.get_all_images():
                        merged_box.add_image(image)
                    for image in next_box.get_all_images():
                        merged_box.add_image(image)

                    current_box = merged_box
                else:
                    # If different types, keep them separate but adjust positions
                    next_box.bbox.y1 = min(next_box.bbox.y1, current_box.bbox.y0)
                    merged.append(current_box)
                    current_box = next_box
            else:
                merged.append(current_box)
                current_box = next_box

        merged.append(current_box)
        return merged

    for page_index, page_layout in enumerate(page_layouts):
        page = Page(page_layout.width, page_layout.height)
        boxes = []

        # Collect text boxes
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                bbox = BBox(element.x0, element.y0, element.x1, element.y1)
                box = Box(bbox)
                for line_obj in element:
                    if isinstance(line_obj, LTTextLineHorizontal):
                        line = extract_lines_from_pdf(line_obj)
                        box.add_line(line)
                boxes.append(box)

        # Add image boxes
        boxes.extend(image_boxes_per_page[page_index])

        # Merge overlapping boxes and add to page
        merged_boxes = merge_overlapping_boxes(boxes)
        for box in merged_boxes:
            page.add_box(box)

        doc.add_page(page)

    return doc
