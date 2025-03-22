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


def extract_images_from_pdf(
    pdf_path: Path, image_extension: str = "png"
) -> list[list[Box]]:
    try:
        doc = fitz.open(pdf_path)
        all_boxes = []

        for page_index, page in enumerate(doc):
            boxes = []
            found_xrefs = {}  # Track count of each xref
            image_positions = {}  # Track original positions of images
            processed_blocks = set()  # Track which blocks we've processed

            try:
                images_info = page.get_images(full=True)
                print(f"Found {len(images_info)} images on page {page_index}")
                xref_to_info = {img[0]: img for img in images_info}
            except Exception as e:
                print(f"Error getting images info: {str(e)}")
                continue

            try:
                blocks = page.get_text("dict")["blocks"]
                print(f"Found {len(blocks)} blocks on page {page_index}")
            except Exception as e:
                print(f"Error getting blocks: {str(e)}")
                continue

            # First pass: collect all image positions from blocks
            for block_idx, block in enumerate(blocks):
                if block["type"] == 1:  # image block
                    xref = block["image"]
                    bbox = block["bbox"]
                    if xref not in image_positions:
                        image_positions[xref] = []
                    image_positions[xref].append((block_idx, bbox))

            # Method 1: Process all image blocks
            for block_idx, block in enumerate(blocks):
                if (
                    block["type"] == 1 and block_idx not in processed_blocks
                ):  # image block
                    xref = block["image"]
                    print(f"Processing block image xref: {xref}")

                    try:
                        bbox = BBox(*block["bbox"])

                        # Handle case where xref is raw image data
                        if isinstance(xref, bytes):
                            img_bytes = xref
                            image = Image(
                                bbox=bbox,
                                stream=img_bytes,
                                name=f"image_page{page_index}_block{block_idx}.{image_extension}",
                                mimetype=f"image/{image_extension}",
                            )
                            box = Box(bbox)
                            box.add_image(image)
                            boxes.append(box)
                            processed_blocks.add(block_idx)
                            print(
                                f"Successfully extracted block image from raw data (block {block_idx})"
                            )
                            continue

                        # Normal case: xref is a reference
                        img_info = xref_to_info.get(xref)
                        if not img_info:
                            continue

                        pix = fitz.Pixmap(doc, xref)
                        if pix.width > 0 and pix.height > 0:
                            if pix.n - pix.alpha > 3:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            img_bytes = pix.tobytes("png")

                            # Get instance count for this block position
                            count = found_xrefs.get(xref, 0)

                            image = Image(
                                bbox=bbox,
                                stream=img_bytes,
                                name=f"image_page{page_index}_xref{xref}_{count}.{image_extension}",
                                mimetype=f"image/{image_extension}",
                            )
                            box = Box(bbox)
                            box.add_image(image)
                            boxes.append(box)
                            found_xrefs[xref] = count + 1
                            processed_blocks.add(block_idx)
                            print(
                                f"Successfully extracted block image {xref} (instance {count})"
                            )
                    except Exception as e:
                        print(f"Error processing block image {xref}: {str(e)}")
                        continue

            # Method 2: Process any remaining image positions
            for xref, positions in image_positions.items():
                for block_idx, bbox in positions:
                    if block_idx in processed_blocks:
                        continue

                    try:
                        # Handle case where xref is raw image data
                        if isinstance(xref, bytes):
                            img_bytes = xref
                            image = Image(
                                bbox=BBox(*bbox),
                                stream=img_bytes,
                                name=f"image_page{page_index}_block{block_idx}.{image_extension}",
                                mimetype=f"image/{image_extension}",
                            )
                            box = Box(BBox(*bbox))
                            box.add_image(image)
                            boxes.append(box)
                            processed_blocks.add(block_idx)
                            print(
                                f"Successfully extracted remaining image from raw data (block {block_idx})"
                            )
                            continue

                        # Normal case: xref is a reference
                        img_info = xref_to_info.get(xref)
                        if not img_info:
                            continue

                        pix = fitz.Pixmap(doc, xref)
                        if pix.width > 0 and pix.height > 0:
                            if pix.n - pix.alpha > 3:
                                pix = fitz.Pixmap(fitz.csRGB, pix)
                            img_bytes = pix.tobytes("png")

                            count = found_xrefs.get(xref, 0)
                            name = f"image_page{page_index}_xref{xref}_{count}.{image_extension}"
                            print(f"Creating image with name: {name}")
                            image = Image(
                                bbox=BBox(*bbox),
                                stream=img_bytes,
                                name=name,
                                mimetype=f"image/{image_extension}",
                            )
                            box = Box(BBox(*bbox))
                            box.add_image(image)
                            boxes.append(box)
                            found_xrefs[xref] = count + 1
                            processed_blocks.add(block_idx)
                            print(
                                f"Successfully extracted remaining image {xref} (instance {count})"
                            )
                    except Exception as e:
                        print(f"Error processing remaining image {xref}: {str(e)}")
                        continue

            all_boxes.append(boxes)

        doc.close()
        return all_boxes
    except Exception as e:
        print(f"Top level error: {str(e)}")
        if "doc" in locals():
            doc.close()
        raise


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

    for page_index, page_layout in enumerate(page_layouts):
        page = Page(page_layout.width, page_layout.height)
        boxes = []

        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                bbox = BBox(element.x0, element.y0, element.x1, element.y1)
                box = Box(bbox)
                for line_obj in element:
                    if isinstance(line_obj, LTTextLineHorizontal):
                        line = extract_lines_from_pdf(line_obj)
                        box.add_line(line)
                boxes.append(box)

        boxes.extend(image_boxes_per_page[page_index])

        boxes.sort(key=lambda b: (-b.bbox.y1, b.bbox.x0))

        for box in boxes:
            page.add_box(box)

        doc.add_page(page)

    return doc
