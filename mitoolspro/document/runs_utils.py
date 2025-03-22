import re
from itertools import accumulate
from math import isclose
from typing import List

from pandas import DataFrame

from mitoolspro.document.document_structure import BBox, Char, Run


def center_runs_vertically(runs: List[Run], reference_y: float, step: int = 4):
    sizes = [run.size for run in runs]
    ypositions = []
    current_pos = 0
    current_step = step
    for size in sizes:
        ypositions.append(current_pos)
        current_pos += current_step + size
        current_step += step
    ypositions = ypositions[::-1]
    current_center = (ypositions[0] + ypositions[-1]) / 2
    shift = reference_y - current_center
    ypositions = [y + shift for y in ypositions]

    for i, run in enumerate(runs):
        run.chars = [
            Char(
                text=char.text,
                fontname=char.fontname,
                size=char.size,
                bbox=BBox(
                    x0=char.bbox.x0,
                    y0=ypositions[i],
                    x1=char.bbox.x1,
                    y1=ypositions[i] + char.size,
                ),
            )
            for char in run.chars
        ]
    return runs


def find_item_runs(sections):
    item_indices = []
    item_pattern = re.compile(r"^(?:Item\s+\d+:|[a-z][\.\)]|\d+[\.\)])", re.IGNORECASE)
    for run, idx in iterate_all_runs(sections):
        if run.is_bold() and item_pattern.match(run.text):
            item_indices.append(idx)
    return item_indices


def iterate_all_runs(sections):
    run_index = 0
    for elem in sections:
        if isinstance(elem, list):
            for run in elem:
                yield run, run_index
                run_index += 1
        else:
            yield elem, run_index
            run_index += 1


def create_run_in_bbox(
    text: str, fontname: str, size: float, bbox: BBox, chars_data: DataFrame
):
    def center_positions(positions: List[float], center: float):
        current_center = (positions[0] + positions[-1]) / 2
        shift = center - current_center
        return [position + shift for position in positions]

    xcenter = bbox.xcenter
    widths = [
        get_char_properties(char, fontname, size, chars_data)["width"].item()
        for char in text
    ]
    xpositions = list(accumulate([width for width in widths]))
    xpositions = center_positions(xpositions, xcenter)
    chars = [
        Char(
            text=char,
            fontname=fontname,
            size=size,
            x0=xposition,
            y0=bbox.y0,
            x1=xposition + width,
            y1=bbox.y1,
        )
        for char, xposition, width in zip(text, xpositions, widths)
    ]
    return Run.from_chars(chars, fontname=fontname, size=size)


def get_char_properties(char: str, fontname: str, size: float, chars_data: DataFrame):
    properties = chars_data[
        (chars_data["fontname"] == fontname) & (chars_data["size"] == size)
    ]
    return properties[properties["text"] == char]
