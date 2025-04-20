import unittest
from math import isclose
from unittest import TestCase

import pandas as pd

from mitoolspro.document.document_structure import BBox, Char, Run
from mitoolspro.document.runs_utils import (
    center_runs_vertically,
    create_run_in_bbox,
    find_item_runs,
    get_char_properties,
    iterate_all_runs,
)


class TestCenterRunsVertically(TestCase):
    def setUp(self):
        self.char1 = Char(
            text="a",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=0, y0=0, x1=10, y1=12),
        )
        self.char2 = Char(
            text="b",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=10, y0=0, x1=20, y1=12),
        )
        self.run1 = Run.from_chars([self.char1, self.char2], fontname="Arial", size=12)

        self.char3 = Char(
            text="c",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=0, y0=0, x1=10, y1=12),
        )
        self.char4 = Char(
            text="d",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=10, y0=0, x1=20, y1=12),
        )
        self.run2 = Run.from_chars([self.char3, self.char4], fontname="Arial", size=12)

    def test_center_runs_vertically(self):
        runs = [self.run1, self.run2]
        reference_y = 50
        centered_runs = center_runs_vertically(runs, reference_y)

        self.assertEqual(len(centered_runs), 2)
        self.assertEqual(len(centered_runs[0].chars), 2)
        self.assertEqual(len(centered_runs[1].chars), 2)

        y_positions = [char.bbox.y0 for char in centered_runs[0].chars]
        self.assertTrue(
            all(isclose(y, y_positions[0], rel_tol=1e-9) for y in y_positions)
        )

    def test_center_runs_vertically_with_step(self):
        runs = [self.run1, self.run2]
        reference_y = 50
        step = 8
        centered_runs = center_runs_vertically(runs, reference_y, step)

        self.assertEqual(len(centered_runs), 2)
        self.assertEqual(len(centered_runs[0].chars), 2)
        self.assertEqual(len(centered_runs[1].chars), 2)

        y_positions = [char.bbox.y0 for char in centered_runs[0].chars]
        self.assertTrue(
            all(isclose(y, y_positions[0], rel_tol=1e-9) for y in y_positions)
        )


class TestFindItemRuns(TestCase):
    def setUp(self):
        self.char1 = Char(
            text="I",
            fontname="Arial-Bold",
            size=12,
            bbox=BBox(x0=0, y0=0, x1=10, y1=12),
        )
        self.char2 = Char(
            text="t",
            fontname="Arial-Bold",
            size=12,
            bbox=BBox(x0=10, y0=0, x1=20, y1=12),
        )
        self.run1 = Run.from_chars(
            [self.char1, self.char2], fontname="Arial-Bold", size=12
        )

    def test_find_item_runs_with_item_format(self):
        sections = [
            [
                Run.from_chars(
                    [
                        Char(
                            text=c,
                            fontname="Arial-Bold",
                            size=12,
                            bbox=BBox(0, 0, 10, 12),
                        )
                        for c in "Item 1:"
                    ],
                    fontname="Arial-Bold",
                    size=12,
                )
            ]
        ]
        item_indices = find_item_runs(sections)
        self.assertEqual(len(item_indices), 1)

    def test_find_item_runs_with_letter_format(self):
        sections = [
            [
                Run.from_chars(
                    [
                        Char(
                            text=c,
                            fontname="Arial-Bold",
                            size=12,
                            bbox=BBox(0, 0, 10, 12),
                        )
                        for c in "a."
                    ],
                    fontname="Arial-Bold",
                    size=12,
                )
            ]
        ]
        item_indices = find_item_runs(sections)
        self.assertEqual(len(item_indices), 1)

    def test_find_item_runs_with_number_format(self):
        sections = [
            [
                Run.from_chars(
                    [
                        Char(
                            text=c,
                            fontname="Arial-Bold",
                            size=12,
                            bbox=BBox(0, 0, 10, 12),
                        )
                        for c in "1)"
                    ],
                    fontname="Arial-Bold",
                    size=12,
                )
            ]
        ]
        item_indices = find_item_runs(sections)
        self.assertEqual(len(item_indices), 1)

    def test_find_item_runs_with_non_bold(self):
        sections = [
            [
                Run.from_chars(
                    [
                        Char(
                            text=c,
                            fontname="Arial",
                            size=12,
                            bbox=BBox(0, 0, 10, 12),
                        )
                        for c in "Item 1:"
                    ],
                    fontname="Arial",
                    size=12,
                )
            ]
        ]
        item_indices = find_item_runs(sections)
        self.assertEqual(len(item_indices), 0)


class TestIterateAllRuns(TestCase):
    def setUp(self):
        self.char1 = Char(
            text="a",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=0, y0=0, x1=10, y1=12),
        )
        self.char2 = Char(
            text="b",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=10, y0=0, x1=20, y1=12),
        )
        self.run1 = Run.from_chars([self.char1, self.char2], fontname="Arial", size=12)

        self.char3 = Char(
            text="c",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=0, y0=0, x1=10, y1=12),
        )
        self.char4 = Char(
            text="d",
            fontname="Arial",
            size=12,
            bbox=BBox(x0=10, y0=0, x1=20, y1=12),
        )
        self.run2 = Run.from_chars([self.char3, self.char4], fontname="Arial", size=12)

    def test_iterate_all_runs_with_nested_lists(self):
        sections = [
            [self.run1, self.run2],
            self.run1,
            [self.run2],
        ]

        runs = list(iterate_all_runs(sections))
        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0][0], self.run1)
        self.assertEqual(runs[1][0], self.run2)
        self.assertEqual(runs[2][0], self.run1)
        self.assertEqual(runs[3][0], self.run2)

    def test_iterate_all_runs_with_single_run(self):
        sections = [self.run1]
        runs = list(iterate_all_runs(sections))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][0], self.run1)

    def test_iterate_all_runs_with_empty_sections(self):
        sections = []
        runs = list(iterate_all_runs(sections))
        self.assertEqual(len(runs), 0)


class TestCreateRunInBbox(TestCase):
    def setUp(self):
        self.chars_data = pd.DataFrame(
            {
                "fontname": ["Arial", "Arial"],
                "size": [12, 12],
                "text": ["a", "b"],
                "width": [10, 10],
            }
        )

    def test_create_run_in_bbox(self):
        text = "ab"
        fontname = "Arial"
        size = 12
        bbox = BBox(x0=0, y0=0, x1=20, y1=12)

        run = create_run_in_bbox(text, fontname, size, bbox, self.chars_data)

        self.assertEqual(len(run.chars), 2)
        self.assertEqual(run.fontname, fontname)
        self.assertEqual(run.size, size)
        self.assertTrue(all(char.fontname == fontname for char in run.chars))
        self.assertTrue(all(char.size == size for char in run.chars))

    def test_create_run_in_bbox_with_single_char(self):
        text = "a"
        fontname = "Arial"
        size = 12
        bbox = BBox(x0=0, y0=0, x1=10, y1=12)

        run = create_run_in_bbox(text, fontname, size, bbox, self.chars_data)

        self.assertEqual(len(run.chars), 1)
        self.assertEqual(run.fontname, fontname)
        self.assertEqual(run.size, size)
        self.assertEqual(run.chars[0].text, "a")


class TestGetCharProperties(TestCase):
    def setUp(self):
        self.chars_data = pd.DataFrame(
            {
                "fontname": ["Arial", "Arial", "Times"],
                "size": [12, 12, 12],
                "text": ["a", "b", "a"],
                "width": [10, 10, 8],
            }
        )

    def test_get_char_properties(self):
        properties = get_char_properties("a", "Arial", 12, self.chars_data)
        self.assertEqual(len(properties), 1)
        self.assertEqual(properties["text"].iloc[0], "a")
        self.assertEqual(properties["width"].iloc[0], 10)

    def test_get_char_properties_with_different_font(self):
        properties = get_char_properties("a", "Times", 12, self.chars_data)
        self.assertEqual(len(properties), 1)
        self.assertEqual(properties["text"].iloc[0], "a")
        self.assertEqual(properties["width"].iloc[0], 8)

    def test_get_char_properties_with_nonexistent_char(self):
        properties = get_char_properties("x", "Arial", 12, self.chars_data)
        self.assertEqual(len(properties), 0)


if __name__ == "__main__":
    unittest.main()
