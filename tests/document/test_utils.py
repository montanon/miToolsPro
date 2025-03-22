from unittest import TestCase

from mitoolspro.document.document_structure import BBox, Box, Image, Line, Run
from mitoolspro.document.utils import merge_overlapping_boxes


class TestMergeOverlappingBoxes(TestCase):
    def test_empty_list(self):
        self.assertEqual(merge_overlapping_boxes([]), [])

    def test_single_box(self):
        box = Box(BBox(0, 0, 10, 10))
        merged = merge_overlapping_boxes([box])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, box.bbox)

    def test_non_overlapping_boxes(self):
        boxes = [
            Box(BBox(0, 0, 10, 10)),
            Box(BBox(0, 20, 10, 30)),
            Box(BBox(0, 40, 10, 50)),
        ]
        merged = merge_overlapping_boxes(boxes)
        self.assertEqual(len(merged), 3)
        for orig, result in zip(boxes, merged):
            self.assertEqual(orig.bbox, result.bbox)

    def test_overlapping_text_boxes(self):
        box1 = Box(BBox(0, 5, 10, 15))
        line1 = Line(BBox(0, 5, 10, 15))
        line1.add_run(Run("Arial", 12))
        box1.add_line(line1)

        box2 = Box(BBox(0, 10, 10, 20))
        line2 = Line(BBox(0, 10, 10, 20))
        line2.add_run(Run("Arial", 12))
        box2.add_line(line2)

        merged = merge_overlapping_boxes([box1, box2])
        self.assertEqual(len(merged), 1)
        merged_box = merged[0]
        self.assertEqual(merged_box.bbox, BBox(0, 5, 10, 20))
        self.assertEqual(len(merged_box.get_all_lines()), 2)

    def test_overlapping_image_boxes(self):
        box1 = Box(BBox(0, 5, 10, 15))
        box1.add_image(Image(BBox(0, 5, 10, 15), b"test1", "img1.png", "image/png"))

        box2 = Box(BBox(0, 10, 10, 20))
        box2.add_image(Image(BBox(0, 10, 10, 20), b"test2", "img2.png", "image/png"))

        merged = merge_overlapping_boxes([box1, box2])
        self.assertEqual(len(merged), 1)
        merged_box = merged[0]
        self.assertEqual(merged_box.bbox, BBox(0, 5, 10, 20))
        self.assertEqual(len(merged_box.get_all_images()), 2)

    def test_mixed_overlapping_boxes(self):
        text_box = Box(BBox(0, 5, 10, 15))
        line = Line(BBox(0, 5, 10, 15))
        line.add_run(Run("Arial", 12))
        text_box.add_line(line)

        image_box = Box(BBox(0, 10, 10, 20))
        image_box.add_image(Image(BBox(0, 10, 10, 20), b"test", "img.png", "image/png"))

        merged = merge_overlapping_boxes([text_box, image_box])
        self.assertEqual(len(merged), 2)
        # Verify boxes are adjusted to not overlap
        self.assertLessEqual(merged[0].bbox.y0, merged[1].bbox.y1)

    def test_multiple_overlapping_text_boxes(self):
        boxes = []
        for i in range(5):
            box = Box(BBox(0, i * 5, 10, (i + 2) * 5))  # Each box overlaps with next
            line = Line(box.bbox)
            line.add_run(Run("Arial", 12))
            box.add_line(line)
            boxes.append(box)

        merged = merge_overlapping_boxes(boxes)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, BBox(0, 0, 10, 25))
        self.assertEqual(len(merged[0].get_all_lines()), 5)

    def test_partial_overlapping_chain(self):
        boxes = [
            Box(BBox(0, 0, 10, 15)),  # Overlaps with box 1
            Box(BBox(0, 10, 10, 25)),  # Overlaps with box 0 and 2
            Box(BBox(0, 20, 10, 35)),  # Overlaps with box 1
        ]
        for box in boxes:
            line = Line(box.bbox)
            line.add_run(Run("Arial", 12))
            box.add_line(line)

        merged = merge_overlapping_boxes(boxes)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, BBox(0, 0, 10, 35))

    def test_complex_mixed_overlapping(self):
        boxes = [
            # Text boxes that overlap
            Box(BBox(0, 0, 10, 15)),
            Box(BBox(0, 10, 10, 25)),
            # Image boxes that overlap
            Box(BBox(20, 0, 30, 15)),
            Box(BBox(20, 10, 30, 25)),
            # Non-overlapping box
            Box(BBox(40, 0, 50, 10)),
        ]

        # Add content to boxes
        for i, box in enumerate(boxes):
            if i < 2:  # Text boxes
                line = Line(box.bbox)
                line.add_run(Run("Arial", 12))
                box.add_line(line)
            elif i < 4:  # Image boxes
                box.add_image(
                    Image(box.bbox, f"test{i}".encode(), f"img{i}.png", "image/png")
                )

        merged = merge_overlapping_boxes(boxes)
        self.assertEqual(len(merged), 3)  # 1 merged text, 1 merged image, 1 standalone

    def test_horizontal_overlap_only(self):
        box1 = Box(BBox(0, 0, 20, 10))
        box2 = Box(BBox(10, 20, 30, 30))
        merged = merge_overlapping_boxes([box1, box2])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].bbox, box1.bbox)
        self.assertEqual(merged[1].bbox, box2.bbox)

    def test_exact_overlap(self):
        bbox = BBox(0, 0, 10, 10)
        box1 = Box(bbox)
        box2 = Box(bbox)
        line = Line(bbox)
        line.add_run(Run("Arial", 12))
        box1.add_line(line)
        box2.add_line(line)

        merged = merge_overlapping_boxes([box1, box2])
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].get_all_lines()), 2)

    def test_nested_boxes(self):
        outer = Box(BBox(0, 0, 20, 20))
        inner = Box(BBox(5, 5, 15, 15))
        line = Line(inner.bbox)
        line.add_run(Run("Arial", 12))
        outer.add_line(line)
        inner.add_line(line)

        merged = merge_overlapping_boxes([outer, inner])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox, outer.bbox)

    def test_sorting_order(self):
        boxes = [
            Box(BBox(0, 0, 10, 10)),
            Box(BBox(0, 20, 10, 30)),
            Box(BBox(0, 10, 10, 20)),
        ]
        merged = merge_overlapping_boxes(boxes)
        # Verify boxes are sorted by y1 descending
        for i in range(len(merged) - 1):
            self.assertGreaterEqual(merged[i].bbox.y1, merged[i + 1].bbox.y1)
