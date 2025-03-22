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
        # Box 4 extends to y=30, and since all boxes overlap in sequence,
        # they form a chain from y=0 to y=30
        self.assertEqual(merged[0].bbox, BBox(0, 0, 10, 30))
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
            Box(BBox(0, 0, 10, 15)),  # Text box 1
            Box(BBox(0, 10, 10, 25)),  # Text box 2
            # Image boxes that overlap
            Box(BBox(20, 0, 30, 15)),  # Image box 1
            Box(BBox(20, 10, 30, 25)),  # Image box 2
            # Non-overlapping box
            Box(BBox(40, 0, 50, 10)),  # Standalone box
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

        # Debug: Print initial box states
        print("\nInitial boxes:")
        for i, box in enumerate(boxes):
            print(
                f"Box {i}: {box.bbox}, Lines: {len(box.get_all_lines())}, Images: {len(box.get_all_images())}"
            )

        merged = merge_overlapping_boxes(boxes)

        # Debug: Print merged box states
        print("\nMerged boxes:")
        for i, box in enumerate(merged):
            print(
                f"Box {i}: {box.bbox}, Lines: {len(box.get_all_lines())}, Images: {len(box.get_all_images())}"
            )

        # Should result in 3 boxes:
        # 1. Merged text box (from boxes[0] and boxes[1])
        # 2. Merged image box (from boxes[2] and boxes[3])
        # 3. Standalone box (boxes[4])
        self.assertEqual(
            len(merged),
            3,
            f"Should have 3 boxes: merged text, merged image, and standalone. Got {len(merged)} boxes",
        )

        # Find the boxes by their content type and position
        text_boxes = [b for b in merged if len(b.get_all_lines()) > 0]
        image_boxes = [b for b in merged if len(b.get_all_images()) > 0]
        standalone_boxes = [b for b in merged if b.bbox.x0 >= 40]

        # Debug: Print categorized boxes
        print("\nCategorized boxes:")
        print(f"Text boxes: {len(text_boxes)}")
        for b in text_boxes:
            print(f"  Text box: {b.bbox}, Lines: {len(b.get_all_lines())}")
        print(f"Image boxes: {len(image_boxes)}")
        for b in image_boxes:
            print(f"  Image box: {b.bbox}, Images: {len(b.get_all_images())}")
        print(f"Standalone boxes: {len(standalone_boxes)}")
        for b in standalone_boxes:
            print(f"  Standalone box: {b.bbox}")

        # Verify text boxes
        self.assertEqual(len(text_boxes), 1, "Should have one merged text box")
        merged_text = text_boxes[0]
        self.assertEqual(
            merged_text.bbox, BBox(0, 0, 10, 25), "Text box should span full height"
        )
        self.assertEqual(
            len(merged_text.get_all_lines()), 2, "Should contain both text lines"
        )

        # Verify image boxes
        self.assertEqual(len(image_boxes), 1, "Should have one merged image box")
        merged_image = image_boxes[0]
        self.assertEqual(
            merged_image.bbox, BBox(20, 0, 30, 25), "Image box should span full height"
        )
        self.assertEqual(
            len(merged_image.get_all_images()), 2, "Should contain both images"
        )

        # Verify standalone box
        self.assertEqual(len(standalone_boxes), 1, "Should have one standalone box")
        self.assertEqual(
            standalone_boxes[0].bbox,
            BBox(40, 0, 50, 10),
            "Standalone box should be unchanged",
        )

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
