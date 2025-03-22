import unittest
from unittest import TestCase

from mitoolspro.document.document_structure import (
    BBox,
    Box,
    Char,
    Document,
    Image,
    Line,
    Page,
    Run,
)


class TestBBox(TestCase):
    def setUp(self):
        self.bbox = BBox(10, 20, 30, 50)

    def test_width(self):
        self.assertEqual(self.bbox.width, 20)

    def test_height(self):
        self.assertEqual(self.bbox.height, 30)

    def test_center(self):
        self.assertEqual(self.bbox.center, (20, 35))

    def test_xcenter(self):
        self.assertEqual(self.bbox.xcenter, 20)

    def test_ycenter(self):
        self.assertEqual(self.bbox.ycenter, 35)

    def test_clone(self):
        cloned = self.bbox.clone()
        self.assertEqual(cloned, self.bbox)
        cloned.x0 = 100
        self.assertNotEqual(cloned.x0, self.bbox.x0)

    def test_merge(self):
        bbox1 = BBox(10, 20, 30, 40)
        bbox2 = BBox(20, 30, 40, 50)
        merged = bbox1.merge(bbox2)

        self.assertEqual(merged.x0, min(bbox1.x0, bbox2.x0))  # min of x0s
        self.assertEqual(merged.y0, min(bbox1.y0, bbox2.y0))  # min of y0s
        self.assertEqual(merged.x1, max(bbox1.x1, bbox2.x1))  # max of x1s
        self.assertEqual(merged.y1, max(bbox1.y1, bbox2.y1))  # max of y1s

        # Test merging with non-overlapping box
        bbox3 = BBox(100, 100, 200, 200)
        merged = bbox1.merge(bbox3)
        self.assertEqual(merged.x0, min(bbox1.x0, bbox3.x0))
        self.assertEqual(merged.y0, min(bbox1.y0, bbox3.y0))
        self.assertEqual(merged.x1, max(bbox1.x1, bbox3.x1))
        self.assertEqual(merged.y1, max(bbox1.y1, bbox3.y1))

    def test_equality(self):
        bbox1 = BBox(10, 20, 30, 50)
        bbox2 = BBox(10, 20, 30, 50)
        bbox3 = BBox(10, 20, 30, 51)

        self.assertEqual(bbox1, bbox2)
        self.assertNotEqual(bbox1, bbox3)
        self.assertNotEqual(bbox1, "not a bbox")

    def test_to_json(self):
        bbox = BBox(10, 20, 30, 50)
        json_data = bbox.to_json()
        self.assertEqual(json_data["x0"], 10)
        self.assertEqual(json_data["y0"], 20)
        self.assertEqual(json_data["x1"], 30)
        self.assertEqual(json_data["y1"], 50)

    def test_from_json(self):
        json_data = {"x0": 10, "y0": 20, "x1": 30, "y1": 50}
        bbox = BBox.from_json(json_data)
        self.assertEqual(bbox.x0, 10)
        self.assertEqual(bbox.y0, 20)
        self.assertEqual(bbox.x1, 30)
        self.assertEqual(bbox.y1, 50)

    def test_overlaps(self):
        bbox1 = BBox(10, 20, 30, 50)
        bbox2 = BBox(15, 25, 35, 55)
        self.assertTrue(bbox1.overlaps(bbox2))

        bbox3 = BBox(40, 60, 60, 80)
        self.assertFalse(bbox1.overlaps(bbox3))

        bbox4 = BBox(5, 10, 25, 45)
        self.assertTrue(bbox1.overlaps(bbox4))


class TestChar(TestCase):
    def setUp(self):
        self.char = Char("A", "Arial-Bold", 12, BBox(10, 20, 20, 32))

    def test_bbox(self):
        bbox = self.char.bbox
        self.assertEqual(bbox.x0, 10)
        self.assertEqual(bbox.y0, 20)
        self.assertEqual(bbox.x1, 20)
        self.assertEqual(bbox.y1, 32)

    def test_width(self):
        self.assertEqual(self.char.width, 10)

    def test_height(self):
        self.assertEqual(self.char.height, 12)

    def test_bold(self):
        self.assertTrue(self.char.bold)
        char2 = Char("B", "Arial", 12, BBox(10, 20, 20, 32))
        self.assertFalse(char2.bold)

    def test_italic(self):
        char1 = Char("A", "Arial-Italic", 12, BBox(10, 20, 20, 32))
        self.assertTrue(char1.italic)
        char2 = Char("B", "Arial-Oblique", 12, BBox(10, 20, 20, 32))
        self.assertTrue(char2.italic)
        char3 = Char("C", "Arial", 12, BBox(10, 20, 20, 32))
        self.assertFalse(char3.italic)

    def test_to_json(self):
        expected = {
            "text": "A",
            "fontname": "Arial-Bold",
            "size": 12,
            "bbox": {
                "x0": 10,
                "y0": 20,
                "x1": 20,
                "y1": 32,
            },
        }
        self.assertEqual(self.char.to_json(), expected)

    def test_from_json(self):
        json_data = {
            "text": "A",
            "fontname": "Arial-Bold",
            "size": 12,
            "bbox": {
                "x0": 10,
                "y0": 20,
                "x1": 20,
                "y1": 32,
            },
        }
        char = Char.from_json(json_data)
        self.assertEqual(char.text, "A")
        self.assertEqual(char.fontname, "Arial-Bold")
        self.assertEqual(char.size, 12)
        self.assertEqual(char.bbox.x0, 10)
        self.assertEqual(char.bbox.y0, 20)
        self.assertEqual(char.bbox.x1, 20)
        self.assertEqual(char.bbox.y1, 32)

    def test_equality(self):
        char1 = Char("A", "Arial-Bold", 12, BBox(10, 20, 20, 32))
        char2 = Char("A", "Arial-Bold", 12, BBox(10, 20, 20, 32))
        char3 = Char("B", "Arial-Bold", 12, BBox(10, 20, 20, 32))

        self.assertEqual(char1, char2)
        self.assertNotEqual(char1, char3)
        self.assertNotEqual(char1, "not a char")


class TestRun(TestCase):
    def setUp(self):
        self.chars = [
            Char("H", "Arial", 12, BBox(0, 0, 10, 12)),
            Char("i", "Arial", 12, BBox(10, 0, 15, 12)),
        ]
        self.run = Run(fontname="Arial", size=12, chars=self.chars)

    def test_text(self):
        self.assertEqual(self.run.text, "Hi")

    def test_append_char(self):
        new_char = Char("!", "Arial", 12, BBox(15, 0, 20, 12))
        self.run.append_char(new_char)
        self.assertEqual(self.run.text, "Hi!")

    def test_add_runs(self):
        other_chars = [Char("!", "Arial", 12, BBox(15, 0, 20, 12))]
        other_run = Run(fontname="Arial", size=12, chars=other_chars)
        combined = self.run + other_run
        self.assertEqual(combined.text, "Hi!")

    def test_from_text(self):
        run = Run.from_text("Test", "Arial", 12)
        self.assertEqual(run.text, "Test")
        self.assertEqual(run.fontname, "Arial")
        self.assertEqual(run.size, 12)

    def test_from_chars(self):
        run = Run.from_chars(self.chars)
        self.assertEqual(run.text, "Hi")
        self.assertEqual(run.fontname, "Arial")
        self.assertEqual(run.size, 12)

    def test_is_bold(self):
        run_bold = Run("Arial-Bold", 12, "Test")
        self.assertTrue(run_bold.is_bold())
        run_normal = Run("Arial", 12, "Test")
        self.assertFalse(run_normal.is_bold())

    def test_is_italic(self):
        run_italic = Run("Arial-Italic", 12, "Test")
        self.assertTrue(run_italic.is_italic())
        run_oblique = Run("Arial-Oblique", 12, "Test")
        self.assertTrue(run_oblique.is_italic())
        run_normal = Run("Arial", 12, "Test")
        self.assertFalse(run_normal.is_italic())

    def test_from_json(self):
        json_data = {
            "fontname": "Arial",
            "size": 12,
            "text": "Hi",
            "chars": [
                {
                    "text": "H",
                    "fontname": "Arial",
                    "size": 12,
                    "bbox": {
                        "x0": 0,
                        "y0": 0,
                        "x1": 10,
                        "y1": 12,
                    },
                },
                {
                    "text": "i",
                    "fontname": "Arial",
                    "size": 12,
                    "bbox": {
                        "x0": 10,
                        "y0": 0,
                        "x1": 15,
                        "y1": 12,
                    },
                },
            ],
        }
        run = Run.from_json(json_data)
        self.assertEqual(run.text, "Hi")
        self.assertEqual(run.fontname, "Arial")
        self.assertEqual(run.size, 12)
        self.assertEqual(len(run.chars), 2)
        self.assertEqual(run.chars[0].text, "H")
        self.assertEqual(run.chars[1].text, "i")

    def test_equality(self):
        run1 = Run.from_text("Test", "Arial", 12)
        run2 = Run.from_text("Test", "Arial", 12)
        run3 = Run.from_text("Different", "Arial", 12)

        self.assertEqual(run1, run2)
        self.assertNotEqual(run1, run3)
        self.assertNotEqual(run1, "not a run")


class TestLine(TestCase):
    def setUp(self):
        self.line = Line(BBox(0, 0, 100, 20))
        run1 = Run.from_text("Hello", "Arial", 12)
        run2 = Run.from_text(" World", "Arial-Bold", 12)
        self.line.add_run(run1)
        self.line.add_run(run2)

    def test_bbox(self):
        bbox = self.line.bbox
        self.assertEqual(bbox.x0, 0)
        self.assertEqual(bbox.y0, 0)
        self.assertEqual(bbox.x1, 100)
        self.assertEqual(bbox.y1, 20)

    def test_text(self):
        self.assertEqual(self.line.text, "Hello World")

    def test_get_all_chars(self):
        chars = self.line.get_all_chars()
        self.assertEqual(len(chars), 11)
        self.assertEqual("".join(c.text for c in chars), "Hello World")

    def test_to_json(self):
        json_data = self.line.to_json()
        self.assertEqual(json_data["bbox"]["x0"], 0)
        self.assertEqual(json_data["bbox"]["y0"], 0)
        self.assertEqual(json_data["bbox"]["x1"], 100)
        self.assertEqual(json_data["bbox"]["y1"], 20)
        self.assertEqual(json_data["text"], "Hello World")

    def test_from_json(self):
        json_data = {
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 100,
                "y1": 20,
            },
            "text": "Hello World",
            "runs": [
                {
                    "fontname": "Arial",
                    "size": 12,
                    "text": "Hello",
                    "chars": [
                        {
                            "text": char,
                            "fontname": "Arial",
                            "size": 12,
                            "bbox": {
                                "x0": i * 10,
                                "y0": 0,
                                "x1": (i + 1) * 10,
                                "y1": 12,
                            },
                        }
                        for i, char in enumerate("Hello")
                    ],
                },
                {
                    "fontname": "Arial-Bold",
                    "size": 12,
                    "text": " World",
                    "chars": [
                        {
                            "text": char,
                            "fontname": "Arial-Bold",
                            "size": 12,
                            "bbox": {
                                "x0": (i + 5) * 10,
                                "y0": 0,
                                "x1": (i + 6) * 10,
                                "y1": 12,
                            },
                        }
                        for i, char in enumerate(" World")
                    ],
                },
            ],
        }
        line = Line.from_json(json_data)
        self.assertEqual(line.text, "Hello World")
        self.assertEqual(len(line.runs), 2)
        self.assertEqual(line.runs[0].text, "Hello")
        self.assertEqual(line.runs[1].text, " World")

    def test_equality(self):
        line1 = Line(BBox(0, 0, 100, 20))
        line1.add_run(Run.from_text("Test", "Arial", 12))

        line2 = Line(BBox(0, 0, 100, 20))
        line2.add_run(Run.from_text("Test", "Arial", 12))

        line3 = Line(BBox(0, 0, 100, 20))
        line3.add_run(Run.from_text("Different", "Arial", 12))

        self.assertEqual(line1, line2)
        self.assertNotEqual(line1, line3)
        self.assertNotEqual(line1, "not a line")


class TestBox(TestCase):
    def setUp(self):
        self.text1 = "First line"
        self.text2 = "Second line"
        self.box = Box(BBox(0, 0, 200, 100))
        line1 = Line(BBox(0, 0, 200, 20))
        line1.add_run(Run.from_text(self.text1, "Arial", 12))
        line2 = Line(BBox(0, 20, 200, 40))
        line2.add_run(Run.from_text(self.text2, "Arial", 12))
        self.box.add_line(line1)
        self.box.add_line(line2)

        self.image_bbox = BBox(50, 50, 150, 150)
        self.image = Image(
            bbox=self.image_bbox, stream=b"test", name="test.jpg", mimetype="image/jpeg"
        )
        self.box.add_image(self.image)

    def test_bbox(self):
        bbox = self.box.bbox
        self.assertEqual(bbox.x0, 0)
        self.assertEqual(bbox.y0, 0)
        self.assertEqual(bbox.x1, 200)
        self.assertEqual(bbox.y1, 100)

    def test_text(self):
        self.assertEqual(self.box.text, "First line\nSecond line")

    def test_get_all_lines(self):
        lines = self.box.get_all_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, self.text1)
        self.assertEqual(lines[1].text, self.text2)

    def test_get_all_chars(self):
        chars = self.box.get_all_chars()
        self.assertEqual(len(chars), len(self.text1) + len(self.text2))

    def test_get_all_images(self):
        images = self.box.get_all_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], self.image)

    def test_add_image_validation(self):
        with self.assertRaises(ValueError):
            self.box.add_image("not an image")

    def test_to_json(self):
        json_data = self.box.to_json()
        self.assertEqual(json_data["bbox"]["x0"], 0)
        self.assertEqual(json_data["bbox"]["y0"], 0)
        self.assertEqual(json_data["bbox"]["x1"], 200)
        self.assertEqual(json_data["bbox"]["y1"], 100)
        self.assertEqual(json_data["text"], "First line\nSecond line")
        self.assertEqual(len(json_data["elements"]), 3)  # 2 lines + 1 image

        line_elements = [el for el in json_data["elements"] if el["type"] == "line"]
        self.assertEqual(len(line_elements), 2)
        self.assertEqual(line_elements[0]["text"], "First line")
        self.assertEqual(line_elements[1]["text"], "Second line")

        image_elements = [el for el in json_data["elements"] if el["type"] == "image"]
        self.assertEqual(len(image_elements), 1)
        self.assertEqual(image_elements[0]["bbox"], self.image_bbox.to_json())
        self.assertEqual(image_elements[0]["name"], "test.jpg")

    def test_from_json(self):
        json_data = {
            "bbox": {
                "x0": 0,
                "y0": 0,
                "x1": 200,
                "y1": 100,
            },
            "text": "First line\nSecond line",
            "elements": [
                {
                    "type": "line",
                    "bbox": {
                        "x0": 0,
                        "y0": 0,
                        "x1": 200,
                        "y1": 20,
                    },
                    "text": "First line",
                    "runs": [
                        {
                            "fontname": "Arial",
                            "size": 12,
                            "text": "First line",
                            "chars": [
                                {
                                    "text": char,
                                    "fontname": "Arial",
                                    "size": 12,
                                    "bbox": {
                                        "x0": i * 10,
                                        "y0": 0,
                                        "x1": (i + 1) * 10,
                                        "y1": 12,
                                    },
                                }
                                for i, char in enumerate("First line")
                            ],
                        }
                    ],
                },
                {
                    "type": "line",
                    "bbox": {
                        "x0": 0,
                        "y0": 20,
                        "x1": 200,
                        "y1": 40,
                    },
                    "text": "Second line",
                    "runs": [
                        {
                            "fontname": "Arial",
                            "size": 12,
                            "text": "Second line",
                            "chars": [
                                {
                                    "text": char,
                                    "fontname": "Arial",
                                    "size": 12,
                                    "bbox": {
                                        "x0": i * 10,
                                        "y0": 20,
                                        "x1": (i + 1) * 10,
                                        "y1": 32,
                                    },
                                }
                                for i, char in enumerate("Second line")
                            ],
                        }
                    ],
                },
                {
                    "type": "image",
                    "bbox": {"x0": 50, "y0": 50, "x1": 150, "y1": 150},
                    "name": "test.jpg",
                    "mimetype": "image/jpeg",
                    "stream": b"test",
                },
            ],
        }
        box = Box.from_json(json_data)
        self.assertEqual(box.text, "First line\nSecond line")

        lines = box.get_all_lines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "First line")
        self.assertEqual(lines[1].text, "Second line")

        images = box.get_all_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].name, "test.jpg")
        self.assertEqual(
            images[0].bbox.to_json(), {"x0": 50, "y0": 50, "x1": 150, "y1": 150}
        )

    def test_equality(self):
        box1 = Box(BBox(0, 0, 200, 100))
        line1 = Line(BBox(0, 0, 200, 20))
        line1.add_run(Run.from_text("Test", "Arial", 12))
        box1.add_line(line1)
        image1 = Image(
            bbox=BBox(50, 50, 150, 150),
            stream=b"test",
            name="test.jpg",
            mimetype="image/jpeg",
        )
        box1.add_image(image1)

        box2 = Box(BBox(0, 0, 200, 100))
        line2 = Line(BBox(0, 0, 200, 20))
        line2.add_run(Run.from_text("Test", "Arial", 12))
        box2.add_line(line2)
        image2 = Image(
            bbox=BBox(50, 50, 150, 150),
            stream=b"test",
            name="test.jpg",
            mimetype="image/jpeg",
        )
        box2.add_image(image2)

        box3 = Box(BBox(0, 0, 200, 100))
        line3 = Line(BBox(0, 0, 200, 20))
        line3.add_run(Run.from_text("Different", "Arial", 12))
        box3.add_line(line3)
        image3 = Image(
            bbox=BBox(60, 60, 160, 160),
            stream=b"test2",
            name="test2.jpg",
            mimetype="image/jpeg",
        )
        box3.add_image(image3)

        self.assertEqual(box1, box2)
        self.assertNotEqual(box1, box3)
        self.assertNotEqual(box1, "not a box")

    def test_merge(self):
        # Create two boxes with different content
        box1 = Box(BBox(0, 0, 100, 50))
        line1 = Line(BBox(0, 0, 100, 20))
        line1.add_run(Run.from_text("Box 1 text", "Arial", 12))
        box1.add_line(line1)
        image1 = Image(BBox(0, 20, 50, 50), b"test1", "img1.jpg", "image/jpeg")
        box1.add_image(image1)

        box2 = Box(BBox(50, 25, 150, 75))
        line2 = Line(BBox(50, 25, 150, 45))
        line2.add_run(Run.from_text("Box 2 text", "Arial", 12))
        box2.add_line(line2)
        image2 = Image(BBox(50, 45, 100, 75), b"test2", "img2.jpg", "image/jpeg")
        box2.add_image(image2)

        # Merge boxes
        merged = box1.merge(box2)

        # Test merged bbox
        self.assertEqual(merged.bbox.x0, 0)
        self.assertEqual(merged.bbox.y0, 0)
        self.assertEqual(merged.bbox.x1, 150)
        self.assertEqual(merged.bbox.y1, 75)

        # Test merged content
        self.assertEqual(len(merged.get_all_lines()), 2)
        self.assertEqual(len(merged.get_all_images()), 2)
        self.assertEqual(merged.get_all_lines()[0].text, "Box 1 text")
        self.assertEqual(merged.get_all_lines()[1].text, "Box 2 text")
        self.assertEqual(merged.get_all_images()[0].name, "img1.jpg")
        self.assertEqual(merged.get_all_images()[1].name, "img2.jpg")

        # Verify original boxes are unchanged
        self.assertEqual(len(box1.get_all_lines()), 1)
        self.assertEqual(len(box1.get_all_images()), 1)
        self.assertEqual(len(box2.get_all_lines()), 1)
        self.assertEqual(len(box2.get_all_images()), 1)


class TestPage(TestCase):
    def setUp(self):
        self.page = Page(595, 842)
        box1 = Box(BBox(50, 50, 545, 150))
        line1 = Line(BBox(50, 50, 545, 70))
        line1.add_run(Run.from_text("Page content", "Arial", 12))
        box1.add_line(line1)
        self.page.add_box(box1)

    def test_text(self):
        self.assertEqual(self.page.text, "Page content")

    def test_get_all_lines(self):
        lines = self.page.get_all_lines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, "Page content")

    def test_get_all_chars(self):
        chars = self.page.get_all_chars()
        self.assertEqual(len(chars), 12)

    def test_to_json(self):
        json_data = self.page.to_json()
        self.assertEqual(json_data["width"], 595)
        self.assertEqual(json_data["height"], 842)
        self.assertEqual(json_data["text"], "Page content")

    def test_append_run(self):
        new_run = Run.from_text("appended", "Arial", 12)
        self.page.append_run(new_run)
        self.assertEqual(self.page.text, "Page content\nappended")

    def test_from_json(self):
        json_data = {
            "width": 595,
            "height": 842,
            "text": "Page content",
            "boxes": [
                {
                    "bbox": {
                        "x0": 50,
                        "y0": 50,
                        "x1": 545,
                        "y1": 150,
                    },
                    "text": "Page content",
                    "elements": [
                        {
                            "type": "line",
                            "bbox": {
                                "x0": 50,
                                "y0": 50,
                                "x1": 545,
                                "y1": 70,
                            },
                            "text": "Page content",
                            "runs": [
                                {
                                    "fontname": "Arial",
                                    "size": 12,
                                    "text": "Page content",
                                    "chars": [
                                        {
                                            "text": char,
                                            "fontname": "Arial",
                                            "size": 12,
                                            "bbox": {
                                                "x0": 50 + i * 10,
                                                "y0": 50,
                                                "x1": 50 + (i + 1) * 10,
                                                "y1": 62,
                                            },
                                        }
                                        for i, char in enumerate("Page content")
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        page = Page.from_json(json_data)
        self.assertEqual(page.width, 595)
        self.assertEqual(page.height, 842)
        self.assertEqual(page.text, "Page content")
        self.assertEqual(len(page.boxes), 1)
        self.assertEqual(len(page.get_all_lines()), 1)

    def test_equality(self):
        page1 = Page(595, 842)
        box1 = Box(BBox(50, 50, 545, 150))
        line1 = Line(BBox(50, 50, 545, 70))
        line1.add_run(Run.from_text("Test", "Arial", 12))
        box1.add_line(line1)
        page1.add_box(box1)

        page2 = Page(595, 842)
        box2 = Box(BBox(50, 50, 545, 150))
        line2 = Line(BBox(50, 50, 545, 70))
        line2.add_run(Run.from_text("Test", "Arial", 12))
        box2.add_line(line2)
        page2.add_box(box2)

        page3 = Page(595, 842)
        box3 = Box(BBox(50, 50, 545, 150))
        line3 = Line(BBox(50, 50, 545, 70))
        line3.add_run(Run.from_text("Different", "Arial", 12))
        box3.add_line(line3)
        page3.add_box(box3)

        self.assertEqual(page1, page2)
        self.assertNotEqual(page1, page3)
        self.assertNotEqual(page1, "not a page")


class TestDocument(TestCase):
    def setUp(self):
        self.document = Document()
        page1 = Page(595, 842)
        box1 = Box(BBox(50, 50, 545, 150))
        line1 = Line(BBox(50, 50, 545, 70))
        line1.add_run(Run.from_text("Page 1", "Arial", 12))
        box1.add_line(line1)
        page1.add_box(box1)

        page2 = Page(595, 842)
        box2 = Box(BBox(50, 50, 545, 150))
        line2 = Line(BBox(50, 50, 545, 70))
        line2.add_run(Run.from_text("Page 2", "Arial", 12))
        box2.add_line(line2)
        page2.add_box(box2)

        self.document.add_page(page1)
        self.document.add_page(page2)

    def test_text(self):
        self.assertEqual(self.document.text, "Page 1\nPage 2")

    def test_get_all_pages(self):
        pages = self.document.get_all_pages()
        self.assertEqual(len(pages), 2)

    def test_get_all_boxes(self):
        boxes = self.document.get_all_boxes()
        self.assertEqual(len(boxes), 2)

    def test_get_all_lines(self):
        lines = self.document.get_all_lines()
        self.assertEqual(len(lines), 2)

    def test_get_all_chars(self):
        chars = self.document.get_all_chars()
        self.assertEqual(len(chars), 12)

    def test_get_all_runs(self):
        runs = self.document.get_all_runs(merge=False)
        self.assertEqual(len(runs), 2)

    def test_get_all_runs_merged(self):
        runs = self.document.get_all_runs(merge=True)
        self.assertEqual(len(runs), 1)

    def test_to_json(self):
        json_data = self.document.to_json()
        self.assertEqual(json_data["text"], "Page 1\nPage 2")
        self.assertEqual(len(json_data["pages"]), 2)

    def test_from_json(self):
        json_data = {
            "text": "Page 1\nPage 2",
            "pages": [
                {
                    "width": 595,
                    "height": 842,
                    "text": "Page 1",
                    "boxes": [
                        {
                            "bbox": {
                                "x0": 50,
                                "y0": 50,
                                "x1": 545,
                                "y1": 150,
                            },
                            "text": "Page 1",
                            "elements": [
                                {
                                    "type": "line",
                                    "bbox": {
                                        "x0": 50,
                                        "y0": 50,
                                        "x1": 545,
                                        "y1": 70,
                                    },
                                    "text": "Page 1",
                                    "runs": [
                                        {
                                            "fontname": "Arial",
                                            "size": 12,
                                            "text": "Page 1",
                                            "chars": [
                                                {
                                                    "text": char,
                                                    "fontname": "Arial",
                                                    "size": 12,
                                                    "bbox": {
                                                        "x0": 50 + i * 10,
                                                        "y0": 50,
                                                        "x1": 50 + (i + 1) * 10,
                                                        "y1": 62,
                                                    },
                                                }
                                                for i, char in enumerate("Page 1")
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "width": 595,
                    "height": 842,
                    "text": "Page 2",
                    "boxes": [
                        {
                            "bbox": {
                                "x0": 50,
                                "y0": 50,
                                "x1": 545,
                                "y1": 150,
                            },
                            "text": "Page 2",
                            "elements": [
                                {
                                    "type": "line",
                                    "bbox": {
                                        "x0": 50,
                                        "y0": 50,
                                        "x1": 545,
                                        "y1": 70,
                                    },
                                    "text": "Page 2",
                                    "runs": [
                                        {
                                            "fontname": "Arial",
                                            "size": 12,
                                            "text": "Page 2",
                                            "chars": [
                                                {
                                                    "text": char,
                                                    "fontname": "Arial",
                                                    "size": 12,
                                                    "bbox": {
                                                        "x0": 50 + i * 10,
                                                        "y0": 50,
                                                        "x1": 50 + (i + 1) * 10,
                                                        "y1": 62,
                                                    },
                                                }
                                                for i, char in enumerate("Page 2")
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ],
        }
        doc = Document.from_json(json_data)
        self.assertEqual(doc.text, "Page 1\nPage 2")
        self.assertEqual(len(doc.pages), 2)
        self.assertEqual(doc.pages[0].text, "Page 1")
        self.assertEqual(doc.pages[1].text, "Page 2")
        self.assertEqual(len(doc.get_all_boxes()), 2)
        self.assertEqual(len(doc.get_all_lines()), 2)
        self.assertEqual(len(doc.get_all_runs(merge=False)), 2)
        self.assertEqual(len(doc.get_all_runs(merge=True)), 1)

    def test_equality(self):
        doc1 = Document()
        page1 = Page(595, 842)
        box1 = Box(BBox(50, 50, 545, 150))
        line1 = Line(BBox(50, 50, 545, 70))
        line1.add_run(Run.from_text("Test", "Arial", 12))
        box1.add_line(line1)
        page1.add_box(box1)
        doc1.add_page(page1)

        doc2 = Document()
        page2 = Page(595, 842)
        box2 = Box(BBox(50, 50, 545, 150))
        line2 = Line(BBox(50, 50, 545, 70))
        line2.add_run(Run.from_text("Test", "Arial", 12))
        box2.add_line(line2)
        page2.add_box(box2)
        doc2.add_page(page2)

        doc3 = Document()
        page3 = Page(595, 842)
        box3 = Box(BBox(50, 50, 545, 150))
        line3 = Line(BBox(50, 50, 545, 70))
        line3.add_run(Run.from_text("Different", "Arial", 12))
        box3.add_line(line3)
        page3.add_box(box3)
        doc3.add_page(page3)

        self.assertEqual(doc1, doc2)
        self.assertNotEqual(doc1, doc3)
        self.assertNotEqual(doc1, "not a document")


class TestImage(TestCase):
    def setUp(self):
        self.bbox = BBox(10, 20, 100, 150)
        self.image = Image(
            bbox=self.bbox, stream=b"test", name="test.jpg", mimetype="image/jpeg"
        )

    def test_bbox(self):
        bbox = self.image.bbox
        self.assertEqual(bbox.x0, 10)
        self.assertEqual(bbox.y0, 20)
        self.assertEqual(bbox.x1, 100)
        self.assertEqual(bbox.y1, 150)
        self.assertEqual(bbox, self.bbox)

    def test_init(self):
        self.assertEqual(self.image.bbox, self.bbox)
        self.assertEqual(self.image.stream, b"test")
        self.assertEqual(self.image.name, "test.jpg")
        self.assertEqual(self.image.mimetype, "image/jpeg")

    def test_to_json(self):
        json_data = self.image.to_json()
        self.assertEqual(json_data["bbox"], self.bbox.to_json())
        self.assertEqual(json_data["stream"], b"test")
        self.assertEqual(json_data["name"], "test.jpg")
        self.assertEqual(json_data["mimetype"], "image/jpeg")

    def test_from_json(self):
        json_data = {
            "bbox": {"x0": 10, "y0": 20, "x1": 100, "y1": 150},
            "stream": b"test",
            "name": "test.jpg",
            "mimetype": "image/jpeg",
        }
        image = Image.from_json(json_data)
        self.assertEqual(image.bbox, self.bbox)
        self.assertEqual(image.stream, b"test")
        self.assertEqual(image.name, "test.jpg")
        self.assertEqual(image.mimetype, "image/jpeg")

    def test_repr(self):
        self.assertEqual(repr(self.image), f"Image(name=test.jpg, bbox={self.bbox})")

    def test_equality(self):
        image1 = Image(
            bbox=BBox(10, 20, 100, 150),
            stream=b"test",
            name="test.jpg",
            mimetype="image/jpeg",
        )
        image2 = Image(
            bbox=BBox(10, 20, 100, 150),
            stream=b"test",
            name="test.jpg",
            mimetype="image/jpeg",
        )
        image3 = Image(
            bbox=BBox(20, 30, 110, 160),
            stream=b"different",
            name="different.jpg",
            mimetype="image/png",
        )

        self.assertEqual(image1, image2)
        self.assertNotEqual(image1, image3)
        self.assertNotEqual(image1, "not an image")


if __name__ == "__main__":
    unittest.main()
