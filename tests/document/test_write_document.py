import os
import tempfile
from pathlib import Path
from unittest import TestCase

import fitz
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter

from mitoolspro.document.document_structure import (
    BBox,
    Box,
    Char,
    Document,
    Line,
    Page,
    Run,
)
from mitoolspro.document.document_structure import (
    Image as DocImage,
)
from mitoolspro.document.write_document import write_docx, write_pdf
from mitoolspro.exceptions import ArgumentValueError


class TestWriteDocument(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.pdf_output_path = Path(cls.temp_dir) / "output.pdf"
        cls.docx_output_path = Path(cls.temp_dir) / "output.docx"
        cls.image_path = Path(cls.temp_dir) / "test_image.png"

        img = PILImage.new("RGB", (100, 100), color="red")
        img.save(cls.image_path)

        with open(cls.image_path, "rb") as f:
            cls.image_data = f.read()

    @classmethod
    def tearDownClass(cls):
        for file in [cls.pdf_output_path, cls.docx_output_path, cls.image_path]:
            if file.exists():
                os.unlink(file)
        os.rmdir(cls.temp_dir)

    def setUp(self):
        self.doc = Document()
        self.page = Page(width=letter[0], height=letter[1])
        self.doc.pages.append(self.page)

    def test_write_pdf_empty_document(self):
        doc = Document()
        with self.assertRaises(ArgumentValueError):
            write_pdf(doc, self.pdf_output_path)

    def test_write_pdf_text_only(self):
        box = Box(BBox(72, 72, 400, 100))
        line = Line(BBox(72, 72, 400, 100))
        run = Run("Helvetica", 12)
        chars = []
        text = "Test text"
        x = 72
        for char in text:
            chars.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 72, x + 6, 84),
                )
            )
            x += 4
        run.chars = chars
        line.add_run(run)
        box.elements.append(line)
        self.page.boxes.append(box)

        write_pdf(self.doc, self.pdf_output_path)

        pdf_doc = fitz.open(self.pdf_output_path)
        page = pdf_doc[0]
        text = page.get_text()
        self.assertIn("Test text", text)
        pdf_doc.close()

    def test_write_pdf_with_images(self):
        box = Box(BBox(72, 72, 400, 100))
        line = Line(BBox(72, 72, 400, 100))
        run = Run("Helvetica", 12)
        chars = []
        text = "Text before image"
        x = 72
        for char in text:
            chars.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 72, x + 4, 84),
                )
            )
            x += 4
        run.chars = chars
        line.add_run(run)
        box.elements.append(line)
        self.page.boxes.append(box)

        image_box = Box(BBox(72, 120, 216, 264))
        image = DocImage(
            bbox=BBox(72, 120, 216, 264),
            stream=self.image_data,
            name="test_image.png",
            mimetype="image/png",
        )
        image_box.elements.append(image)
        self.page.boxes.append(image_box)

        write_pdf(self.doc, self.pdf_output_path)

        pdf_doc = fitz.open(self.pdf_output_path)
        page = pdf_doc[0]
        text = page.get_text()
        self.assertIn("Text before image", text)

        images = page.get_images()
        self.assertEqual(len(images), 1)
        pdf_doc.close()

    def test_write_pdf_multiple_pages(self):
        first_page = self.page
        second_page = Page(width=letter[0], height=letter[1])
        self.doc.pages.append(second_page)

        box1 = Box(BBox(72, 72, 400, 100))
        line1 = Line(BBox(72, 72, 400, 100))
        run1 = Run("Helvetica", 12)
        chars1 = []
        text1 = "First page text"
        x = 72
        for char in text1:
            chars1.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 72, x + 6, 84),
                )
            )
            x += 4
        run1.chars = chars1
        line1.add_run(run1)
        box1.elements.append(line1)
        first_page.boxes.append(box1)

        box2 = Box(BBox(72, 72, 400, 100))
        line2 = Line(BBox(72, 72, 400, 100))
        run2 = Run("Helvetica", 12)
        chars2 = []
        text2 = "Second page text"
        x = 72
        for char in text2:
            chars2.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 72, x + 6, 84),
                )
            )
        run2.chars = chars2
        line2.add_run(run2)
        box2.elements.append(line2)
        second_page.boxes.append(box2)

        write_pdf(self.doc, self.pdf_output_path)

        pdf_doc = fitz.open(self.pdf_output_path)
        self.assertEqual(len(pdf_doc), 2)
        self.assertIn("First page text", pdf_doc[0].get_text())
        self.assertIn("Second page text", pdf_doc[1].get_text())
        pdf_doc.close()

    def test_write_pdf_text_styles(self):
        box = Box(BBox(72, 72, 400, 120))

        line1 = Line(BBox(72, 72, 400, 100))
        run1 = Run("Helvetica", 12)
        chars1 = []
        text1 = "Normal text"
        x = 72
        for char in text1:
            chars1.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 72, x + 4, 84),
                )
            )
            x += 4
        run1.chars = chars1
        line1.add_run(run1)
        box.elements.append(line1)

        line2 = Line(BBox(72, 100, 400, 120))
        run2 = Run("Helvetica-Bold", 14)
        chars2 = []
        text2 = "Bold text"
        x = 72
        for char in text2:
            chars2.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=14,
                    bbox=BBox(x, 100, x + 7, 114),
                )
            )
            x += 4
        run2.chars = chars2
        line2.add_run(run2)
        box.elements.append(line2)

        line3 = Line(BBox(72, 120, 400, 140))
        run3 = Run("Helvetica-Oblique", 12)
        chars3 = []
        text3 = "Italic text"
        x = 72
        for char in text3:
            chars3.append(
                Char(
                    text=char,
                    fontname="Helvetica",
                    size=12,
                    bbox=BBox(x, 120, x + 4, 134),
                )
            )
            x += 4
        run3.chars = chars3
        line3.add_run(run3)
        box.elements.append(line3)

        self.page.boxes.append(box)

        write_pdf(self.doc, self.pdf_output_path)

        pdf_doc = fitz.open(self.pdf_output_path)
        page = pdf_doc[0]
        text = page.get_text()
        self.assertIn("Normal text", text)
        self.assertIn("Bold text", text)
        self.assertIn("Italic text", text)
        pdf_doc.close()

    def test_write_docx_empty_document(self):
        doc = Document()
        with self.assertRaises(ArgumentValueError):
            write_docx(doc, self.docx_output_path)

    def test_write_docx_text_only(self):
        box = Box(BBox(72, 72, 400, 100))
        line = Line(BBox(72, 72, 400, 100))
        run = Run("Helvetica", 12, "Test text")
        line.add_run(run)
        box.elements.append(line)
        self.page.boxes.append(box)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())

    def test_write_docx_with_images(self):
        box = Box(BBox(72, 72, 400, 100))
        line = Line(BBox(72, 72, 400, 100))
        run = Run("Helvetica", 12, "Text before image")
        line.add_run(run)
        box.elements.append(line)
        self.page.boxes.append(box)

        image_box = Box(BBox(72, 120, 216, 264))
        image = DocImage(
            bbox=BBox(72, 120, 216, 264),
            stream=self.image_data,
            name="test_image.png",
            mimetype="image/png",
        )
        image_box.elements.append(image)
        self.page.boxes.append(image_box)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())

    def test_write_docx_text_styles(self):
        box = Box(BBox(72, 72, 400, 120))

        line1 = Line(BBox(72, 72, 400, 100))
        run1 = Run("Helvetica", 12, "Normal text")
        line1.add_run(run1)
        box.elements.append(line1)

        line2 = Line(BBox(72, 100, 400, 120))
        run2 = Run("Helvetica-Bold", 14, "Bold text")
        line2.add_run(run2)
        box.elements.append(line2)

        line3 = Line(BBox(72, 120, 400, 140))
        run3 = Run("Helvetica-Oblique", 12, "Italic text")
        line3.add_run(run3)
        box.elements.append(line3)

        self.page.boxes.append(box)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())

    def test_write_docx_multiple_pages(self):
        first_page = self.page
        second_page = Page(width=letter[0], height=letter[1])
        self.doc.pages.append(second_page)

        box1 = Box(BBox(72, 72, 400, 100))
        line1 = Line(BBox(72, 72, 400, 100))
        run1 = Run("Helvetica", 12, "First page text")
        line1.add_run(run1)
        box1.elements.append(line1)
        first_page.boxes.append(box1)

        box2 = Box(BBox(72, 72, 400, 100))
        line2 = Line(BBox(72, 72, 400, 100))
        run2 = Run("Helvetica", 12, "Second page text")
        line2.add_run(run2)
        box2.elements.append(line2)
        second_page.boxes.append(box2)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())

    def test_write_docx_special_characters(self):
        box = Box(BBox(72, 72, 400, 100))
        line = Line(BBox(72, 72, 400, 100))
        run = Run("Helvetica", 12, "Special chars: áéíóú ñ")
        line.add_run(run)
        box.elements.append(line)
        self.page.boxes.append(box)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())

    def test_write_docx_multiple_lines(self):
        box = Box(BBox(72, 72, 400, 120))

        line1 = Line(BBox(72, 72, 400, 100))
        run1 = Run("Helvetica", 12, "First line")
        line1.add_run(run1)
        box.elements.append(line1)

        line2 = Line(BBox(72, 100, 400, 120))
        run2 = Run("Helvetica", 12, "Second line")
        line2.add_run(run2)
        box.elements.append(line2)

        self.page.boxes.append(box)

        write_docx(self.doc, self.docx_output_path)
        self.assertTrue(self.docx_output_path.exists())
