import os
import tempfile
from pathlib import Path
from unittest import TestCase

import fitz
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

from mitoolspro.document.document_structure import BBox, Box, Document
from mitoolspro.document.document_structure import Image as DocImage
from mitoolspro.document.from_pdf import extract_images_from_pdf, pdf_to_document


class TestFromPDF(TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for our test files
        cls.temp_dir = tempfile.mkdtemp()
        cls.pdf_path = Path(cls.temp_dir) / "test.pdf"
        cls.image_path = Path(cls.temp_dir) / "test_image.png"

        # Create a test image
        img = PILImage.new("RGB", (100, 100), color="red")
        img.save(cls.image_path)

        # Create a PDF with PyMuPDF directly for better image handling
        doc = fitz.open()
        page = doc.new_page()

        # Add text with specific fonts and sizes
        page.insert_text(
            (72, 72), "This is normal text", fontname="Helvetica", fontsize=12
        )
        page.insert_text(
            (72, 100), "This is bold text", fontname="Helvetica-Bold", fontsize=14
        )
        page.insert_text(
            (72, 128), "Special chars: áéíóú ñ", fontname="Helvetica", fontsize=12
        )
        page.insert_text(
            (72, 156),
            "First paragraph\nwith multiple lines",
            fontname="Helvetica",
            fontsize=12,
        )
        page.insert_text(
            (72, 184), "Second paragraph", fontname="Helvetica", fontsize=12
        )

        # Add text before first image
        page.insert_text(
            (72, 212), "Text before image", fontname="Helvetica", fontsize=12
        )

        # Read image data once
        with open(cls.image_path, "rb") as f:
            img_data = f.read()

        # Add first image
        img_rect1 = fitz.Rect(72, 240, 216, 384)  # 2 inch square
        page.insert_image(img_rect1, stream=img_data, keep_proportion=True)

        # Add text between images
        page.insert_text(
            (72, 412), "Text between images", fontname="Helvetica", fontsize=12
        )

        # Add second image with different size
        img_rect2 = fitz.Rect(72, 440, 144, 512)  # 1 inch square
        page.insert_image(img_rect2, stream=img_data, keep_proportion=True)

        # Add final text
        page.insert_text(
            (72, 540), "Text after image", fontname="Helvetica", fontsize=12
        )

        # Save the PDF with high quality settings
        doc.save(str(cls.pdf_path), garbage=4, deflate=True, clean=True)
        doc.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        os.unlink(cls.pdf_path)
        os.unlink(cls.image_path)
        os.rmdir(cls.temp_dir)

    def test_extract_images_from_pdf(self):
        # Test image extraction
        image_boxes = extract_images_from_pdf(self.pdf_path)

        # Debug information
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(0)

        # Print debug info about blocks
        blocks = page.get_text("dict")["blocks"]
        image_blocks = [b for b in blocks if b["type"] == 1]
        print(f"\nFound {len(image_blocks)} image blocks")

        # Print debug info about images
        images = page.get_images()
        print(f"Found {len(images)} raw images")

        # We should have one page with two images
        self.assertEqual(len(image_boxes), 1, "Expected 1 page")  # One page
        self.assertEqual(
            len(image_boxes[0]), 2, f"Expected 2 images, got {len(image_boxes[0])}"
        )  # Two images on the page

        # Test the extracted images
        for boxes in image_boxes:
            for box in boxes:
                # Verify box structure
                self.assertIsInstance(box, Box)
                self.assertEqual(len(box.get_all_images()), 1)

                # Verify image properties
                image = box.get_all_images()[0]
                self.assertIsInstance(image, DocImage)
                self.assertIsInstance(image.bbox, BBox)
                self.assertIsNotNone(image.stream)
                self.assertTrue(image.name.endswith(".png"))
                self.assertEqual(image.mimetype, "image/png")

        doc.close()

    def test_extract_images_from_reportlab_pdf(self):
        # Create a new PDF with reportlab
        reportlab_pdf_path = Path(self.temp_dir) / "reportlab_test.pdf"

        # Create the PDF
        c = canvas.Canvas(str(reportlab_pdf_path), pagesize=letter)
        page_height = letter[1]  # Need this for coordinate conversion

        # Add some text for context
        c.drawString(1 * inch, 10 * inch, "Test PDF with multiple images")

        # Add first image at specific coordinates (2x2 inches)
        first_img_y = 7 * inch
        c.drawImage(
            str(self.image_path), 1 * inch, first_img_y, width=2 * inch, height=2 * inch
        )

        # Add some text between images
        c.drawString(1 * inch, 6 * inch, "Text between images")

        # Add second image at different coordinates (1x1 inch)
        second_img_y = 3 * inch
        c.drawImage(
            str(self.image_path),
            1 * inch,
            second_img_y,
            width=1 * inch,
            height=1 * inch,
        )

        # Add final text
        c.drawString(1 * inch, 2 * inch, "Text after images")

        c.save()

        # Now test the image extraction
        try:
            image_boxes = extract_images_from_pdf(reportlab_pdf_path)

            # Debug information
            doc = fitz.open(reportlab_pdf_path)
            page = doc.load_page(0)

            # Print debug info
            blocks = page.get_text("dict")["blocks"]
            image_blocks = [b for b in blocks if b["type"] == 1]
            print(f"\nReportlab PDF: Found {len(image_blocks)} image blocks")

            images = page.get_images()
            print(f"Reportlab PDF: Found {len(images)} raw images")

            # Verify basic structure
            self.assertEqual(len(image_boxes), 1, "Expected 1 page")
            self.assertEqual(len(image_boxes[0]), 2, "Expected 2 images")

            # Test the extracted images
            boxes = image_boxes[0]  # Get boxes from first page

            # Find the larger and smaller images
            if (
                boxes[0].bbox.y1 - boxes[0].bbox.y0
                > boxes[1].bbox.y1 - boxes[1].bbox.y0
            ):
                larger_box = boxes[0]
                smaller_box = boxes[1]
            else:
                larger_box = boxes[1]
                smaller_box = boxes[0]

            # Verify larger image (2x2 inches)
            larger_height = larger_box.bbox.y1 - larger_box.bbox.y0
            larger_width = larger_box.bbox.x1 - larger_box.bbox.x0
            self.assertAlmostEqual(
                larger_height,
                2 * inch,
                delta=5,
                msg="Larger image height should be 2 inches",
            )
            self.assertAlmostEqual(
                larger_width,
                2 * inch,
                delta=5,
                msg="Larger image width should be 2 inches",
            )

            # Convert reportlab y-coordinate to PyMuPDF coordinate system
            expected_larger_y = page_height - (
                first_img_y + 2 * inch
            )  # Add height since y is from bottom
            self.assertAlmostEqual(
                larger_box.bbox.y0,
                expected_larger_y,
                delta=5,
                msg="Larger image should be at correct y-position",
            )

            # Verify smaller image (1x1 inch)
            smaller_height = smaller_box.bbox.y1 - smaller_box.bbox.y0
            smaller_width = smaller_box.bbox.x1 - smaller_box.bbox.x0
            self.assertAlmostEqual(
                smaller_height,
                1 * inch,
                delta=5,
                msg="Smaller image height should be 1 inch",
            )
            self.assertAlmostEqual(
                smaller_width,
                1 * inch,
                delta=5,
                msg="Smaller image width should be 1 inch",
            )

            # Convert reportlab y-coordinate to PyMuPDF coordinate system
            expected_smaller_y = page_height - (
                second_img_y + 1 * inch
            )  # Add height since y is from bottom
            self.assertAlmostEqual(
                smaller_box.bbox.y0,
                expected_smaller_y,
                delta=5,
                msg="Smaller image should be at correct y-position",
            )

            # Verify each image's basic properties
            for box in boxes:
                self.assertIsInstance(box, Box)
                self.assertEqual(len(box.get_all_images()), 1)

                image = box.get_all_images()[0]
                self.assertIsInstance(image, DocImage)
                self.assertIsInstance(image.bbox, BBox)
                self.assertIsNotNone(image.stream)
                self.assertTrue(image.name.endswith(".png"))
                self.assertEqual(image.mimetype, "image/png")

            doc.close()
        finally:
            # Clean up
            if reportlab_pdf_path.exists():
                os.unlink(reportlab_pdf_path)

    def test_pdf_to_document_structure(self):
        # Test full PDF conversion
        doc = pdf_to_document(self.pdf_path)

        # Basic document structure tests
        self.assertIsInstance(doc, Document)
        self.assertEqual(len(doc.pages), 1)  # Should have one page

        # Test page properties
        page = doc.pages[0]
        self.assertGreater(len(page.boxes), 0)

        # Test text content
        text = doc.text
        self.assertIn("This is normal text", text)
        self.assertIn("This is bold text", text)
        self.assertIn("Special chars:", text)
        self.assertIn("First paragraph", text)
        self.assertIn("Second paragraph", text)
        self.assertIn("Text before image", text)
        self.assertIn("Text after image", text)

        # Test text properties
        found_normal = False
        found_bold = False
        for box in page.boxes:
            for line in box.get_all_lines():
                for run in line.runs:
                    if "Helvetica" in run.fontname and run.size == 12:
                        found_normal = True
                    if "Helvetica-Bold" in run.fontname and run.size == 14:
                        found_bold = True

        self.assertTrue(found_normal, "Normal text style not found")
        self.assertTrue(found_bold, "Bold text style not found")

        # Test image content
        images_found = 0
        for box in page.boxes:
            images = box.get_all_images()
            images_found += len(images)
            for image in images:
                self.assertIsInstance(image.bbox, BBox)
                self.assertIsNotNone(image.stream)
                self.assertTrue(image.name.startswith("image_page0"))
                self.assertTrue(image.name.endswith(".png"))
                self.assertEqual(image.mimetype, "image/png")

        self.assertEqual(images_found, 2, "Expected 2 images in the document")

    def test_pdf_to_document_layout(self):
        # Test layout preservation
        doc = pdf_to_document(self.pdf_path)
        page = doc.pages[0]

        # Verify vertical ordering of elements
        boxes = page.boxes
        y_positions = [(box.bbox.y0, box.bbox.y1) for box in boxes]

        # Check that boxes don't overlap vertically (allowing for small tolerance)
        for i in range(len(y_positions) - 1):
            self.assertGreaterEqual(
                y_positions[i][0],
                y_positions[i + 1][1] - 1,  # 1 point tolerance
                "Boxes overlap vertically",
            )

        # Verify horizontal alignment
        for box in boxes:
            self.assertGreater(box.bbox.x0, 0)  # Should have left margin
            self.assertLess(box.bbox.x1, letter[0])  # Should be within page width

    def test_pdf_to_document_error_handling(self):
        # Test with non-existent file
        with self.assertRaises(ValueError):
            pdf_to_document(Path("nonexistent.pdf"))

        # Test with invalid file
        invalid_path = Path(self.temp_dir) / "invalid.pdf"
        with open(invalid_path, "w") as f:
            f.write("Not a PDF file")

        with self.assertRaises(ValueError):
            pdf_to_document(invalid_path)

        os.unlink(invalid_path)

    def test_extract_images_error_handling(self):
        # Test with non-existent file
        with self.assertRaises(Exception):
            extract_images_from_pdf(Path("nonexistent.pdf"))

        # Test with invalid file
        invalid_path = Path(self.temp_dir) / "invalid.pdf"
        with open(invalid_path, "w") as f:
            f.write("Not a PDF file")

        with self.assertRaises(Exception):
            extract_images_from_pdf(invalid_path)

        os.unlink(invalid_path)
