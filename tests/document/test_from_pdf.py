import os
import tempfile
import unittest
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
        cls.whitespace_pdf_path = Path(cls.temp_dir) / "whitespace_test.pdf"

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

        # Create a PDF with various whitespace characters
        doc = fitz.open()
        page = doc.new_page()

        # Add text with various whitespace characters
        page.insert_text(
            (72, 72), "Text with\twhitespace", fontname="Helvetica", fontsize=12
        )
        page.insert_text(
            (72, 100), "Text with\r\nline breaks", fontname="Helvetica", fontsize=12
        )
        page.insert_text(
            (72, 128),
            "Text with\xa0non-breaking space",
            fontname="Helvetica",
            fontsize=12,
        )
        page.insert_text(
            (72, 156), "Text with multiple   spaces", fontname="Helvetica", fontsize=12
        )

        doc.save(str(cls.whitespace_pdf_path), garbage=4, deflate=True, clean=True)
        doc.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        os.unlink(cls.pdf_path)
        os.unlink(cls.image_path)
        os.unlink(cls.whitespace_pdf_path)
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
        # Test layout preservation for PyMuPDF PDF
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
                "Boxes overlap vertically in PyMuPDF PDF",
            )

        # Verify horizontal alignment
        for box in boxes:
            self.assertGreater(
                box.bbox.x0, 0, "Box should have left margin in PyMuPDF PDF"
            )
            self.assertLess(
                box.bbox.x1, letter[0], "Box should be within page width in PyMuPDF PDF"
            )

    def test_reportlab_pdf_to_document_layout(self):
        # Create a new PDF with reportlab
        reportlab_pdf_path = Path(self.temp_dir) / "reportlab_test.pdf"

        # Create the PDF with ReportLab
        c = canvas.Canvas(str(reportlab_pdf_path), pagesize=letter)
        page_height = letter[1]

        # Define image positions
        first_img_y = 7 * inch  # Position of larger image
        second_img_y = 3 * inch  # Position of smaller image

        # Add text and images with specific positions
        c.drawString(1 * inch, 10 * inch, "Test PDF with multiple images")
        c.drawImage(
            str(self.image_path), 1 * inch, first_img_y, width=2 * inch, height=2 * inch
        )
        c.drawString(1 * inch, 6 * inch, "Text between images")
        c.drawImage(
            str(self.image_path),
            1 * inch,
            second_img_y,
            width=1 * inch,
            height=1 * inch,
        )
        c.drawString(1 * inch, 2 * inch, "Text after images")
        c.save()

        try:
            # Test the ReportLab PDF layout
            doc = pdf_to_document(reportlab_pdf_path)
            page = doc.pages[0]
            boxes = page.boxes

            # Test vertical ordering
            y_positions = [(box.bbox.y0, box.bbox.y1) for box in boxes]
            for i in range(len(y_positions) - 1):
                self.assertGreaterEqual(
                    y_positions[i][0],
                    y_positions[i + 1][1] - 1,  # 1 point tolerance
                    "Boxes overlap vertically in ReportLab PDF",
                )

            # Test horizontal alignment
            for box in boxes:
                self.assertGreater(
                    box.bbox.x0, 0, "Box should have left margin in ReportLab PDF"
                )
                self.assertLess(
                    box.bbox.x1,
                    letter[0],
                    "Box should be within page width in ReportLab PDF",
                )

            # Test image content and sizes
            images = []
            for box in boxes:
                images.extend(box.get_all_images())

            self.assertEqual(len(images), 2, "Should have 2 images in ReportLab PDF")

            # Sort images by size (larger first)
            images.sort(
                key=lambda img: (img.bbox.y1 - img.bbox.y0)
                * (img.bbox.x1 - img.bbox.x0),
                reverse=True,
            )

            # Test larger image (2x2 inches)
            larger_img = images[0]
            self.assertAlmostEqual(
                larger_img.bbox.x1 - larger_img.bbox.x0,
                2 * inch,
                delta=5,
                msg="Larger image width should be 2 inches",
            )
            self.assertAlmostEqual(
                larger_img.bbox.y1 - larger_img.bbox.y0,
                2 * inch,
                delta=5,
                msg="Larger image height should be 2 inches",
            )

            # Test smaller image (1x1 inch)
            smaller_img = images[1]
            self.assertAlmostEqual(
                smaller_img.bbox.x1 - smaller_img.bbox.x0,
                1 * inch,
                delta=5,
                msg="Smaller image width should be 1 inch",
            )
            self.assertAlmostEqual(
                smaller_img.bbox.y1 - smaller_img.bbox.y0,
                1 * inch,
                delta=5,
                msg="Smaller image height should be 1 inch",
            )

            # Test vertical positioning
            expected_larger_y = page_height - (first_img_y + 2 * inch)
            expected_smaller_y = page_height - (second_img_y + 1 * inch)

            self.assertAlmostEqual(
                larger_img.bbox.y0,
                expected_larger_y,
                delta=5,
                msg="Larger image should be at correct y-position",
            )
            self.assertAlmostEqual(
                smaller_img.bbox.y0,
                expected_smaller_y,
                delta=5,
                msg="Smaller image should be at correct y-position",
            )

        finally:
            if reportlab_pdf_path.exists():
                os.unlink(reportlab_pdf_path)

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

    def test_whitespace_handling(self):
        # Test PDF with various whitespace characters
        doc = pdf_to_document(self.whitespace_pdf_path)

        # Verify document structure
        self.assertIsInstance(doc, Document)
        self.assertEqual(len(doc.pages), 1)

        # Get all text content
        text = doc.text

        # Verify whitespace is handled correctly
        self.assertIn("Text with:whitespace", text)  # Tab is converted to colon
        self.assertIn("Text with\n\nline breaks", text)
        self.assertIn("Text with non-breaking space", text)
        self.assertIn("Text with multiple   spaces", text)

        # Verify no errors occurred during processing
        for page in doc.pages:
            for box in page.boxes:
                for line in box.get_all_lines():
                    for run in line.runs:
                        for char in run.chars:
                            # Verify each character is valid
                            self.assertIsInstance(char.text, str)
                            self.assertTrue(
                                len(char.text) <= 1
                            )  # Each char should be a single character
                            if char.text.isspace():
                                self.assertEqual(
                                    len(char.text), 1
                                )  # Whitespace should be single character


class TestMultilingualPDFs(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.fitz_pdf_path = Path(cls.temp_dir) / "multilingual_fitz.pdf"
        cls.reportlab_pdf_path = Path(cls.temp_dir) / "multilingual_reportlab.pdf"
        cls.image_path = Path(cls.temp_dir) / "test_image.png"

        # Create a test image
        img = PILImage.new("RGB", (100, 100), color="red")
        img.save(cls.image_path)

        # Create multilingual text samples with proper font mapping
        cls.text_samples = {
            "Spanish": {
                "text": "¡Hola! ¿Cómo estás? Él es un niño. Ángel y Óscar son amigos. La Ñ es una letra especial.",
                "font": "Helvetica",
            },
            "French": {
                "text": "Bonjour! Comment allez-vous? L'été est chaud. Le café est délicieux. L'école est fermée.",
                "font": "Helvetica",
            },
            "German": {
                "text": "Guten Tag! Wie geht es Ihnen? Der Frühling ist schön. Das Wetter ist kalt. Die Sonne scheint.",
                "font": "Helvetica",
            },
            "Russian": {
                "text": "Привет! Как дела? Это хорошая книга. Я люблю читать. Добро пожаловать!",
                "font": "Times-Roman",  # Using Times-Roman for Cyrillic support
            },
            "Japanese": {
                "text": "こんにちは！お元気ですか？本を読むのが好きです。今日はいい天気です。",
                "font": "Times-Roman",  # Fallback to Times-Roman
            },
            "Chinese": {
                "text": "你好！最近怎么样？我喜欢读书。今天天气很好。欢迎光临！",
                "font": "Times-Roman",  # Fallback to Times-Roman
            },
            "Arabic": {
                "text": "مرحبا! كيف حالك؟ أنا أحب القراءة. الطقس جميل اليوم. أهلاً بك!",
                "font": "Times-Roman",  # Using Times-Roman for Arabic support
            },
            "Hebrew": {
                "text": "שלום! מה שלומך? אני אוהב לקרוא. מזג האוויר יפה היום. ברוכים הבאים!",
                "font": "Times-Roman",  # Using Times-Roman for Hebrew support
            },
        }

        # Create PDF with PyMuPDF
        doc = fitz.open()
        page = doc.new_page()
        y_pos = 72
        page_width = page.rect.width
        margin = 72  # 1 inch margin
        max_width = page_width - (2 * margin)  # Available width for text

        for language, sample in cls.text_samples.items():
            try:
                # Split text into lines and insert each line separately
                lines = sample["text"].split("\n")
                for line in lines:
                    if line.strip():  # Only insert non-empty lines
                        text = f"{language}: {line}"
                        # Calculate text width and split if needed
                        text_width = fitz.get_text_length(
                            text, fontname=sample["font"], fontsize=12
                        )
                        if text_width > max_width:
                            # Split text into multiple lines if it's too wide
                            words = text.split()
                            current_line = []
                            current_width = 0
                            for word in words:
                                word_width = fitz.get_text_length(
                                    word + " ", fontname=sample["font"], fontsize=12
                                )
                                if current_width + word_width > max_width:
                                    # Insert current line and start new line
                                    page.insert_text(
                                        (margin, y_pos),
                                        " ".join(current_line),
                                        fontname=sample["font"],
                                        fontsize=12,
                                    )
                                    y_pos += 20
                                    current_line = [word]
                                    current_width = word_width
                                else:
                                    current_line.append(word)
                                    current_width += word_width
                            # Insert the last line
                            if current_line:
                                page.insert_text(
                                    (margin, y_pos),
                                    " ".join(current_line),
                                    fontname=sample["font"],
                                    fontsize=12,
                                )
                                y_pos += 20
                        else:
                            # Insert text as is if it fits
                            page.insert_text(
                                (margin, y_pos),
                                text,
                                fontname=sample["font"],
                                fontsize=12,
                            )
                            y_pos += 20
                y_pos += 10  # Add extra space between language blocks
            except Exception as e:
                print(f"Warning: Could not insert {language} text: {e}")
        doc.save(str(cls.fitz_pdf_path), garbage=4, deflate=True, clean=True)
        doc.close()

        # Create PDF with ReportLab
        c = canvas.Canvas(str(cls.reportlab_pdf_path), pagesize=letter)
        y_pos = 10 * inch
        page_width = letter[0]
        margin = 72  # 1 inch margin
        max_width = page_width - (2 * margin)  # Available width for text

        for language, sample in cls.text_samples.items():
            try:
                c.setFont(sample["font"], 12)
                # Split text into lines and draw each line separately
                lines = sample["text"].split("\n")
                for line in lines:
                    if line.strip():  # Only draw non-empty lines
                        text = f"{language}: {line}"
                        # Use drawText with text wrapping
                        text_obj = c.beginText(margin, y_pos)
                        text_obj.setFont(sample["font"], 12)

                        # Split text into words and build lines
                        words = text.split()
                        current_line = []
                        current_width = 0

                        for word in words:
                            word_width = c.stringWidth(word + " ", sample["font"], 12)
                            if current_width + word_width > max_width:
                                # Draw current line and start new line
                                text_obj.textLine(" ".join(current_line))
                                current_line = [word]
                                current_width = word_width
                            else:
                                current_line.append(word)
                                current_width += word_width

                        # Draw the last line
                        if current_line:
                            text_obj.textLine(" ".join(current_line))

                        c.drawText(text_obj)
                        y_pos -= 0.3 * inch  # Reduced spacing between lines
                y_pos -= 0.2 * inch  # Add extra space between language blocks
            except Exception as e:
                print(f"Warning: Could not insert {language} text: {e}")
        c.save()

    @classmethod
    def tearDownClass(cls):
        for file in [cls.fitz_pdf_path, cls.reportlab_pdf_path, cls.image_path]:
            if file.exists():
                os.unlink(file)
        os.rmdir(cls.temp_dir)

    def test_multilingual_fitz_pdf(self):
        doc = pdf_to_document(self.fitz_pdf_path)
        self.assertIsInstance(doc, Document)
        self.assertEqual(len(doc.pages), 1)

        page = doc.pages[0]
        text = page.text.replace("\n", " ").replace("  ", " ")

        # Test Latin-based languages (these should work reliably)
        latin_languages = ["Spanish", "French", "German"]
        for language in latin_languages:
            sample = self.text_samples[language]
            # Normalize whitespace for comparison
            expected_text = f"{language}: {sample['text']}"
            self.assertIn(expected_text, text)

        # Test text properties
        for box in page.boxes:
            for line in box.get_all_lines():
                for run in line.runs:
                    self.assertIn(
                        run.fontname,
                        [sample["font"] for sample in self.text_samples.values()],
                    )
                    self.assertEqual(run.size, 12)

    def test_multilingual_reportlab_pdf(self):
        doc = pdf_to_document(self.reportlab_pdf_path)
        self.assertIsInstance(doc, Document)
        self.assertEqual(len(doc.pages), 1)

        page = doc.pages[0]
        text = page.text.replace("\n", " ").replace("  ", " ")

        # Test Latin-based languages (these should work reliably)
        latin_languages = ["Spanish", "French", "German"]
        for language in latin_languages:
            sample = self.text_samples[language]
            # Normalize whitespace for comparison
            expected_text = f"{language}: {sample['text']}"
            self.assertIn(expected_text, text)

        # Test text properties
        for box in page.boxes:
            for line in box.get_all_lines():
                for run in line.runs:
                    # Include all possible fonts that might be used
                    expected_fonts = [
                        "Helvetica",
                        "Times-Roman",
                        "ZapfDingbats",
                        "Courier",
                        "Symbol",
                        "Times-Bold",
                        "Times-Italic",
                        "Helvetica-Bold",
                        "Helvetica-Oblique",
                        "Courier-Bold",
                        "Courier-Oblique",
                    ]
                    self.assertIn(run.fontname, expected_fonts)
                    self.assertAlmostEqual(run.size, 12)

    def test_multilingual_text_operations(self):
        # Test text operations with multilingual content
        doc = pdf_to_document(self.fitz_pdf_path)
        page = doc.pages[0]

        # Test text concatenation
        all_text = ""
        for box in page.boxes:
            for line in box.get_all_lines():
                all_text += line.text + "\n"

        # Normalize whitespace for comparison
        all_text = all_text.replace("\n", " ").replace("  ", " ")

        # Verify Latin-based languages are present
        latin_languages = ["Spanish", "French", "German"]
        for language in latin_languages:
            sample = self.text_samples[language]
            expected_text = f"{language}: {sample['text']}"
            self.assertIn(expected_text, all_text)

        # Test character operations
        chars = page.get_all_chars()
        self.assertGreater(len(chars), 0)

        # Test run operations
        runs = page.get_all_runs(merge=False)
        self.assertGreater(len(runs), 0)

        # Test merged runs
        merged_runs = page.get_all_runs(merge=True)
        self.assertGreater(len(merged_runs), 0)

    def test_multilingual_layout(self):
        # Test layout preservation for both PDF types
        for pdf_path in [self.fitz_pdf_path, self.reportlab_pdf_path]:
            doc = pdf_to_document(pdf_path)
            page = doc.pages[0]
            boxes = page.boxes

            # Verify vertical ordering
            y_positions = [(box.bbox.y0, box.bbox.y1) for box in boxes]
            for i in range(len(y_positions) - 1):
                self.assertGreaterEqual(
                    y_positions[i][0],
                    y_positions[i + 1][1] - 1,  # 1 point tolerance
                    f"Boxes overlap vertically in {pdf_path.name}",
                )

            # Verify horizontal alignment
            for box in boxes:
                self.assertGreater(
                    box.bbox.x0, 0, f"Box should have left margin in {pdf_path.name}"
                )
                self.assertLess(
                    box.bbox.x1,
                    letter[0],
                    f"Box should be within page width in {pdf_path.name}",
                )

    def test_multilingual_json_serialization(self):
        # Test JSON serialization with multilingual content
        doc = pdf_to_document(self.fitz_pdf_path)
        json_data = doc.to_json()
        reconstructed = Document.from_json(json_data)
        reconstructed_text = reconstructed.text.replace("\n", " ").replace("  ", " ")

        # Verify text content for Latin-based languages
        latin_languages = ["Spanish", "French", "German"]
        for language in latin_languages:
            sample = self.text_samples[language]
            # Normalize whitespace for comparison
            expected_text = f"{language}: {sample['text']}"
            self.assertIn(expected_text, reconstructed_text)

        # Verify structure
        self.assertEqual(len(doc.pages), len(reconstructed.pages))
        self.assertEqual(len(doc.get_all_boxes()), len(reconstructed.get_all_boxes()))
        self.assertEqual(len(doc.get_all_lines()), len(reconstructed.get_all_lines()))
        self.assertEqual(
            len(doc.get_all_runs(merge=False)),
            len(reconstructed.get_all_runs(merge=False)),
        )
        self.assertEqual(
            len(doc.get_all_runs(merge=True)),
            len(reconstructed.get_all_runs(merge=True)),
        )


if __name__ == "__main__":
    unittest.main()
