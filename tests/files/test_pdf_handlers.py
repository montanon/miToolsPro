import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import PyPDF2

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.files.pdf_handlers import (
    extract_pdf_metadata,
    extract_pdf_title,
    pdf_to_markdown,
    pdf_to_markdown_file,
    set_folder_pdfs_titles_as_filenames,
    set_pdf_title_as_filename,
)


class TestExtractPdfMetadata(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.pdf_with_metadata = self.test_dir / "with_metadata.pdf"
        with open(self.pdf_with_metadata, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata(
                {
                    "/Title": "Test PDF",
                    "/Author": "Jane Doe",
                    "/Subject": "Testing Metadata",
                }
            )
            writer.write(f)
        self.pdf_without_metadata = self.test_dir / "without_metadata.pdf"
        with open(self.pdf_without_metadata, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.write(f)
        self.corrupted_pdf = self.test_dir / "corrupted.pdf"
        with open(self.corrupted_pdf, "wb") as f:
            f.write(b"%PDF-1.4\nINVALID CONTENT")
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_metadata_valid_pdf(self):
        result = extract_pdf_metadata(self.pdf_with_metadata)
        expected = {
            "Title": "Test PDF",
            "Author": "Jane Doe",
            "Subject": "Testing Metadata",
            "Producer": "mtp",
        }
        self.assertEqual(result, expected)

    def test_extract_metadata_empty_pdf(self):
        result = extract_pdf_metadata(self.pdf_without_metadata)
        self.assertEqual(result, {"Producer": "mtp"})

    def test_non_existing_file(self):
        non_existing_file = self.test_dir / "non_existent.pdf"
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(non_existing_file)

    def test_corrupted_pdf(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.corrupted_pdf)

    def test_non_pdf_file(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.non_pdf_file)

    def test_large_pdf_with_metadata(self):
        large_pdf = self.test_dir / "large_with_metadata.pdf"
        with open(large_pdf, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Large Test PDF"})
            for _ in range(100):
                writer.add_blank_page(width=210, height=297)
            writer.write(f)
        result = extract_pdf_metadata(large_pdf)
        self.assertEqual(result, {"Title": "Large Test PDF", "Producer": "mtp"})

    def test_unicode_metadata(self):
        unicode_pdf = self.test_dir / "unicode_metadata.pdf"
        with open(unicode_pdf, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Тестовый PDF"})  # Russian text
            writer.write(f)
        result = extract_pdf_metadata(unicode_pdf)
        self.assertEqual(result, {"Title": "Тестовый PDF", "Producer": "mtp"})


class TestExtractPdfTitle(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        with open(self.pdf_with_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Test PDF Title"})
            writer.write(f)
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        with open(self.pdf_without_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Author": "Jane Doe"})
            writer.write(f)
        self.corrupted_pdf = self.test_dir / "corrupted.pdf"
        with open(self.corrupted_pdf, "wb") as f:
            f.write(b"%PDF-1.4\nINVALID CONTENT")
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_title_from_valid_pdf(self):
        result = extract_pdf_title(self.pdf_with_title)
        self.assertEqual(result, "Test PDF Title")

    def test_extract_title_from_pdf_without_title(self):
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(self.pdf_without_title)
        self.assertIn("has no title in its metadata", str(context.exception))

    def test_non_existent_file(self):
        non_existing_file = self.test_dir / "non_existent.pdf"
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(non_existing_file)
        self.assertIn("is not a valid file path", str(context.exception))

    def test_extract_title_from_corrupted_pdf(self):
        with self.assertRaises(ArgumentValueError) as context:
            extract_pdf_title(self.corrupted_pdf)
        self.assertIn("Error reading PDF", str(context.exception))

    def test_extract_title_from_non_pdf_file(self):
        with self.assertRaises(ArgumentTypeError) as context:
            extract_pdf_title(self.non_pdf_file)
        self.assertIn("is not a valid PDF file", str(context.exception))


class TestSetPdfTitleAsFilename(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        with open(self.pdf_with_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Title": "Valid PDF Title"})
            writer.write(f)
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        with open(self.pdf_without_title, "wb") as f:
            writer = PyPDF2.PdfWriter()
            writer.add_metadata({"/Author": "Jane Doe"})
            writer.write(f)
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        with open(self.non_pdf_file, "w") as f:
            f.write("This is not a PDF file.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_pdf_with_title(self):
        set_pdf_title_as_filename(self.pdf_with_title)
        expected_filename = self.test_dir / "Valid_PDF_Title.pdf"
        self.assertTrue(expected_filename.exists())

    def test_pdf_without_title(self):
        with self.assertRaises(ArgumentValueError) as context:
            set_pdf_title_as_filename(self.pdf_without_title)
        self.assertIn("has no title in its metadata", str(context.exception))

    def test_non_pdf_file(self):
        with self.assertRaises(ArgumentTypeError) as context:
            set_pdf_title_as_filename(self.non_pdf_file)
        self.assertIn("is not a valid PDF file", str(context.exception))

    def test_rename_with_duplicate_filename(self):
        duplicate_pdf = self.test_dir / "Valid_PDF_Title.pdf"
        duplicate_pdf.touch()  # Create a duplicate file
        set_pdf_title_as_filename(self.pdf_with_title)
        expected_filename = self.test_dir / "Valid_PDF_Title_1.pdf"
        self.assertTrue(expected_filename.exists())

    def test_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Valid_PDF_Title.pdf"
        duplicate_pdf.touch()  # Create a duplicate file
        set_pdf_title_as_filename(self.pdf_with_title, overwrite=True)
        self.assertTrue(duplicate_pdf.exists())
        self.assertFalse(self.pdf_with_title.exists())  # Original should be renamed


class TestSetFolderPDFsTitlesAsFilenames(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.pdf_with_title = self.test_dir / "with_title.pdf"
        self.create_pdf_with_metadata(self.pdf_with_title, title="Test Title")
        self.pdf_without_title = self.test_dir / "without_title.pdf"
        self.create_pdf_with_metadata(self.pdf_without_title, title=None)
        self.non_pdf_file = self.test_dir / "not_a_pdf.txt"
        self.non_pdf_file.write_text("This is a text file.")

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def create_pdf_with_metadata(file_path: Path, title: str = None):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_blank_page(width=72, height=72)  # Add a blank page
        if title:
            pdf_writer.add_metadata({"/Title": title})
        with open(file_path, "wb") as f:
            pdf_writer.write(f)

    def test_valid_folder_with_pdfs(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        expected_file = self.test_dir / "Test_Title.pdf"
        self.assertTrue(expected_file.exists())

    def test_pdf_without_title(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)

    def test_skip_non_pdf_files(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        self.assertTrue(self.non_pdf_file.exists())  # Non-PDF file remains unchanged

    def test_invalid_folder_path(self):
        with self.assertRaises(ArgumentValueError):
            set_folder_pdfs_titles_as_filenames("./non_existent_folder")

    def test_attempt_mode(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir, attempt=True)
        old_file = self.test_dir / "with_title.pdf"
        new_file = self.test_dir / "Test_Title.pdf"
        self.assertTrue(old_file.exists())  # Ensure no renaming occurred
        self.assertFalse(new_file.exists())

    def test_dont_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Test_Title.pdf"
        self.create_pdf_with_metadata(duplicate_pdf, title="Test Title")
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=False)
        self.assertTrue((self.test_dir / "Test_Title_1.pdf").exists())
        self.assertFalse((self.test_dir / "Test_Title.pdf").exists())

    def test_overwrite_existing_file(self):
        duplicate_pdf = self.test_dir / "Test_Title.pdf"
        self.create_pdf_with_metadata(duplicate_pdf, title="Test Title")
        set_folder_pdfs_titles_as_filenames(self.test_dir, overwrite=True)
        self.assertTrue((self.test_dir / "Test_Title.pdf").exists())
        self.assertFalse((self.test_dir / "Test_Title_1.pdf").exists())

    def test_failure_handling_in_pdf_processing(self):
        corrupted_pdf = self.test_dir / "corrupted.pdf"
        corrupted_pdf.write_text("This is not a valid PDF.")
        set_folder_pdfs_titles_as_filenames(self.test_dir)


class TestPdfToMarkdown(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_pdf = self.test_dir / "test.pdf"

        # Create a PDF with actual content
        writer = PyPDF2.PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        writer.add_metadata({"/Title": "Test PDF"})
        with open(self.test_pdf, "wb") as f:
            writer.write(f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pdf_to_markdown_without_page_numbers(self):
        result = pdf_to_markdown(self.test_pdf, page_number=False)
        self.assertIsInstance(result, str)
        self.assertNotIn("Page 1 of 1", result)

    def test_pdf_to_markdown_with_page_numbers(self):
        result = pdf_to_markdown(self.test_pdf, page_number=True)
        self.assertIsInstance(result, str)

    def test_pdf_to_markdown_invalid_file(self):
        invalid_pdf = self.test_dir / "invalid.pdf"
        invalid_pdf.write_text("Not a PDF file")
        with self.assertRaises(Exception):
            pdf_to_markdown(invalid_pdf)

    def test_pdf_to_markdown_non_existent_file(self):
        non_existent = self.test_dir / "non_existent.pdf"
        with self.assertRaises(Exception):
            pdf_to_markdown(non_existent)


class TestPdfToMarkdownFile(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.test_pdf = self.test_dir / "test.pdf"
        self.test_md = self.test_dir / "test.md"
        writer = PyPDF2.PdfWriter()
        page = writer.add_blank_page(width=612, height=792)
        writer.add_metadata({"/Title": "Test PDF"})
        with open(self.test_pdf, "wb") as f:
            writer.write(f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pdf_to_markdown_file_with_output_path(self):
        pdf_to_markdown_file(self.test_pdf, self.test_md)

        self.assertTrue(self.test_md.exists())
        content = self.test_md.read_text()
        self.assertIsInstance(content, str)
        self.assertNotIn("Page 1 of 1", content)

    def test_pdf_to_markdown_file_without_output_path(self):
        pdf_to_markdown_file(self.test_pdf)

        expected_output = self.test_pdf.with_suffix(".md")
        self.assertTrue(expected_output.exists())
        content = expected_output.read_text()
        self.assertIsInstance(content, str)
        self.assertNotIn("Page 1 of 1", content)

    def test_pdf_to_markdown_file_with_page_numbers(self):
        pdf_to_markdown_file(self.test_pdf, self.test_md, page_number=True)

        self.assertTrue(self.test_md.exists())
        content = self.test_md.read_text()
        self.assertIsInstance(content, str)

    def test_pdf_to_markdown_file_invalid_pdf(self):
        invalid_pdf = self.test_dir / "invalid.pdf"
        invalid_pdf.write_text("Not a PDF file")
        with self.assertRaises(Exception):
            pdf_to_markdown_file(invalid_pdf, self.test_md)


if __name__ == "__main__":
    unittest.main()
