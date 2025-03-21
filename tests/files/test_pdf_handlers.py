import tempfile
import unittest
from pathlib import Path

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


class TestPDFHandlers(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a test PDF with metadata
        self.pdf_path = self.test_dir / "test.pdf"
        self.create_test_pdf(self.pdf_path, title="Test Document")

        # Create a PDF without metadata
        self.pdf_no_meta_path = self.test_dir / "no_meta.pdf"
        self.create_test_pdf(self.pdf_no_meta_path)

        # Create invalid PDF
        self.invalid_pdf = self.test_dir / "invalid.pdf"
        self.invalid_pdf.write_text("Not a PDF file")

        # Create non-PDF file
        self.non_pdf = self.test_dir / "test.txt"
        self.non_pdf.write_text("Text file")

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_test_pdf(self, path, title=None):
        pdf_writer = PyPDF2.PdfWriter()
        # Create two pages to ensure we get page numbers
        pdf_writer.add_blank_page(width=612, height=792)
        pdf_writer.add_blank_page(width=612, height=792)

        if title:
            pdf_writer.add_metadata({"/Title": title})

        with open(path, "wb") as f:
            pdf_writer.write(f)

    def test_extract_pdf_metadata_valid(self):
        metadata = extract_pdf_metadata(self.pdf_path)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata.get("Title"), "Test Document")

    def test_extract_pdf_metadata_no_metadata(self):
        metadata = extract_pdf_metadata(self.pdf_no_meta_path)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata, {"Producer": "mtp"})

    def test_extract_pdf_metadata_invalid_file(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.invalid_pdf)

    def test_extract_pdf_metadata_nonexistent_file(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_metadata(self.test_dir / "nonexistent.pdf")

    def test_extract_pdf_title_valid(self):
        title = extract_pdf_title(self.pdf_path)
        self.assertEqual(title, "Test Document")

    def test_extract_pdf_title_no_metadata(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_title(self.pdf_no_meta_path)

    def test_extract_pdf_title_invalid_file(self):
        with self.assertRaises(ArgumentTypeError):
            extract_pdf_title(self.non_pdf)

    def test_extract_pdf_title_nonexistent_file(self):
        with self.assertRaises(ArgumentValueError):
            extract_pdf_title(self.test_dir / "nonexistent.pdf")

    def test_set_pdf_title_as_filename_valid(self):
        set_pdf_title_as_filename(self.pdf_path)
        self.assertTrue((self.test_dir / "Test_Document.pdf").exists())
        self.assertFalse(self.pdf_path.exists())

    def test_set_pdf_title_as_filename_attempt_true(self):
        original_path = self.pdf_path
        set_pdf_title_as_filename(self.pdf_path, attempt=True)
        self.assertTrue(original_path.exists())
        self.assertFalse((self.test_dir / "Test_Document.pdf").exists())

    def test_set_pdf_title_as_filename_overwrite(self):
        target_path = self.test_dir / "Test_Document.pdf"
        self.create_test_pdf(target_path)

        set_pdf_title_as_filename(self.pdf_path, overwrite=True)
        self.assertTrue(target_path.exists())
        self.assertFalse(self.pdf_path.exists())

    def test_set_pdf_title_as_filename_no_overwrite(self):
        target_path = self.test_dir / "Test_Document.pdf"
        self.create_test_pdf(target_path)
        set_pdf_title_as_filename(self.pdf_path, overwrite=False)
        self.assertTrue(target_path.exists())
        self.assertTrue((self.test_dir / "Test_Document_1.pdf").exists())
        self.assertFalse(self.pdf_path.exists())

    def test_set_folder_pdfs_titles_as_filenames(self):
        set_folder_pdfs_titles_as_filenames(self.test_dir)
        self.assertTrue((self.test_dir / "Test_Document.pdf").exists())
        self.assertFalse(self.pdf_path.exists())
        self.assertTrue(
            self.pdf_no_meta_path.exists()
        )  # Should be skipped due to no title
        self.assertTrue(self.non_pdf.exists())  # Should be skipped as non-PDF

    def test_set_folder_pdfs_titles_as_filenames_invalid_dir(self):
        with self.assertRaises(ArgumentValueError):
            set_folder_pdfs_titles_as_filenames(self.test_dir / "nonexistent")

    def test_pdf_to_markdown_valid(self):
        markdown = pdf_to_markdown(self.pdf_path)
        self.assertIsInstance(markdown, str)

    def test_pdf_to_markdown_with_page_numbers(self):
        markdown = pdf_to_markdown(self.pdf_path, page_number=True)
        self.assertIsInstance(markdown, str)
        self.assertIn("-----", markdown)
        pages = markdown.split("-----")
        self.assertEqual(len(pages), 3)

    def test_pdf_to_markdown_invalid_file(self):
        with self.assertRaises(ArgumentValueError):
            pdf_to_markdown(self.invalid_pdf)

    def test_pdf_to_markdown_file_valid(self):
        output_path = pdf_to_markdown_file(self.pdf_path)
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.suffix, ".md")

        content = output_path.read_text()
        self.assertIsInstance(content, str)

    def test_pdf_to_markdown_file_custom_output(self):
        output_path = self.test_dir / "custom_output.md"
        result_path = pdf_to_markdown_file(self.pdf_path, output_path)

        self.assertEqual(result_path, output_path)
        self.assertTrue(output_path.exists())

        content = output_path.read_text()
        self.assertIsInstance(content, str)

    def test_pdf_to_markdown_file_with_page_numbers(self):
        output_path = pdf_to_markdown_file(self.pdf_path, page_number=True)
        content = output_path.read_text()
        # The output should contain at least one page separator
        self.assertIn("-----", content)
        # Each page should be separated by -----
        pages = content.split("-----")
        # We created a 2-page PDF, so we should have 3 parts (including empty first/last parts)
        self.assertEqual(len(pages), 3)


if __name__ == "__main__":
    unittest.main()
