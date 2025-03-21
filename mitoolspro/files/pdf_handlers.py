import logging
import re
from os import PathLike
from pathlib import Path
from typing import Dict, Union

import pymupdf
import pymupdf4llm
import PyPDF2

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.files import remove_characters_from_string, rename_file

logger = logging.getLogger("mtp")
PATTERN = "^([A-Za-z0-9.]+-)+[A-Za-z0-9]+.pdf$"


def extract_pdf_metadata(pdf_filename: PathLike) -> Union[Dict[str, str], None]:
    pdf_filename = Path(pdf_filename)
    if not pdf_filename.is_file():
        raise ArgumentValueError(f"'{pdf_filename}' is not a valid file path.")
    metadata = {}
    try:
        with pdf_filename.open("rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            doc_info = pdf_reader.metadata or {}
            for key, value in doc_info.items():
                sanitized_key = key.lstrip("/")
                text_value = str(value).encode("utf-8", errors="ignore").decode("utf-8")
                if sanitized_key == "Producer" and text_value == "PyPDF2":
                    text_value = "mtp"
                metadata[sanitized_key] = text_value
    except (FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
        raise ArgumentValueError(f"Error reading PDF '{pdf_filename}': {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")
    return metadata


def extract_pdf_title(pdf_filename: PathLike) -> str:
    pdf_filename = Path(pdf_filename)
    if not pdf_filename.is_file():
        raise ArgumentValueError(f"'{pdf_filename}' is not a valid file path.")
    if pdf_filename.suffix.lower() != ".pdf":
        raise ArgumentTypeError(f"'{pdf_filename}' is not a valid PDF file.")
    metadata = extract_pdf_metadata(pdf_filename)
    if "Title" in metadata:
        return metadata["Title"]
    raise ArgumentValueError(f"'{pdf_filename}' has no title in its metadata.")


def set_pdf_title_as_filename(
    pdf_filename: PathLike, attempt: bool = False, overwrite: bool = False
) -> None:
    pdf_filename = Path(pdf_filename).resolve(strict=True)
    if pdf_filename.suffix.lower() != ".pdf":
        raise ArgumentTypeError(f"'{pdf_filename}' is not a valid PDF file.")
    title = extract_pdf_title(pdf_filename)
    title = remove_characters_from_string(title).replace(" ", "_")
    new_filename = pdf_filename.with_name(f"{title}.pdf")
    rename_file(
        file=pdf_filename, new_name=new_filename, overwrite=overwrite, attempt=attempt
    )


def set_folder_pdfs_titles_as_filenames(
    folder_path: PathLike, attempt: bool = False, overwrite: bool = False
) -> None:
    folder = Path(folder_path).resolve(strict=False)
    if not folder.is_dir():
        raise ArgumentValueError(f"'{folder_path=}' is not a valid directory.")
    for file in folder.iterdir():
        if not file.is_file() or file.suffix.lower() != ".pdf":
            continue
        try:
            set_pdf_title_as_filename(file, overwrite=overwrite, attempt=attempt)
        except Exception as e:
            logger.warning(f"Error processing '{file.name}': {e}")


def _clean_markdown_footer(md: str) -> str:
    return re.sub(r"\n?Page \d+.*", "", md).strip()


def pdf_to_markdown(pdf_path: PathLike, page_number: bool = False) -> str:
    try:
        document = pymupdf.open(pdf_path)
    except Exception as e:
        raise ArgumentValueError(f"Could not open PDF '{pdf_path}': {e}")

    md_document = []
    for n in range(document.page_count):
        try:
            md_page = pymupdf4llm.to_markdown(document, pages=[n], show_progress=False)
            if not page_number:
                md_page = _clean_markdown_footer(md_page)
            md_document.append(md_page)
        except Exception as e:
            raise RuntimeError(f"Error converting page {n} to markdown: {e}")

    return "\n".join(md_document)


def pdf_to_markdown_file(
    pdf_path: PathLike, output_path: PathLike = None, page_number: bool = False
) -> Path:
    md_document = pdf_to_markdown(pdf_path=pdf_path, page_number=page_number)
    output_path = Path(output_path or Path(pdf_path).with_suffix(".md"))
    with output_path.open("w", encoding="utf-8") as f:
        f.write(md_document)
    return output_path
