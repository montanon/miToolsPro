from os import PathLike
from pathlib import Path

import pypandoc

from mitoolspro.exceptions import ArgumentValueError


def convert_file(
    source_file: PathLike,
    output_file: PathLike,
    output_format: str,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    if Path(output_file).exists() and not exist_ok and not overwrite:
        raise ArgumentValueError(f"'{output_file}' already exists.")
    elif Path(output_file).exists() and exist_ok and not overwrite:
        return
    output = pypandoc.convert_file(
        source_file=source_file, to=output_format, outputfile=output_file
    )
    assert output == "", f"Conversion failed: {output}"


def convert_directory_files(
    source_directory: PathLike,
    output_directory: PathLike,
    input_format: str,
    output_format: str,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    for file in source_directory.glob(f"*.{input_format}"):
        output_file = (
            output_directory / file.with_suffix(f".{output_format}").name
            if output_directory
            else file.with_suffix(f".{output_format}")
        )
        convert_file(
            source_file=file,
            output_file=output_file,
            output_format=output_format,
            exist_ok=exist_ok,
            overwrite=overwrite,
        )


def convert_docx_to_pdf(
    source_file: PathLike,
    output_file: PathLike,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    convert_file(
        source_file=source_file,
        output_file=output_file,
        output_format="pdf",
        exist_ok=exist_ok,
        overwrite=overwrite,
    )


def convert_directory_docxs_to_pdfs(
    source_directory: PathLike,
    output_directory: PathLike = None,
    exist_ok: bool = True,
    overwrite: bool = True,
) -> None:
    convert_directory_files(
        source_directory=source_directory,
        output_directory=output_directory,
        input_format="docx",
        output_format="pdf",
        exist_ok=exist_ok,
        overwrite=overwrite,
    )
