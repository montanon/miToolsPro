import hashlib
import re
from typing import Optional

import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor
from nbformat.notebooknode import NotebookNode, from_dict

from mitoolspro.notebooks.objects import (
    CodeMirrorMode,
    KernelSpec,
    LanguageInfo,
    Notebook,
    NotebookCell,
    NotebookCellFactory,
    NotebookMetadata,
)


def custom_notebook_to_notebooknode(custom_nb: Notebook) -> NotebookNode:
    nb_dict = custom_nb.to_dict()
    return from_dict(nb_dict)


def notebooknode_to_custom_notebook(nb_node: NotebookNode) -> Notebook:
    cells = [
        NotebookCellFactory.create_cell(
            cell_type=cell["cell_type"],
            execution_count=cell.get("execution_count"),
            cell_id=cell.get("cell_id", ""),
            metadata=cell.get("metadata", {}),
            outputs=cell.get("outputs", []),
            source=cell.get("source", []),
        )
        for cell in nb_node["cells"]
    ]

    metadata = NotebookMetadata(
        kernelspec=KernelSpec(
            display_name=nb_node["metadata"]["kernelspec"]["display_name"],
            language=nb_node["metadata"]["kernelspec"]["language"],
            name=nb_node["metadata"]["kernelspec"]["name"],
        ),
        language_info=LanguageInfo(
            codemirror_mode=CodeMirrorMode(
                name=nb_node["metadata"]["language_info"]["codemirror_mode"]["name"],
                version=nb_node["metadata"]["language_info"]["codemirror_mode"][
                    "version"
                ],
            ),
            file_extension=nb_node["metadata"]["language_info"]["file_extension"],
            mimetype=nb_node["metadata"]["language_info"]["mimetype"],
            name=nb_node["metadata"]["language_info"]["name"],
            nbconvert_exporter=nb_node["metadata"]["language_info"][
                "nbconvert_exporter"
            ],
            pygments_lexer=nb_node["metadata"]["language_info"]["pygments_lexer"],
            version=nb_node["metadata"]["language_info"]["version"],
        ),
    )

    return Notebook(
        cells=cells,
        metadata=metadata,
        nbformat=nb_node["nbformat"],
        nbformat_minor=nb_node["nbformat_minor"],
        name=nb_node.get("name", ""),
        notebook_id=nb_node.get("notebook_id", ""),
    )


def clear_notebook_output(notebook_path: str, clean_notebook_path: str) -> None:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(
            f,
            as_version=4,
        )
    co_processor = ClearOutputPreprocessor()
    co_processor.preprocess(nb, {"metadata": {"path": "./"}})
    with open(clean_notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def create_notebook(
    cells: list[NotebookCell],
    metadata: NotebookMetadata,
    nbformat: int,
    nbformat_minor: int,
    name: Optional[str] = "",
    notebook_id: Optional[str] = "",
) -> Notebook:
    return Notebook(
        cells=cells,
        metadata=metadata,
        nbformat=nbformat,
        nbformat_minor=nbformat_minor,
        name=name,
        notebook_id=notebook_id,
    )


def create_notebook_metadata(
    language_info: LanguageInfo, kernelspec: Optional[KernelSpec] = None
) -> NotebookMetadata:
    return NotebookMetadata(kernelspec=kernelspec, language_info=language_info)


def create_notebook_cell(
    cell_type: str,
    execution_count: None,
    cell_id: str,
    metadata: dict,
    outputs: list,
    source: list,
) -> NotebookCell:
    cell = NotebookCellFactory.create_cell(
        cell_type=cell_type,
        execution_count=execution_count,
        cell_id=cell_id,
        metadata=metadata,
        outputs=outputs,
        source=source,
    )
    return cell


def create_notebook_cell_id(notebook_seed: str, cell_seed: str) -> str:
    seed = notebook_seed + cell_seed
    hasher = hashlib.sha256(seed.encode())
    hash_str = hasher.hexdigest()
    return hash_str[:16]


def create_code_mirror_mode(name: str, version: int) -> CodeMirrorMode:
    return CodeMirrorMode(name=name, version=version)


def create_language_info(
    codemirror_mode: CodeMirrorMode,
    file_extension: str,
    mimetype: str,
    name: str,
    nbconvert_exporter: str,
    pygments_lexer: str,
    version: str,
) -> LanguageInfo:
    return LanguageInfo(
        codemirror_mode=codemirror_mode,
        file_extension=file_extension,
        mimetype=mimetype,
        name=name,
        nbconvert_exporter=nbconvert_exporter,
        pygments_lexer=pygments_lexer,
        version=version,
    )


def create_kernel_spec(display_name: str, language: str, name: str) -> KernelSpec:
    return KernelSpec(display_name=display_name, language=language, name=name)


def validate_hex_string(value: str) -> str:
    if not re.match(r"^[0-9a-fA-F]{16}$", value):
        raise ValueError(f"The value {value} is not a valid hex string.")
    return value
