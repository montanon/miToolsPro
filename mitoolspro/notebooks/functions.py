import sys
from os import PathLike
from typing import List, Optional, Union

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
    NotebookCells,
    NotebookMetadata,
    NotebookSection,
    NotebookSections,
    create_notebook_cell_id,
)


def read_notebook(notebook_path: Union[str, PathLike]) -> Notebook:
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        return notebooknode_to_custom_notebook(nb)
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")
    except nbformat.reader.NotJSONError:
        raise ValueError(f"Invalid notebook format in file: {notebook_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading notebook {notebook_path}: {str(e)}")


def write_notebook(notebook: Notebook, notebook_path: Union[str, PathLike]) -> None:
    try:
        validate_notebook(notebook)
        nb_node = custom_notebook_to_notebooknode(notebook)
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb_node, f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found for notebook: {notebook_path}")
    except PermissionError:
        raise PermissionError(f"No write permission for notebook: {notebook_path}")
    except Exception as e:
        raise RuntimeError(f"Error writing notebook {notebook_path}: {str(e)}")


def validate_notebook(notebook: Notebook) -> None:
    nbformat.validate(custom_notebook_to_notebooknode(notebook))


def custom_notebook_to_notebooknode(custom_nb: Notebook) -> NotebookNode:
    nb_dict = custom_nb.to_nb()
    return from_dict(nb_dict)


def notebooknode_to_custom_notebook(nb_node: NotebookNode) -> Notebook:
    try:
        cells = [
            NotebookCellFactory.create_cell(
                cell_type=cell["cell_type"],
                execution_count=cell.get("execution_count"),
                id=cell.get("id", ""),
                metadata=cell.get("metadata", {}),
                outputs=cell.get("outputs", []),
                source=cell.get("source", []),
            )
            for cell in nb_node.get("cells", [])
        ]
    except KeyError as e:
        raise ValueError(f"Invalid cell structure: missing required field {e}")
    try:
        metadata = NotebookMetadata(
            kernelspec=KernelSpec(
                display_name=nb_node.get("metadata", {})
                .get("kernelspec", {})
                .get("display_name", ""),
                language=nb_node.get("metadata", {})
                .get("kernelspec", {})
                .get("language", ""),
                name=nb_node.get("metadata", {}).get("kernelspec", {}).get("name", ""),
            ),
            language_info=LanguageInfo(
                codemirror_mode=CodeMirrorMode(
                    name=nb_node.get("metadata", {})
                    .get("language_info", {})
                    .get("codemirror_mode", {})
                    .get("name", ""),
                    version=nb_node.get("metadata", {})
                    .get("language_info", {})
                    .get("codemirror_mode", {})
                    .get("version", 4),
                ),
                file_extension=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("file_extension", ""),
                mimetype=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("mimetype", ""),
                name=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("name", ""),
                nbconvert_exporter=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("nbconvert_exporter", ""),
                pygments_lexer=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("pygments_lexer", ""),
                version=nb_node.get("metadata", {})
                .get("language_info", {})
                .get("version", ""),
            ),
        )
    except KeyError as e:
        raise ValueError(f"Invalid metadata structure: missing required field {e}")

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
    cells: Union[
        Union[NotebookCell, NotebookCells, NotebookSection, NotebookSections],
        List[Union[NotebookCell, NotebookCells, NotebookSection, NotebookSections]],
    ],
    metadata: NotebookMetadata,
    nbformat: int,
    nbformat_minor: int,
    name: Optional[str] = "",
    notebook_id: Optional[str] = "",
) -> Notebook:
    if not isinstance(cells, list):
        cells = [cells]

    all_cells = []
    section_indices = []
    current_index = 0

    for cell_container in cells:
        if isinstance(cell_container, NotebookCell):
            all_cells.append(cell_container)
            current_index += 1
        elif isinstance(cell_container, NotebookCells):
            all_cells.extend(cell_container.cells)
            current_index += len(cell_container.cells)
        elif isinstance(cell_container, NotebookSection):
            start_index = current_index
            all_cells.extend(cell_container.cells)
            current_index += len(cell_container.cells)
            section_indices.append((start_index, current_index))
        elif isinstance(cell_container, NotebookSections):
            for section in cell_container.sections:
                start_index = current_index
                all_cells.extend(section.cells)
                current_index += len(section.cells)
                section_indices.append((start_index, current_index))

    notebook = Notebook(
        cells=NotebookCells(all_cells),
        metadata=metadata,
        nbformat=nbformat,
        nbformat_minor=nbformat_minor,
        name=name,
        notebook_id=notebook_id,
    )
    object.__setattr__(notebook, "_section_indices", section_indices)
    return notebook


def create_notebook_metadata(
    language_info: LanguageInfo, kernelspec: Optional[KernelSpec] = None
) -> NotebookMetadata:
    return NotebookMetadata(kernelspec=kernelspec, language_info=language_info)


def create_notebook_section(
    title: str,
    cells: list[NotebookCell],
    notebook_seed: str,
    section_seed: str,
) -> NotebookSection:
    title_cell = create_notebook_cell(
        cell_type="markdown",
        execution_count=None,
        notebook_seed=notebook_seed,
        cell_seed=f"{section_seed}_title",
        metadata={},
        outputs=[],
        source=[title],
    )
    section_cells = NotebookCells([title_cell] + cells)
    return NotebookSection(cells=section_cells)


def create_notebook_sections(
    sections: list[tuple[str, list[NotebookCell]]],
    notebook_seed: str,
) -> NotebookSections:
    notebook_sections = []
    for i, (title, cells) in enumerate(sections):
        section = create_notebook_section(
            title=title,
            cells=cells,
            notebook_seed=notebook_seed,
            section_seed=f"section_{i}",
        )
        notebook_sections.append(section)
    return NotebookSections(sections=notebook_sections)


def create_notebook_cell(
    cell_type: str,
    source: list,
    notebook_seed: str,
    cell_seed: str,
    metadata: Optional[dict] = None,
    outputs: Optional[list] = None,
    execution_count: Optional[int] = None,
    deletable: bool = True,
    editable: bool = True,
) -> NotebookCell:
    cell = NotebookCellFactory.create_cell(
        cell_type=cell_type,
        execution_count=execution_count,
        id=create_notebook_cell_id(notebook_seed=notebook_seed, cell_seed=cell_seed),
        metadata=metadata,
        outputs=outputs,
        source=source,
        deletable=deletable,
        editable=editable,
    )
    return cell


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


def create_default_metadata() -> NotebookMetadata:
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    codemirror_mode = create_code_mirror_mode(name="python", version=4)
    language_info = create_language_info(
        codemirror_mode=codemirror_mode,
        file_extension=".py",
        mimetype="text/x-python",
        name="python",
        nbconvert_exporter="python",
        pygments_lexer="ipython3",
        version=python_version,
    )

    kernelspec = create_kernel_spec(
        display_name=f"Python {python_version}",
        language="python",
        name=f"python{sys.version_info.major}",
    )

    return create_notebook_metadata(
        language_info=language_info,
        kernelspec=kernelspec,
    )
