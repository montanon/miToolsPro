import json
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from os import PathLike
from typing import Any, Dict, Iterator, List, Optional

from mitoolspro.notebooks.functions import create_notebook_cell_id, validate_hex_string


@dataclass(frozen=True)
class Notebook:
    cells: "NotebookCells"
    metadata: "NotebookMetadata"
    nbformat: int
    nbformat_minor: int
    name: str
    path: PathLike = field(default="")
    notebook_id: str = field(
        default=uuid.uuid4().hex, metadata={"validator": validate_hex_string}
    )

    def __post_init__(self):
        new_cells = []
        for i, cell in enumerate(self.cells):
            cell_id = create_notebook_cell_id(self.notebook_id, str(i))
            new_cell = replace(cell, id=validate_hex_string(cell_id))
            new_cells.append(new_cell)
        object.__setattr__(self, "cells", NotebookCells(new_cells))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


class NotebookEncoder(json.JSONEncoder):
    def default(self, output: Any) -> Any:
        if is_dataclass(output):
            return asdict(output)
        if isinstance(output, PathLike):
            return str(output)
        return super().default(output)


@dataclass(frozen=True)
class NotebookCells:
    cells: list["NotebookCell"]

    def __iter__(self) -> Iterator["NotebookCell"]:
        return iter(self.cells)

    def __getitem__(self, index) -> "NotebookCell":
        return self.cells[index]

    def __len__(self) -> int:
        return len(self.cells)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True)
class NotebookSection(NotebookCells):
    cells: NotebookCells

    def __post_init__(self):
        super().__init__(self.cells)
        if not isinstance(self.cells, NotebookCells):
            raise ValueError("cells must be an instance of NotebookCells.")
        try:
            first_cell = next(iter(self.cells))
        except StopIteration:
            raise ValueError("The cells list is empty.")
        if not isinstance(first_cell, MarkdownCell):
            raise ValueError("The first cell must be a MarkdownCell.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True)
class NotebookCell:
    cell_type: str
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    outputs: List[Any] = field(default_factory=list)
    source: List[str] = field(default_factory=list)
    cell_id: str = field(default="", metadata={"validator": validate_hex_string})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True)
class MarkdownCell(NotebookCell):
    def __post_init__(self):
        if object.__getattribute__(self, "cell_type") != "markdown":
            raise ValueError(
                f"cell_type of MarkdownCell must be 'markdown', got {object.__getattribute__(self, 'cell_type')}"
            )
        object.__setattr__(self, "cell_type", "markdown")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True)
class CodeCell(NotebookCell):
    def __post_init__(self):
        if object.__getattribute__(self, "cell_type") != "code":
            raise ValueError(
                f"cell_type of CodeCell must be 'code', got {object.__getattribute__(self, 'cell_type')}"
            )
        object.__setattr__(self, "cell_type", "code")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True)
class ImportCell(NotebookCell):
    def __post_init__(self):
        if object.__getattribute__(self, "cell_type") != "code":
            raise ValueError(
                f"cell_type of CodeCell must be 'code', got {object.__getattribute__(self, 'cell_type')}"
            )
        object.__setattr__(self, "cell_type", "code")

        source = object.__getattribute__(self, "source")
        for line in source:
            line = line.strip()
            if line and not (line.startswith("import ") or line.startswith("from ")):
                raise ValueError(
                    f"ImportCell can only contain import statements or empty lines, got: {line}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass(frozen=True, init=True)
class NotebookCellFactory:
    cell_types = {"code": CodeCell, "markdown": MarkdownCell, "import": ImportCell}

    @staticmethod
    def create_cell(cell_type: str, *args, **kwargs):
        cell_class = NotebookCellFactory.cell_types.get(cell_type.lower(), NotebookCell)
        cell = cell_class(cell_type=cell_type, *args, **kwargs)
        return cell


@dataclass(frozen=True)
class NotebookMetadata:
    kernelspec: "KernelSpec"
    language_info: "LanguageInfo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass
class KernelSpec:
    display_name: str
    language: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass
class LanguageInfo:
    codemirror_mode: "CodeMirrorMode"
    file_extension: str
    mimetype: str
    name: str
    nbconvert_exporter: str
    pygments_lexer: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)


@dataclass
class CodeMirrorMode:
    name: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), cls=NotebookEncoder, **json_kwargs)
