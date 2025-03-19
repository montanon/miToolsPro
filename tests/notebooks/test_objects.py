import json
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List
from unittest import TestCase

from mitoolspro.notebooks.objects import (
    CodeCell,
    CodeMirrorMode,
    ImportCell,
    KernelSpec,
    LanguageInfo,
    MarkdownCell,
    Notebook,
    NotebookCell,
    NotebookCells,
    NotebookMetadata,
    NotebookSection,
    NotebookSections,
)


class TestNotebookCell(TestCase):
    def setUp(self):
        self.cell = NotebookCell(
            cell_type="code",
            execution_count=1,
            metadata={"key": "value"},
            outputs=["output"],
            source=["print('hello')"],
            cell_id="1234567890abcdef",
        )

    def test_creation(self):
        self.assertEqual(self.cell.cell_type, "code")
        self.assertEqual(self.cell.execution_count, 1)
        self.assertEqual(self.cell.metadata, {"key": "value"})
        self.assertEqual(self.cell.outputs, ["output"])
        self.assertEqual(self.cell.source, ["print('hello')"])
        self.assertEqual(self.cell.cell_id, "1234567890abcdef")

    def test_default_values(self):
        cell = NotebookCell(cell_type="code")
        self.assertEqual(cell.execution_count, None)
        self.assertEqual(cell.metadata, {})
        self.assertEqual(cell.outputs, [])
        self.assertEqual(cell.source, [])
        self.assertEqual(cell.cell_id, "")

    def test_to_dict(self):
        expected = {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {"key": "value"},
            "outputs": ["output"],
            "source": ["print('hello')"],
            "cell_id": "1234567890abcdef",
        }
        self.assertEqual(self.cell.to_dict(), expected)

    def test_to_json(self):
        json_str = self.cell.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"cell_type": "code"', json_str)
        self.assertIn('"execution_count": 1', json_str)


class TestMarkdownCell(TestCase):
    def setUp(self):
        self.cell = MarkdownCell(
            cell_type="markdown",
            metadata={"key": "value"},
            source=["# Title"],
            cell_id="1234567890abcdef",
        )

    def test_creation(self):
        self.assertEqual(self.cell.cell_type, "markdown")
        self.assertEqual(self.cell.source, ["# Title"])
        self.assertEqual(self.cell.cell_id, "1234567890abcdef")

    def test_invalid_cell_type(self):
        with self.assertRaises(ValueError):
            MarkdownCell(cell_type="code")

    def test_to_dict(self):
        expected = {
            "cell_type": "markdown",
            "execution_count": None,
            "metadata": {"key": "value"},
            "outputs": [],
            "source": ["# Title"],
            "cell_id": "1234567890abcdef",
        }
        self.assertEqual(self.cell.to_dict(), expected)


class TestCodeCell(TestCase):
    def setUp(self):
        self.cell = CodeCell(
            cell_type="code",
            execution_count=1,
            metadata={"key": "value"},
            outputs=["output"],
            source=["print('hello')"],
            cell_id="1234567890abcdef",
        )

    def test_creation(self):
        self.assertEqual(self.cell.cell_type, "code")
        self.assertEqual(self.cell.execution_count, 1)
        self.assertEqual(self.cell.source, ["print('hello')"])

    def test_invalid_cell_type(self):
        with self.assertRaises(ValueError):
            CodeCell(cell_type="markdown")


class TestImportCell(TestCase):
    def setUp(self):
        self.cell = ImportCell(
            cell_type="code",
            metadata={"key": "value"},
            source=["import numpy as np", "from pandas import DataFrame"],
            cell_id="1234567890abcdef",
        )

    def test_creation(self):
        self.assertEqual(self.cell.cell_type, "code")
        self.assertEqual(
            self.cell.source, ["import numpy as np", "from pandas import DataFrame"]
        )

    def test_invalid_cell_type(self):
        with self.assertRaises(ValueError):
            ImportCell(cell_type="markdown")

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            ImportCell(
                cell_type="code",
                source=["import numpy as np", "print('hello')"],
            )


class TestNotebookCells(TestCase):
    def setUp(self):
        self.cells = [
            MarkdownCell(cell_type="markdown", source=["# Title"]),
            CodeCell(cell_type="code", source=["print('hello')"]),
        ]
        self.notebook_cells = NotebookCells(cells=self.cells)

    def test_creation(self):
        self.assertEqual(len(self.notebook_cells), 2)
        self.assertEqual(self.notebook_cells[0], self.cells[0])
        self.assertEqual(self.notebook_cells[1], self.cells[1])

    def test_iteration(self):
        cells = list(self.notebook_cells)
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[0], self.cells[0])
        self.assertEqual(cells[1], self.cells[1])

    def test_indexing(self):
        self.assertEqual(self.notebook_cells[0], self.cells[0])
        self.assertEqual(self.notebook_cells[-1], self.cells[-1])
        with self.assertRaises(IndexError):
            self.notebook_cells[2]

    def test_to_dict(self):
        expected = {"cells": [cell.to_dict() for cell in self.cells]}
        self.assertEqual(self.notebook_cells.to_dict(), expected)


class TestNotebookSection(TestCase):
    def setUp(self):
        self.title_cell = MarkdownCell(cell_type="markdown", source=["# Section"])
        self.code_cell = CodeCell(cell_type="code", source=["print('hello')"])
        self.cells = NotebookCells(cells=[self.title_cell, self.code_cell])
        self.section = NotebookSection(cells=self.cells)

    def test_creation(self):
        self.assertEqual(len(self.section.cells), 2)
        self.assertEqual(self.section.cells[0], self.title_cell)
        self.assertEqual(self.section.cells[1], self.code_cell)

    def test_invalid_cells_type(self):
        with self.assertRaises(ValueError):
            NotebookSection(cells=[self.title_cell, self.code_cell])

    def test_empty_cells(self):
        with self.assertRaises(ValueError):
            NotebookSection(cells=NotebookCells(cells=[]))

    def test_first_cell_not_markdown(self):
        with self.assertRaises(ValueError):
            NotebookSection(
                cells=NotebookCells(cells=[self.code_cell, self.title_cell])
            )

    def test_to_dict(self):
        expected = {"cells": [cell.to_dict() for cell in self.cells]}
        self.assertEqual(self.section.to_dict(), expected)


class TestNotebookSections(TestCase):
    def setUp(self):
        self.title_cell1 = MarkdownCell(cell_type="markdown", source=["# Section 1"])
        self.title_cell2 = MarkdownCell(cell_type="markdown", source=["# Section 2"])
        self.code_cell1 = CodeCell(cell_type="code", source=["print('hello')"])
        self.code_cell2 = CodeCell(cell_type="code", source=["print('world')"])

        self.section1 = NotebookSection(
            cells=NotebookCells(cells=[self.title_cell1, self.code_cell1])
        )
        self.section2 = NotebookSection(
            cells=NotebookCells(cells=[self.title_cell2, self.code_cell2])
        )
        self.sections = NotebookSections(sections=[self.section1, self.section2])

    def test_creation(self):
        self.assertEqual(len(self.sections), 2)
        self.assertEqual(self.sections[0], self.section1)
        self.assertEqual(self.sections[1], self.section2)

    def test_iteration(self):
        sections = list(self.sections)
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0], self.section1)
        self.assertEqual(sections[1], self.section2)

    def test_indexing(self):
        self.assertEqual(self.sections[0], self.section1)
        self.assertEqual(self.sections[-1], self.section2)
        with self.assertRaises(IndexError):
            self.sections[2]

    def test_to_dict(self):
        expected = {"sections": [self.section1.to_dict(), self.section2.to_dict()]}
        self.assertEqual(self.sections.to_dict(), expected)


class TestNotebook(TestCase):
    def setUp(self):
        self.title_cell1 = MarkdownCell(cell_type="markdown", source=["# Section 1"])
        self.title_cell2 = MarkdownCell(cell_type="markdown", source=["# Section 2"])
        self.code_cell1 = CodeCell(cell_type="code", source=["print('hello')"])
        self.code_cell2 = CodeCell(cell_type="code", source=["print('world')"])

        self.section1 = NotebookSection(
            cells=NotebookCells(cells=[self.title_cell1, self.code_cell1])
        )
        self.section2 = NotebookSection(
            cells=NotebookCells(cells=[self.title_cell2, self.code_cell2])
        )

        self.codemirror_mode = CodeMirrorMode(name="python", version=4)
        self.language_info = LanguageInfo(
            codemirror_mode=self.codemirror_mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.0",
        )
        self.kernelspec = KernelSpec(
            display_name="Python 3",
            language="python",
            name="python3",
        )
        self.metadata = NotebookMetadata(
            language_info=self.language_info,
            kernelspec=self.kernelspec,
        )

        self.notebook = Notebook(
            cells=NotebookCells(
                cells=[
                    self.title_cell1,
                    self.code_cell1,
                    self.title_cell2,
                    self.code_cell2,
                ]
            ),
            metadata=self.metadata,
            nbformat=4,
            nbformat_minor=5,
            name="Test Notebook",
            notebook_id="1234567890abcdef",
            _section_indices=[(0, 2), (2, 4)],
        )

    def test_creation(self):
        self.assertEqual(len(self.notebook.cells), 4)
        self.assertEqual(self.notebook.nbformat, 4)
        self.assertEqual(self.notebook.nbformat_minor, 5)
        self.assertEqual(self.notebook.name, "Test Notebook")
        self.assertEqual(self.notebook.notebook_id, "1234567890abcdef")
        self.assertEqual(self.notebook._section_indices, [(0, 2), (2, 4)])

    def test_sections_property(self):
        sections = self.notebook.sections
        self.assertEqual(len(sections), 2)
        self.assertEqual(len(sections[0].cells), 2)
        self.assertEqual(len(sections[1].cells), 2)
        self.assertEqual(sections[0].cells[0].cell_type, self.title_cell1.cell_type)
        self.assertEqual(sections[0].cells[0].source, self.title_cell1.source)
        self.assertEqual(sections[1].cells[0].cell_type, self.title_cell2.cell_type)
        self.assertEqual(sections[1].cells[0].source, self.title_cell2.source)

    def test_sections_property_empty_indices(self):
        notebook = Notebook(
            cells=NotebookCells(cells=[self.title_cell1, self.code_cell1]),
            metadata=self.metadata,
            nbformat=4,
            nbformat_minor=5,
            name="Test Notebook",
            notebook_id="1234567890abcdef",
            _section_indices=[],
        )
        self.assertEqual(len(notebook.sections), 0)

    def test_get_sections(self):
        sections = self.notebook.get_sections()
        self.assertEqual(len(sections), 2)
        self.assertEqual(len(sections[0].cells), 2)
        self.assertEqual(len(sections[1].cells), 2)
        self.assertEqual(sections[0].cells[0].cell_type, self.title_cell1.cell_type)
        self.assertEqual(sections[0].cells[0].source, self.title_cell1.source)
        self.assertEqual(sections[1].cells[0].cell_type, self.title_cell2.cell_type)
        self.assertEqual(sections[1].cells[0].source, self.title_cell2.source)

    def test_to_dict(self):
        nb_dict = self.notebook.to_dict()

        self.assertEqual(nb_dict["nbformat"], 4)
        self.assertEqual(nb_dict["nbformat_minor"], 5)
        self.assertEqual(nb_dict["name"], "Test Notebook")
        self.assertEqual(nb_dict["notebook_id"], "1234567890abcdef")
        self.assertEqual(nb_dict["path"], "")

        cells = nb_dict["cells"]
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[0]["cell_type"], "markdown")
        self.assertEqual(cells[0]["source"], ["# Section 1"])
        self.assertEqual(cells[1]["cell_type"], "code")
        self.assertEqual(cells[1]["source"], ["print('hello')"])
        self.assertEqual(cells[2]["cell_type"], "markdown")
        self.assertEqual(cells[2]["source"], ["# Section 2"])
        self.assertEqual(cells[3]["cell_type"], "code")
        self.assertEqual(cells[3]["source"], ["print('world')"])

        metadata = nb_dict["metadata"]
        self.assertEqual(metadata["kernelspec"]["display_name"], "Python 3")
        self.assertEqual(metadata["kernelspec"]["language"], "python")
        self.assertEqual(metadata["kernelspec"]["name"], "python3")
        self.assertEqual(metadata["language_info"]["name"], "python")
        self.assertEqual(metadata["language_info"]["version"], "3.8.0")
        self.assertEqual(metadata["language_info"]["file_extension"], ".py")
        self.assertEqual(metadata["language_info"]["mimetype"], "text/x-python")
        self.assertEqual(metadata["language_info"]["nbconvert_exporter"], "python")
        self.assertEqual(metadata["language_info"]["pygments_lexer"], "ipython3")
        self.assertEqual(metadata["language_info"]["codemirror_mode"]["name"], "python")
        self.assertEqual(metadata["language_info"]["codemirror_mode"]["version"], 4)

    def test_to_json(self):
        json_str = self.notebook.to_json(indent=4)
        self.assertIsInstance(json_str, str)

        try:
            parsed = json.loads(json_str)
            self.assertEqual(parsed["nbformat"], 4)
            self.assertEqual(parsed["nbformat_minor"], 5)
            self.assertEqual(parsed["name"], "Test Notebook")
            self.assertEqual(parsed["notebook_id"], "1234567890abcdef")
            self.assertEqual(parsed["path"], "")

            cells = parsed["cells"]
            self.assertEqual(len(cells), 4)
            self.assertEqual(cells[0]["cell_type"], "markdown")
            self.assertEqual(cells[0]["source"], ["# Section 1"])

            metadata = parsed["metadata"]
            self.assertEqual(metadata["kernelspec"]["display_name"], "Python 3")
            self.assertEqual(metadata["language_info"]["name"], "python")
        except json.JSONDecodeError as e:
            self.fail(f"Invalid JSON: {e}")


class TestNotebookMetadata(TestCase):
    def setUp(self):
        self.codemirror_mode = CodeMirrorMode(name="python", version=4)
        self.language_info = LanguageInfo(
            codemirror_mode=self.codemirror_mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.0",
        )
        self.kernelspec = KernelSpec(
            display_name="Python 3",
            language="python",
            name="python3",
        )
        self.metadata = NotebookMetadata(
            language_info=self.language_info,
            kernelspec=self.kernelspec,
        )

    def test_creation(self):
        self.assertEqual(self.metadata.language_info, self.language_info)
        self.assertEqual(self.metadata.kernelspec, self.kernelspec)

    def test_to_dict(self):
        expected = {
            "kernelspec": self.kernelspec.to_dict(),
            "language_info": self.language_info.to_dict(),
        }
        self.assertEqual(self.metadata.to_dict(), expected)


class TestKernelSpec(TestCase):
    def setUp(self):
        self.kernelspec = KernelSpec(
            display_name="Python 3",
            language="python",
            name="python3",
        )

    def test_creation(self):
        self.assertEqual(self.kernelspec.display_name, "Python 3")
        self.assertEqual(self.kernelspec.language, "python")
        self.assertEqual(self.kernelspec.name, "python3")

    def test_to_dict(self):
        expected = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        self.assertEqual(self.kernelspec.to_dict(), expected)


class TestLanguageInfo(TestCase):
    def setUp(self):
        self.codemirror_mode = CodeMirrorMode(name="python", version=4)
        self.language_info = LanguageInfo(
            codemirror_mode=self.codemirror_mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.0",
        )

    def test_creation(self):
        self.assertEqual(self.language_info.codemirror_mode, self.codemirror_mode)
        self.assertEqual(self.language_info.file_extension, ".py")
        self.assertEqual(self.language_info.mimetype, "text/x-python")
        self.assertEqual(self.language_info.name, "python")
        self.assertEqual(self.language_info.nbconvert_exporter, "python")
        self.assertEqual(self.language_info.pygments_lexer, "ipython3")
        self.assertEqual(self.language_info.version, "3.8.0")

    def test_to_dict(self):
        expected = {
            "codemirror_mode": self.codemirror_mode.to_dict(),
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0",
        }
        self.assertEqual(self.language_info.to_dict(), expected)


class TestCodeMirrorMode(TestCase):
    def setUp(self):
        self.codemirror_mode = CodeMirrorMode(name="python", version=4)

    def test_creation(self):
        self.assertEqual(self.codemirror_mode.name, "python")
        self.assertEqual(self.codemirror_mode.version, 4)

    def test_to_dict(self):
        expected = {
            "name": "python",
            "version": 4,
        }
        self.assertEqual(self.codemirror_mode.to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
