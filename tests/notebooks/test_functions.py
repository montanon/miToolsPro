import json
import os
import tempfile
from unittest import TestCase

import nbformat
from nbformat.notebooknode import NotebookNode

from mitoolspro.notebooks.functions import (
    clear_notebook_output,
    create_code_mirror_mode,
    create_kernel_spec,
    create_language_info,
    create_notebook,
    create_notebook_cell,
    create_notebook_metadata,
    create_notebook_section,
    create_notebook_sections,
    notebooknode_to_custom_notebook,
    read_notebook,
    validate_notebook,
    write_notebook,
)
from mitoolspro.notebooks.objects import (
    CodeCell,
    CodeMirrorMode,
    KernelSpec,
    LanguageInfo,
    MarkdownCell,
    Notebook,
    NotebookCells,
    NotebookMetadata,
    NotebookSection,
    NotebookSections,
)


class TestReadNotebook(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, "test.ipynb")

        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# Test"),
            nbformat.v4.new_code_cell("print('hello')"),
        ]
        with open(self.notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    def tearDown(self):
        if os.path.exists(self.notebook_path):
            os.remove(self.notebook_path)
        os.rmdir(self.temp_dir)

    def test_read_valid_notebook(self):
        notebook = read_notebook(self.notebook_path)
        self.assertIsInstance(notebook, Notebook)
        self.assertEqual(len(notebook.cells), 2)
        self.assertEqual(notebook.cells[0].cell_type, "markdown")
        self.assertEqual(notebook.cells[1].cell_type, "code")

    def test_read_nonexistent_notebook(self):
        with self.assertRaises(FileNotFoundError):
            read_notebook("nonexistent.ipynb")

    def test_read_invalid_notebook(self):
        with open(self.notebook_path, "w", encoding="utf-8") as f:
            f.write("invalid json")
        with self.assertRaises(ValueError):
            read_notebook(self.notebook_path)


class TestWriteNotebook(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.notebook_path = os.path.join(self.temp_dir, "test.ipynb")

        self.notebook = Notebook(
            cells=NotebookCells(
                [
                    MarkdownCell(cell_type="markdown", source=["# Test"]),
                    CodeCell(cell_type="code", source=["print('hello')"]),
                ]
            ),
            metadata=NotebookMetadata(
                kernelspec=KernelSpec(
                    display_name="Python 3", language="python", name="python3"
                ),
                language_info=LanguageInfo(
                    codemirror_mode=CodeMirrorMode(name="python", version=4),
                    file_extension=".py",
                    mimetype="text/x-python",
                    name="python",
                    nbconvert_exporter="python",
                    pygments_lexer="ipython3",
                    version="3.8.0",
                ),
            ),
            nbformat=4,
            nbformat_minor=5,
            name="Test Notebook",
        )

    def tearDown(self):
        if os.path.exists(self.notebook_path):
            os.remove(self.notebook_path)
        os.rmdir(self.temp_dir)

    def test_write_valid_notebook(self):
        write_notebook(self.notebook, self.notebook_path)
        self.assertTrue(os.path.exists(self.notebook_path))

        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        self.assertEqual(len(nb.cells), 2)
        self.assertEqual(nb.cells[0].cell_type, "markdown")
        self.assertEqual(nb.cells[1].cell_type, "code")

        # Verify custom properties are not in the written file
        nb_dict = nbformat.read(self.notebook_path, as_version=4)
        self.assertNotIn("_section_indices", nb_dict)
        self.assertNotIn("name", nb_dict)
        self.assertNotIn("notebook_id", nb_dict)
        self.assertNotIn("path", nb_dict)

    def test_write_nonexistent_directory(self):
        with self.assertRaises(FileNotFoundError):
            write_notebook(self.notebook, "/nonexistent/path/test.ipynb")

    def test_write_invalid_notebook(self):
        invalid_notebook = Notebook(
            cells=NotebookCells([]),
            metadata=NotebookMetadata(
                kernelspec=KernelSpec(display_name="", language="", name=""),
                language_info=LanguageInfo(
                    codemirror_mode=CodeMirrorMode(name="", version=4),
                    file_extension="",
                    mimetype="",
                    name="",
                    nbconvert_exporter="",
                    pygments_lexer="",
                    version="",
                ),
            ),
            nbformat=3,  # Invalid nbformat version
            nbformat_minor=5,
            name="Test Notebook",
        )
        with self.assertRaises(Exception):
            write_notebook(invalid_notebook, self.notebook_path)


class TestValidateNotebook(TestCase):
    def setUp(self):
        self.valid_notebook = Notebook(
            cells=NotebookCells(
                [
                    MarkdownCell(cell_type="markdown", source=["# Test"]),
                    CodeCell(cell_type="code", source=["print('hello')"]),
                ]
            ),
            metadata=NotebookMetadata(
                kernelspec=KernelSpec(
                    display_name="Python 3", language="python", name="python3"
                ),
                language_info=LanguageInfo(
                    codemirror_mode=CodeMirrorMode(name="python", version=4),
                    file_extension=".py",
                    mimetype="text/x-python",
                    name="python",
                    nbconvert_exporter="python",
                    pygments_lexer="ipython3",
                    version="3.8.0",
                ),
            ),
            nbformat=4,
            nbformat_minor=5,
            name="Test Notebook",
        )

    def test_validate_valid_notebook(self):
        validate_notebook(self.valid_notebook)

    def test_validate_invalid_notebook(self):
        invalid_notebook = Notebook(
            cells=NotebookCells([]),
            metadata=NotebookMetadata(
                kernelspec=KernelSpec(display_name="", language="", name=""),
                language_info=LanguageInfo(
                    codemirror_mode=CodeMirrorMode(name="", version=4),
                    file_extension="",
                    mimetype="",
                    name="",
                    nbconvert_exporter="",
                    pygments_lexer="",
                    version="",
                ),
            ),
            nbformat=3,  # Invalid nbformat version
            nbformat_minor=5,
            name="Test Notebook",
        )
        with self.assertRaises(Exception):
            validate_notebook(invalid_notebook)


class TestNotebookNodeToCustomNotebook(TestCase):
    def setUp(self):
        self.nb_node = NotebookNode(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Test"]},
                    {"cell_type": "code", "source": ["print('hello')"]},
                ],
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    },
                    "language_info": {
                        "codemirror_mode": {"name": "python", "version": 4},
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.0",
                    },
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )

    def test_conversion_valid_notebook(self):
        notebook = notebooknode_to_custom_notebook(self.nb_node)
        self.assertIsInstance(notebook, Notebook)
        self.assertEqual(len(notebook.cells), 2)
        self.assertEqual(notebook.cells[0].cell_type, "markdown")
        self.assertEqual(notebook.cells[1].cell_type, "code")
        self.assertEqual(notebook.metadata.kernelspec.display_name, "Python 3")
        self.assertEqual(notebook.metadata.language_info.name, "python")

    def test_conversion_invalid_notebook(self):
        invalid_node = NotebookNode(
            {
                "cells": [{"invalid": "cell"}],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
        with self.assertRaises(ValueError):
            notebooknode_to_custom_notebook(invalid_node)


class TestCreateNotebook(TestCase):
    def setUp(self):
        self.metadata = NotebookMetadata(
            kernelspec=KernelSpec(
                display_name="Python 3", language="python", name="python3"
            ),
            language_info=LanguageInfo(
                codemirror_mode=CodeMirrorMode(name="python", version=4),
                file_extension=".py",
                mimetype="text/x-python",
                name="python",
                nbconvert_exporter="python",
                pygments_lexer="ipython3",
                version="3.8.0",
            ),
        )
        self.cells = [
            MarkdownCell(cell_type="markdown", source=["# Test"]),
            CodeCell(cell_type="code", source=["print('hello')"]),
        ]

    def test_create_from_cells(self):
        notebook = create_notebook(
            cells=self.cells,
            metadata=self.metadata,
            nbformat=4,
            nbformat_minor=5,
        )
        self.assertIsInstance(notebook, Notebook)
        self.assertEqual(len(notebook.cells), 2)
        self.assertEqual(notebook._section_indices, [])

    def test_create_from_sections(self):
        section = NotebookSection(
            cells=NotebookCells(
                [
                    MarkdownCell(cell_type="markdown", source=["# Section"]),
                    CodeCell(cell_type="code", source=["print('section')"]),
                ]
            )
        )
        notebook = create_notebook(
            cells=[section],
            metadata=self.metadata,
            nbformat=4,
            nbformat_minor=5,
        )
        self.assertIsInstance(notebook, Notebook)
        self.assertEqual(len(notebook.cells), 2)
        self.assertEqual(notebook._section_indices, [(0, 2)])

    def test_create_from_sections_list(self):
        sections = NotebookSections(
            sections=[
                NotebookSection(
                    cells=NotebookCells(
                        [
                            MarkdownCell(cell_type="markdown", source=["# Section 1"]),
                            CodeCell(cell_type="code", source=["print('section 1')"]),
                        ]
                    )
                ),
                NotebookSection(
                    cells=NotebookCells(
                        [
                            MarkdownCell(cell_type="markdown", source=["# Section 2"]),
                            CodeCell(cell_type="code", source=["print('section 2')"]),
                        ]
                    )
                ),
            ]
        )
        notebook = create_notebook(
            cells=[sections],
            metadata=self.metadata,
            nbformat=4,
            nbformat_minor=5,
        )
        self.assertIsInstance(notebook, Notebook)
        self.assertEqual(len(notebook.cells), 4)
        self.assertEqual(notebook._section_indices, [(0, 2), (2, 4)])


class TestCreateNotebookMetadata(TestCase):
    def setUp(self):
        self.language_info = LanguageInfo(
            codemirror_mode=CodeMirrorMode(name="python", version=4),
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

    def test_create_with_kernelspec(self):
        metadata = create_notebook_metadata(
            language_info=self.language_info,
            kernelspec=self.kernelspec,
        )
        self.assertIsInstance(metadata, NotebookMetadata)
        self.assertEqual(metadata.kernelspec, self.kernelspec)
        self.assertEqual(metadata.language_info, self.language_info)

    def test_create_without_kernelspec(self):
        metadata = create_notebook_metadata(language_info=self.language_info)
        self.assertIsInstance(metadata, NotebookMetadata)
        self.assertIsNone(metadata.kernelspec)
        self.assertEqual(metadata.language_info, self.language_info)


class TestCreateNotebookSection(TestCase):
    def setUp(self):
        self.cells = [
            CodeCell(cell_type="code", source=["print('hello')"]),
        ]
        self.notebook_seed = "test_notebook"
        self.section_seed = "test_section"

    def test_create_section(self):
        section = create_notebook_section(
            title="# Test Section",
            cells=self.cells,
            notebook_seed=self.notebook_seed,
            section_seed=self.section_seed,
        )
        self.assertIsInstance(section, NotebookSection)
        self.assertEqual(len(section.cells), 2)  # Title cell + code cell
        self.assertEqual(section.cells[0].cell_type, "markdown")
        self.assertEqual(section.cells[0].source, ["# Test Section"])
        self.assertEqual(section.cells[1].cell_type, "code")
        self.assertEqual(section.cells[1].source, ["print('hello')"])


class TestCreateNotebookSections(TestCase):
    def setUp(self):
        self.sections = [
            (
                "# Section 1",
                [
                    CodeCell(cell_type="code", source=["print('section 1')"]),
                ],
            ),
            (
                "# Section 2",
                [
                    CodeCell(cell_type="code", source=["print('section 2')"]),
                ],
            ),
        ]
        self.notebook_seed = "test_notebook"

    def test_create_sections(self):
        sections = create_notebook_sections(
            sections=self.sections,
            notebook_seed=self.notebook_seed,
        )
        self.assertIsInstance(sections, NotebookSections)
        self.assertEqual(len(sections.sections), 2)
        self.assertEqual(sections.sections[0].cells[0].source, ["# Section 1"])
        self.assertEqual(sections.sections[1].cells[0].source, ["# Section 2"])


class TestCreateNotebookCell(TestCase):
    def setUp(self):
        self.notebook_seed = "test_notebook"
        self.cell_seed = "test_cell"

    def test_create_markdown_cell(self):
        cell = create_notebook_cell(
            cell_type="markdown",
            execution_count=None,
            notebook_seed=self.notebook_seed,
            cell_seed=self.cell_seed,
            metadata={},
            outputs=[],
            source=["# Test"],
        )
        self.assertIsInstance(cell, MarkdownCell)
        self.assertEqual(cell.cell_type, "markdown")
        self.assertEqual(cell.source, ["# Test"])

    def test_create_code_cell(self):
        cell = create_notebook_cell(
            cell_type="code",
            execution_count=1,
            notebook_seed=self.notebook_seed,
            cell_seed=self.cell_seed,
            metadata={},
            outputs=[],
            source=["print('hello')"],
        )
        self.assertIsInstance(cell, CodeCell)
        self.assertEqual(cell.cell_type, "code")
        self.assertEqual(cell.source, ["print('hello')"])
        self.assertEqual(cell.execution_count, 1)


class TestCreateCodeMirrorMode(TestCase):
    def test_create_code_mirror_mode(self):
        mode = create_code_mirror_mode(name="python", version=4)
        self.assertIsInstance(mode, CodeMirrorMode)
        self.assertEqual(mode.name, "python")
        self.assertEqual(mode.version, 4)


class TestCreateLanguageInfo(TestCase):
    def setUp(self):
        self.codemirror_mode = CodeMirrorMode(name="python", version=4)

    def test_create_language_info(self):
        language_info = create_language_info(
            codemirror_mode=self.codemirror_mode,
            file_extension=".py",
            mimetype="text/x-python",
            name="python",
            nbconvert_exporter="python",
            pygments_lexer="ipython3",
            version="3.8.0",
        )
        self.assertIsInstance(language_info, LanguageInfo)
        self.assertEqual(language_info.codemirror_mode, self.codemirror_mode)
        self.assertEqual(language_info.file_extension, ".py")
        self.assertEqual(language_info.mimetype, "text/x-python")
        self.assertEqual(language_info.name, "python")
        self.assertEqual(language_info.nbconvert_exporter, "python")
        self.assertEqual(language_info.pygments_lexer, "ipython3")
        self.assertEqual(language_info.version, "3.8.0")


class TestCreateKernelSpec(TestCase):
    def test_create_kernel_spec(self):
        kernel_spec = create_kernel_spec(
            display_name="Python 3",
            language="python",
            name="python3",
        )
        self.assertIsInstance(kernel_spec, KernelSpec)
        self.assertEqual(kernel_spec.display_name, "Python 3")
        self.assertEqual(kernel_spec.language, "python")
        self.assertEqual(kernel_spec.name, "python3")


class TestClearNotebookOutput(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.temp_dir, "input.ipynb")
        self.output_path = os.path.join(self.temp_dir, "output.ipynb")

        # Create a test notebook with outputs
        nb = nbformat.v4.new_notebook()
        cell = nbformat.v4.new_code_cell("print('hello')")
        cell.outputs = [nbformat.v4.new_output("stream", text="hello\n")]
        nb.cells = [cell]
        with open(self.input_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    def tearDown(self):
        for path in [self.input_path, self.output_path]:
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self.temp_dir)

    def test_clear_notebook_output(self):
        clear_notebook_output(self.input_path, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))

        # Verify outputs are cleared
        with open(self.output_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        self.assertEqual(len(nb.cells[0].outputs), 0)

    def test_clear_nonexistent_notebook(self):
        with self.assertRaises(FileNotFoundError):
            clear_notebook_output("nonexistent.ipynb", self.output_path)
