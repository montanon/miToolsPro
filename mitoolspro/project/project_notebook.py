from mitoolspro.notebooks import (
    Notebook,
    NotebookSections,
    create_code_mirror_mode,
    create_kernel_spec,
    create_language_info,
    create_notebook,
    create_notebook_cell,
    create_notebook_metadata,
    create_notebook_section,
)


class ProjectNotebook:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.notebook_seed = f"{self.project_name}_notebook"
        self.metadata = {}

        self.imports_source = [
            "import mitoolspro as mtp",
            "from mitoolspro.project import Project",
        ]
        self._imports_cell = create_notebook_cell(
            cell_type="import",
            notebook_seed=self.notebook_seed,
            cell_seed="imports",
            source=self.imports_source,
            deletable=False,
            editable=False,
        )

        self._load_source = ["pr = Project.load(auto_load=True)", "pr.project_tree()"]
        self._load_cell = create_notebook_cell(
            cell_type="code",
            notebook_seed=self.notebook_seed,
            cell_seed="load_project",
            source=self._load_source,
            deletable=False,
            editable=False,
        )
        self._load_section = create_notebook_section(
            title=f"# Project: {self.project_name.title()}",
            cells=[self._load_cell],
            notebook_seed=self.notebook_seed,
            section_seed="load_section",
        )

        self._clean_cell = create_notebook_cell(
            cell_type="code",
            notebook_seed=self.notebook_seed,
            cell_seed="clean_project",
            source=[],
        )

        self._closure_source = ["***"]
        self._closure = create_notebook_cell(
            cell_type="markdown",
            notebook_seed=self.notebook_seed,
            cell_seed="closure",
            source=self._closure_source,
            deletable=False,
            editable=False,
        )

    def notebook(self) -> Notebook:
        return Notebook(
            cells=[
                self._imports_cell,
                self._load_section,
            ]
        )
