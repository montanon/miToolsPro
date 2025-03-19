from os import PathLike

from mitoolspro.notebooks import (
    Notebook,
    create_default_metadata,
    create_notebook,
    create_notebook_cell,
    create_notebook_section,
    write_notebook,
)


class ProjectNotebook:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.notebook_seed = f"{self.project_name}_notebook"
        self.metadata = create_default_metadata()

        self.imports_source = [
            "from mitoolspro.project import Project",
        ]
        self._imports_cell = create_notebook_cell(
            cell_type="import",
            notebook_seed=self.notebook_seed,
            cell_seed="imports",
            source=self.imports_source,
        )

        self._load_source = ["pr = Project.load(auto_load=True)", "pr.project_tree()"]
        self._load_cell = create_notebook_cell(
            cell_type="code",
            notebook_seed=self.notebook_seed,
            cell_seed="load_project",
            source=self._load_source,
        )
        self._load_section = create_notebook_section(
            title=f"# Project: {self.project_name.title()}",
            cells=[self._load_cell],
            notebook_seed=self.notebook_seed,
            section_seed="load_section",
        )

        self._add_path_source = [
            "# pr.add_path(key: str, path: PathLike, update=True)",
            "",
        ]
        self._add_var_source = ["# pr.add_var(key: str, values: Any, update=True)", ""]
        self._add_section = create_notebook_section(
            title="## Add Paths and Vars to your Project",
            cells=[
                create_notebook_cell(
                    cell_type="markdown",
                    source="### Add Paths",
                    notebook_seed=self.notebook_seed,
                    cell_seed="add_path_title",
                ),
                create_notebook_cell(
                    cell_type="code",
                    source=self._add_path_source,
                    notebook_seed=self.notebook_seed,
                    cell_seed="add_path_source",
                ),
                create_notebook_cell(
                    cell_type="markdown",
                    source="### Add Vars",
                    notebook_seed=self.notebook_seed,
                    cell_seed="add_var_title",
                ),
                create_notebook_cell(
                    cell_type="code",
                    source=self._add_var_source,
                    notebook_seed=self.notebook_seed,
                    cell_seed="add_var_source",
                ),
            ],
            notebook_seed=self.notebook_seed,
            section_seed="add_section",
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
        )

    def notebook(self) -> Notebook:
        return create_notebook(
            cells=[
                self._imports_cell,
                self._load_section,
                self._add_section,
                self._clean_cell,
                self._closure,
            ],
            metadata=self.metadata,
            name=self.project_name,
            notebook_id=self.notebook_seed,
        )

    def write(self, path: PathLike) -> None:
        write_notebook(self.notebook(), path)


if __name__ == "__main__":
    pn = ProjectNotebook("test")
    pn.write("test.ipynb")
