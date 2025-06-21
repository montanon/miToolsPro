from pathlib import Path
from typing import Dict, List, Optional

from pandas import DataFrame

from mitoolspro.regressions.linear_models import QuantileRegressionModel
from mitoolspro.regressions.wrappers.linear_models import (
    QuantilesRegressionResults,
    QuantilesRegressionSpecs,
)
from mitoolspro.utils.objects import StringMapper


class QuantilesRegressionManager:
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: str,
        independent_variables: List[str],
        control_variables: Optional[List[str]],
        quantiles: List[float],
        *,
        quadratic: Optional[bool] = False,
        str_mapper: Optional[StringMapper] = None,
        group_col: Optional[str] = None,
        groups: Optional[List[str]] = None,
        output_folder: Optional[Path] = None,
        all_groups_label: Optional[str] = "All",
        max_iter: Optional[int] = 5_000,
        recalculate: Optional[bool] = False,
    ):
        self.data = data
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.control_variables = control_variables or []
        self.quantiles = quantiles
        self.groups = groups or [all_groups_label]
        self.group_col = group_col
        self.str_mapper = str_mapper
        self.output_folder = output_folder
        self.quadratic = quadratic
        self.all_groups_label = all_groups_label
        self.recalculate = recalculate
        self.max_iter = max_iter
        self.model_cls = QuantileRegressionModel
        self.specs_cls = QuantilesRegressionSpecs
        self.result_cls = QuantilesRegressionResults

        self.models: Dict[str, QuantileRegressionModel] = {}
        self.results: Dict[str, QuantilesRegressionResults] = {}
        self.specs: Dict[str, QuantilesRegressionSpecs] = {}

    def run(self):
        for group in self.groups:
            group_data = self._get_group_data(group)
            self.specs[group] = self._create_specs(group, group_data)
            self.models[group] = self._create_model(self.specs[group])
            self.results[group] = self.models[group].fit(max_iter=self.max_iter)

    def _get_group_data(self, group: str) -> DataFrame:
        if group == self.all_groups_label or self.group_col is None:
            return self.data.copy(deep=True)
        return self.data[self.data[self.group_col] == group].copy(deep=True)

    def _create_specs(
        self, group: str, group_data: DataFrame
    ) -> QuantilesRegressionSpecs:
        return self.specs_cls(
            dependent_variable=self.dependent_variable,
            independent_variables=self.independent_variables,
            quantiles=self.quantiles,
            quadratic=self.quadratic,
            regression_type="quadratic" if self.quadratic else "linear",
            control_variables=self.control_variables,
            data=group_data,
            group=group,
        )

    def _create_model(
        self, specs: QuantilesRegressionSpecs, *args, **kwargs
    ) -> QuantileRegressionModel:
        return self.model_cls(
            data=specs.data,
            formula=specs.formula,
            dependent_variable=specs.dependent_variable,
            independent_variables=specs.independent_variables,
            control_variables=specs.control_variables,
            quantiles=self.quantiles,
            *args,
            **kwargs,
        )

    def get_result(self, group: str):
        return self.results[group]
