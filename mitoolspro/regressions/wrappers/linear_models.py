import pickle
import re
from io import StringIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandas import DataFrame, Series
from statsmodels.regression.quantile_regression import RegressionResultsWrapper

from mitoolspro.regressions.wrappers.base import (
    BaseRegressionResult,
    BaseRegressionSpecs,
    BaseRegressionStrs,
)
from mitoolspro.regressions.wrappers.utils import (
    create_regression_id,
    regex_symbol_replacement,
)
from mitoolspro.utils.objects import StringMapper


class QuantileRegressionStrs(BaseRegressionStrs):
    UNNAMED: str = "Unnamed: 0"
    COEF: str = "coef"
    T_VALUE: str = "t"
    P_VALUE: str = "P>|t|"
    VALUE: str = "Value"
    QUANTILE: str = "Quantile"
    INDEPENDENT_VARS: str = "Independent Vars"
    REGRESSION_TYPE: str = "Regression Type"
    REGRESSION_DEGREE: str = "Regression Degree"
    DEPENDENT_VAR: str = "Dependent Var"
    VARIABLE_TYPE: str = "Variable Type"
    EXOG_VAR: str = "Exog"
    CONTROL_VAR: str = "Control"
    ID: str = "Id"
    QUADRATIC_REG: str = "quadratic"
    LINEAR_REG: str = "linear"
    QUADRATIC_VAR_SUFFIX: str = "_square"
    INDEPENDENT_VARS_PATTERN: str = r"^I\((.*)\)$"
    STATS: str = "Stats"
    INTERCEPT: str = "Intercept"
    ANNOTATION: str = "Q"
    PARQUET_SUFFIX: str = "regressions"
    EXCEL_SUFFIX: str = "regressions"
    MAIN_PLOT: str = "regression_data"
    PLOTS_SUFFIX: str = "regression"
    ADJ_METHOD: str = "Adj Method"
    DATE: str = "Date"
    TIME: str = "Time"
    PSEUDO_R_SQUARED: str = "Pseudo R-squared"
    BANDWIDTH: str = "Bandwidth"
    SPARSITY: str = "Sparsity"
    N_OBSERVATIONS: str = "N Observations"
    DF_RESIDUALS: str = "Df Residuals"
    DF_MODEL: str = "Df Model"
    KURTOSIS: str = "Kurtosis"
    SKEWNESS: str = "Skewness"


class QuantilesRegressionSpecs(BaseRegressionSpecs):
    def __init__(
        self,
        dependent_variable: str,
        independent_variables: List[str],
        quantiles: List[float],
        quadratic: bool,
        data: DataFrame,
        group: Optional[str] = None,
        control_variables: Optional[List[str]] = None,
    ):
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.quadratic = "quadratic" if quadratic else "linear"
        if self.quadratic and not any(
            [
                f"{var}{QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX}"
                in self.independent_variables
                for var in self.independent_variables
            ]
        ):
            self.independent_variables += [
                f"{var}{QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX}"
                for var in independent_variables
            ]
        self.independent_variables.sort()
        self.control_variables = control_variables or []
        self.control_variables.sort()
        self.variables = (
            [self.dependent_variable]
            + self.independent_variables
            + self.control_variables
        )
        self.quantiles = quantiles
        self.regression_type = "quantile"
        self.data = data
        self.regression_id = create_regression_id(
            self.regression_type,
            self.quadratic,
            self.dependent_variable,
            self.independent_variables,
            self.control_variables,
        )
        self.group = group
        self.formula = self.get_formula()

    def get_formula(self, str_mapper: Optional[StringMapper] = None) -> str:
        if str_mapper:
            independent_variables = str_mapper.prettify_strs(self.independent_variables)
            control_variables = str_mapper.prettify_strs(self.control_variables)
            dependent_variable = str_mapper.prettify_str(self.dependent_variable)
        else:
            independent_variables = self.independent_variables
            control_variables = self.control_variables
            dependent_variable = self.dependent_variable
        formula_terms = [
            var
            for var in independent_variables
            if QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX not in var
        ]
        formula_terms += [
            f"I({var})"
            for var in independent_variables
            if QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX in var
        ]
        if control_variables:
            formula_terms += control_variables
        formula = f"{dependent_variable} ~ " + " + ".join(formula_terms)
        return formula

    def get_id(self):
        return self.regression_id

    def data_statistics_table(self, str_mapper: Optional[StringMapper] = None):
        table = self.data[[self.variables]].describe(percentiles=[0.5]).T
        table.columns = [
            QuantileRegressionStrs.N_OBSERVATIONS,
            "Mean",
            "Std. Dev.",
            "Min",
            "Median",
            "Max",
        ]
        table[QuantileRegressionStrs.KURTOSIS] = self.data[[self.variables]].kurtosis()
        table[QuantileRegressionStrs.SKEWNESS] = self.data[[self.variables]].skew()
        table[QuantileRegressionStrs.N_OBSERVATIONS] = table[
            QuantileRegressionStrs.N_OBSERVATIONS
        ].astype(int)
        numeric_cols = [
            c for c in table.columns if c != QuantileRegressionStrs.N_OBSERVATIONS
        ]
        table[numeric_cols] = table[numeric_cols].round(7)
        table.columns = (
            pd.MultiIndex.from_product([[self.group], table.columns])
            if self.group
            else table.columns
        )
        if str_mapper:
            table.index = table.index.map(lambda x: str_mapper.prettify_str(x))
        return table.sort_index(ascending=True)

    def data_statistics_latex_table(self, str_mapper: Optional[StringMapper] = None):
        table = self.data_statistics_table(str_mapper)
        symbols_pattern = r"([\ \_\-\&\%\$\#])"
        table = table.rename(
            index=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x)
            if isinstance(x, str)
            else str(round(x, 1))
        )
        table_latex = table.to_latex(
            multirow=True, multicolumn=True, multicolumn_format="c"
        )
        table_text = (
            "\\begin{adjustbox}{width=\\textwidth,center}\n"
            + f"{table_latex}"
            + "\\end{adjustbox}\n"
        )
        return table_text

    def store(self, folder_path: Path):
        self.data = None
        with open(folder_path / f"{self.regression_id}.reg_specs", "wb") as file:
            pickle.dump(self, file)


class QuantilesRegressionResults(BaseRegressionResult):
    def __init__(
        self,
        results: dict[str, RegressionResultsWrapper],
        specs: QuantilesRegressionSpecs,
        t_values: Optional[bool] = True,
    ):
        self.results = results
        self.specs = specs
        self.t_values = t_values
        self.independent_variables = specs.independent_variables
        self.control_variables = specs.control_variables
        self.dependent_variable = specs.dependent_variable
        self.coefficients = self.get_coefficients()
        self.residuals = self.get_residuals()
        self.summary = self.get_summaries()
        self.stats = self.get_stats()

    def get_coefficients(
        self,
        t_values: Optional[bool] = None,
    ) -> DataFrame:
        if t_values is None:
            t_values = self.t_values
        regression_coeffs = self._process_result_wrappers_coeffs(self.results, t_values)

        regression_coeffs = self._melt_and_rename_regression_coeffs(regression_coeffs)
        regression_coeffs = self._update_regression_coeffs_independent_vars(
            regression_coeffs
        )
        regression_coeffs = self._set_regression_coeffs_info(
            regression_coeffs, self.results, self.independent_variables
        )
        regression_coeffs = self._classify_regression_coeffs_variables(
            regression_coeffs, self.independent_variables
        )
        regression_coeffs = self._sort_and_set_regression_coeffs_index(
            regression_coeffs
        )
        return regression_coeffs

    def get_residuals(self) -> dict[float, Series]:
        return {q: result.resid for q, result in self.results.items()}

    def get_summaries(self) -> None:
        pass

    def _process_result_wrappers_coeffs(
        self,
        results: dict[int, RegressionResultsWrapper],
        t_value: Optional[bool] = True,
    ) -> DataFrame:
        return pd.concat(
            [
                self._process_quantile_regression_coeffs_result(q, result, t_value)
                for q, result in results.items()
            ],
            axis=1,
        )

    def _process_quantile_regression_coeffs_result(
        self,
        q: float,
        result: RegressionResultsWrapper,
        t_values: Optional[bool] = True,
    ) -> DataFrame:
        coeffs = pd.concat(
            pd.read_html(StringIO(result.summary().tables[1].as_html()), header=0)
        )
        coeffs = coeffs.set_index(QuantileRegressionStrs.UNNAMED)
        coeffs[QuantileRegressionStrs.VALUE] = coeffs[
            [
                QuantileRegressionStrs.COEF,
                QuantileRegressionStrs.T_VALUE,
                QuantileRegressionStrs.P_VALUE,
            ]
        ].apply(self._quantile_regression_value, args=(t_values,), axis=1)
        coeffs = coeffs[[QuantileRegressionStrs.VALUE]]
        coeffs.columns = pd.MultiIndex.from_tuples(
            [(str(q), c) for c in coeffs.columns]
        )
        return coeffs

    def _quantile_regression_value(
        self, row: Series, t_values: Optional[bool] = True
    ) -> Series:
        coeff = round(row[QuantileRegressionStrs.COEF], 3)
        t_value = round(row[QuantileRegressionStrs.T_VALUE], 3)
        p_value = row[QuantileRegressionStrs.P_VALUE]
        if p_value <= 0.001:
            return f"{coeff}({t_value})***" if t_values else f"{coeff}***"
        elif p_value <= 0.01:
            return f"{coeff}({t_value})**" if t_values else f"{coeff}**"
        elif p_value <= 0.05:
            return f"{coeff}({t_value})*" if t_values else f"{coeff}*"
        else:
            return f"{coeff}({t_value})" if t_values else f"{coeff}"

    def _melt_and_rename_regression_coeffs(
        self, regression_coeffs: DataFrame
    ) -> DataFrame:
        regression_coeffs.columns = regression_coeffs.columns.get_level_values(0)
        regression_coeffs = regression_coeffs.melt(ignore_index=False).reset_index()
        regression_coeffs.columns = [
            QuantileRegressionStrs.INDEPENDENT_VARS,
            QuantileRegressionStrs.QUANTILE,
            QuantileRegressionStrs.VALUE,
        ]
        regression_coeffs[QuantileRegressionStrs.INDEPENDENT_VARS] = regression_coeffs[
            QuantileRegressionStrs.INDEPENDENT_VARS
        ].replace({"const": QuantileRegressionStrs.INTERCEPT})
        return regression_coeffs

    def _update_regression_coeffs_independent_vars(
        self,
        regression_coeffs: DataFrame,
    ) -> DataFrame:
        regression_coeffs[QuantileRegressionStrs.INDEPENDENT_VARS] = regression_coeffs[
            QuantileRegressionStrs.INDEPENDENT_VARS
        ].replace(QuantileRegressionStrs.INDEPENDENT_VARS_PATTERN, r"\1", regex=True)
        return regression_coeffs

    def _set_regression_coeffs_info(
        self,
        regression_coeffs: DataFrame,
        results: dict[int, RegressionResultsWrapper],
        independent_variables: List[str],
    ) -> DataFrame:
        regression_coeffs[QuantileRegressionStrs.REGRESSION_TYPE] = type(
            list(results.values())[0].model
        ).__name__
        reg_degree = (
            QuantileRegressionStrs.QUADRATIC_REG
            if all(
                f"{var}{QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX}"
                in regression_coeffs[QuantileRegressionStrs.INDEPENDENT_VARS].values
                for var in independent_variables
            )
            else QuantileRegressionStrs.LINEAR_REG
        )
        regression_coeffs[QuantileRegressionStrs.REGRESSION_DEGREE] = reg_degree
        regression_coeffs[QuantileRegressionStrs.DEPENDENT_VAR] = list(
            results.values()
        )[0].model.endog_names
        return regression_coeffs

    def _classify_regression_coeffs_variables(
        self, regression_coeffs: DataFrame, independent_variables: List[str]
    ) -> DataFrame:
        regression_coeffs[QuantileRegressionStrs.VARIABLE_TYPE] = regression_coeffs[
            QuantileRegressionStrs.INDEPENDENT_VARS
        ].apply(
            lambda x: QuantileRegressionStrs.EXOG_VAR
            if x.replace(QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX, "")
            in independent_variables
            else QuantileRegressionStrs.CONTROL_VAR
        )
        return regression_coeffs

    def _sort_and_set_regression_coeffs_index(
        self, regression_coeffs: DataFrame
    ) -> DataFrame:
        regression_coeffs = regression_coeffs.sort_values(
            by=[
                QuantileRegressionStrs.VARIABLE_TYPE,
                QuantileRegressionStrs.INDEPENDENT_VARS,
                QuantileRegressionStrs.QUANTILE,
            ],
            ascending=[False, True, True],
        )
        regression_coeffs[QuantileRegressionStrs.QUANTILE] = regression_coeffs[
            QuantileRegressionStrs.QUANTILE
        ].astype(float)
        regression_coeffs = regression_coeffs.set_index(
            [
                QuantileRegressionStrs.REGRESSION_TYPE,
                QuantileRegressionStrs.REGRESSION_DEGREE,
                QuantileRegressionStrs.DEPENDENT_VAR,
                QuantileRegressionStrs.VARIABLE_TYPE,
                QuantileRegressionStrs.INDEPENDENT_VARS,
                QuantileRegressionStrs.QUANTILE,
            ]
        )
        return regression_coeffs

    def get_stats(
        self,
    ) -> DataFrame:
        _stats_name_remap_dict = {
            "Dep. Variable:": QuantileRegressionStrs.DEPENDENT_VAR,
            "Model:": QuantileRegressionStrs.REGRESSION_TYPE,
            "Method:": QuantileRegressionStrs.ADJ_METHOD,
            "Date:": QuantileRegressionStrs.DATE,
            "Time:": QuantileRegressionStrs.TIME,
            "Pseudo R-squared:": QuantileRegressionStrs.PSEUDO_R_SQUARED,
            "Bandwidth:": QuantileRegressionStrs.BANDWIDTH,
            "Sparsity:": QuantileRegressionStrs.SPARSITY,
            "No. Observations:": QuantileRegressionStrs.N_OBSERVATIONS,
            "Df Residuals:": QuantileRegressionStrs.DF_RESIDUALS,
            "Df Model:": QuantileRegressionStrs.DF_MODEL,
        }
        regression_stats = []
        for q, result in self.results.items():
            stats = result.summary().tables[0].as_html()
            stats = pd.read_html(StringIO(stats), index_col=0)[0].reset_index()
            stats = pd.concat(
                [stats.iloc[:-1, :2], stats.iloc[:, 2:].rename(columns={2: 0, 3: 1})],
                axis=0,
                ignore_index=True,
            )
            stats.columns = [QuantileRegressionStrs.STATS, QuantileRegressionStrs.VALUE]
            stats[QuantileRegressionStrs.QUANTILE] = q
            stats = stats.set_index(
                [QuantileRegressionStrs.QUANTILE, QuantileRegressionStrs.STATS]
            )
            regression_stats.append(stats)
        regression_stats = pd.concat(regression_stats, axis=0)
        regression_stats.index = regression_stats.index.set_levels(
            regression_stats.index.levels[
                regression_stats.index.names.index(QuantileRegressionStrs.STATS)
            ].map(_stats_name_remap_dict.get),
            level=QuantileRegressionStrs.STATS,
        )
        return regression_stats
