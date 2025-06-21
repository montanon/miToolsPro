import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from mitoolspro.utils.objects import StringMapper


@dataclass(frozen=True)
class QuantileRegressionStrs:
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


class QuantilesRegressionSpecs:
    def __init__(
        self,
        dependent_variable: str,
        independent_variables: List[str],
        quantiles: List[float],
        quadratic: bool,
        regression_type: str,
        data: DataFrame,
        group: Optional[str] = None,
        control_variables: Optional[List[str]] = None,
    ):
        self.dependent_variable = dependent_variable
        self.independent_variables = independent_variables
        self.quadratic = quadratic
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
        self.regression_type = regression_type
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
            + "\end{adjustbox}\n"
        )
        return table_text

    def store(self, folder_path: Path):
        self.data = None
        with open(folder_path / f"{self.regression_id}.reg_specs", "wb") as file:
            pickle.dump(self, file)


def regex_symbol_replacement(match):
    return rf"\{match.group(0)}"


def create_regression_id(
    regression_type: str,
    regression_degree: str,
    regression_dependent_var: str,
    regression_indep_vars: List[str],
    control_variables: List[str],
    id_len: Optional[int] = 6,
) -> str:
    str_to_hash = " ".join(
        [
            regression_type,
            regression_degree if regression_degree else "linear",
        ]
    )
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{str_to_hash}".encode("utf-8"))
    kind_id = id_hasher.hexdigest()[:id_len]

    id_hasher = hashlib.md5()
    id_hasher.update(rf"{regression_dependent_var}".encode("utf-8"))
    dep_id = id_hasher.hexdigest()[:id_len]

    str_to_hash = " ".join([v for v in regression_indep_vars if "_square" not in v])
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{str_to_hash}".encode("utf-8"))
    indep_id = id_hasher.hexdigest()[:id_len]

    control_vars_str = " ".join([v for v in control_variables])
    id_hasher = hashlib.md5()
    id_hasher.update(rf"{control_vars_str}".encode("utf-8"))
    control_vars_id = id_hasher.hexdigest()[:id_len] if control_variables else "None"
    return f"{kind_id}-{dep_id}-{indep_id}-{control_vars_id}"
