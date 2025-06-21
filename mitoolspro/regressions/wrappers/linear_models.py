import pickle
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from mitoolspro.regressions.wrappers.base import (
    BaseRegressionSpecs,
    BaseRegressionStrs,
)
from mitoolspro.regressions.wrappers.utils import (
    create_regression_id,
    prettify_index_level,
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


class QuantilesRegression:
    def __init__(self, coeffs, stats):
        self.coeffs = coeffs
        self.stats = stats

        self.id = self.coeffs.index.get_level_values(
            QuantileRegressionStrs.ID
        ).tolist()[0]
        self.group = self.coeffs.columns.tolist()[0]

        self.dependent_variables = self.coeffs.index.get_level_values(
            QuantileRegressionStrs.DEPENDENT_VAR
        ).tolist()[0]

        self.independent_variables = (
            self.coeffs.loc[
                self.coeffs.index.get_level_values(QuantileRegressionStrs.VARIABLE_TYPE)
                == QuantileRegressionStrs.EXOG_VAR
            ]
            .index.get_level_values(QuantileRegressionStrs.INDEPENDENT_VARS)
            .unique()
            .tolist()
        )
        self.control_variables = (
            self.coeffs.loc[
                self.coeffs.index.get_level_values(QuantileRegressionStrs.VARIABLE_TYPE)
                == QuantileRegressionStrs.CONTROL_VAR
            ]
            .index.get_level_values(QuantileRegressionStrs.INDEPENDENT_VARS)
            .unique()
            .tolist()
        )

        self.quantiles = (
            self.coeffs.index.get_level_values(QuantileRegressionStrs.QUANTILE)
            .unique()
            .tolist()
        )
        self.quadratic = (
            self.coeffs.index.get_level_values(
                QuantileRegressionStrs.REGRESSION_DEGREE
            ).tolist()[0]
            == QuantileRegressionStrs.QUADRATIC_REG
        )
        self.regression_type = self.coeffs.index.get_level_values(
            QuantileRegressionStrs.REGRESSION_TYPE
        ).tolist()[0]

    def coefficients(self, quantiles: Optional[List[float]] = None):
        if quantiles is None:
            return self.coeffs
        return self.coeffs.loc[
            self.coeffs.index.get_level_values(QuantileRegressionStrs.QUANTILE).isin(
                quantiles
            )
        ]

    def n_obs(self, quantiles: Optional[List[float]] = None):
        if quantiles is None:
            stats = self.stats.loc[
                (slice(None), QuantileRegressionStrs.N_OBSERVATIONS), :
            ]
        else:
            stats = self.stats.loc[
                (quantiles, QuantileRegressionStrs.N_OBSERVATIONS), :
            ]
        stats.index = stats.index.droplevel(QuantileRegressionStrs.STATS)
        stats.columns = [QuantileRegressionStrs.N_OBSERVATIONS]
        return stats

    def r_squared(self, quantiles: Optional[List[float]] = None):
        if quantiles is None:
            stats = self.stats.loc[
                (slice(None), QuantileRegressionStrs.PSEUDO_R_SQUARED), :
            ]
        else:
            stats = self.stats.loc[
                (quantiles, QuantileRegressionStrs.PSEUDO_R_SQUARED), :
            ]
        stats.index = stats.index.droplevel(QuantileRegressionStrs.STATS)
        stats.columns = [QuantileRegressionStrs.PSEUDO_R_SQUARED]
        return stats

    def coefficients_quantiles_table(self, quantiles: Optional[List[float]] = None):
        table = self.coeffs.unstack(level=QuantileRegressionStrs.QUANTILE)
        if quantiles is not None:
            table = table.loc[:, (slice(None), quantiles)]
        return table.sort_index(
            axis=0,
            level=[
                QuantileRegressionStrs.VARIABLE_TYPE,
                QuantileRegressionStrs.INDEPENDENT_VARS,
            ],
            ascending=[False, True],
        )

    def coefficients_quantiles_latex_table(
        self,
        quantiles: Optional[List[float]] = None,
        note: Optional[bool] = False,
        str_mapper: Optional[StringMapper] = None,
    ):
        table = self.coefficients_quantiles_table(quantiles).droplevel(
            [
                QuantileRegressionStrs.ID,
                QuantileRegressionStrs.REGRESSION_TYPE,
                QuantileRegressionStrs.REGRESSION_DEGREE,
                QuantileRegressionStrs.VARIABLE_TYPE,
            ],
            axis=0,
        )
        if str_mapper is not None:
            levels_to_remap = [
                QuantileRegressionStrs.DEPENDENT_VAR,
                QuantileRegressionStrs.INDEPENDENT_VARS,
            ]
            pretty_index = table.index.set_levels(
                [
                    prettify_index_level(
                        str_mapper,
                        QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX,
                        level,
                        level_id,
                        levels_to_remap,
                    )
                    for level, level_id in zip(table.index.levels, table.index.names)
                ],
                level=table.index.names,
            )
            table.index = pretty_index
        symbols_pattern = r"([\ \_\-\&\%\$\#])"
        table = table.rename(
            columns=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x)
            if isinstance(x, str)
            else str(round(x, 1)),
            index=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x)
            if isinstance(x, str)
            else str(round(x, 1)),
        ).to_latex(multirow=True, multicolumn=True, multicolumn_format="c")
        table_text = (
            "\\begin{adjustbox}{width=\\textwidth,center}\n"
            + f"{table}"
            + "\end{adjustbox}\n"
        )
        table_text = (
            table_text
            + "{\\centering\\tiny Note: * p\\textless0.05, ** p\\textless0.01, *** p\\textless0.001\\par}"
            if note
            else table_text
        )
        print(table_text)

    def model_specification(self, str_mapper: Optional[StringMapper] = None):
        if str_mapper:
            independent_variables = [
                str_mapper.prettify_str(var)
                if QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX not in var
                else f"{str_mapper.prettify_str(var.replace(QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX, ''))}{QuantileRegressionStrs.QUADRATIC_VAR_SUFFIX}"
                for var in self.independent_variables
            ]
            control_variables = [
                str_mapper.prettify_str(var) for var in self.control_variables
            ]
        else:
            independent_variables = self.independent_variables
            control_variables = self.control_variables
        model_specification = f"{self.dependent_variables if not str_mapper else str_mapper.prettify_str(self.dependent_variables)}"
        model_specification += f" ~ {' + '.join(independent_variables)}"
        model_specification += (
            f" + {' + '.join([var for var in control_variables if var != 'Intercept'])}"
            if control_variables
            else ""
        )
        model_specification = model_specification.split(" + ")
        lines = []
        line = ""
        for string in model_specification[:-1]:
            if len(line) + len(string) < 120:
                line += f"{string} + "
            else:
                lines.append(line + r"\\")
                line = string + " + "
        lines.append(model_specification[-1])
        model_specification = "".join(lines)
        symbols_pattern = r"([\ \_\-\&\%\$\#])"
        model_specification = re.sub(
            symbols_pattern, regex_symbol_replacement, model_specification
        ).replace("~", "\\sim")
        print(f"${model_specification}$")

    def abstract_model_specification(self):
        pass

    def quantile_model_equation(self):
        print(
            "$\\min_{\\beta} \\sum_{i:y_g \\geq x_g^T\\beta} q |y_g - x_g^T\\beta| + \\sum_{g:y_g < x_g^T\\beta} (1-q) |y_g - x_g^T\\beta|$"
        )

    def store(self, folder_path: Path):
        with open(folder_path / f"{self.id}.reg_coeffs", "wb") as file:
            pickle.dump(self, file)
