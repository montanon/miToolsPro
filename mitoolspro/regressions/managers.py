import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

import pandas as pd

from mitoolspro.regressions.linear_models import QuantilesRegressionModel
from mitoolspro.regressions.wrappers.linear_models import (
    QuantileRegressionStrs,
    QuantilesRegressionResults,
    QuantilesRegressionSpecs,
)
from mitoolspro.regressions.wrappers.utils import (
    prettify_index_level,
    regex_symbol_replacement,
)
from mitoolspro.utils.objects import StringMapper


class QuantilesRegression:
    def __init__(
        self,
        specs: QuantilesRegressionSpecs,
        *,
        max_iter: int = 10_000,
        error_method: str = "robust",
        t_values: Optional[bool] = True,
        str_mapper: Optional[StringMapper] = None,
    ):
        self.specs = specs
        self.max_iter = max_iter
        self.error_method = error_method
        self.t_values = t_values
        self.str_mapper = str_mapper

        self.model = QuantilesRegressionModel(
            data=self.specs.data,
            dependent_variable=self.specs.dependent_variable,
            independent_variables=self.specs.independent_variables,
            control_variables=self.specs.control_variables,
            quantiles=self.specs.quantiles,
        )

        results = self.model.fit(max_iter=self.max_iter, error_method=self.error_method)
        self.results = QuantilesRegressionResults(
            results=results, specs=self.specs, t_values=self.t_values
        )
        self.coeffs = self._set_coefficients()
        self.stats = self._set_stats()
        self.residuals = self._set_residuals()

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

    def _set_coefficients(self):
        coefficients = self.results.coefficients.copy(deep=True)
        if self.specs.group is not None:
            coefficients.columns = [self.specs.group]
        coefficients["Id"] = self.specs.regression_id
        coefficients = coefficients.set_index("Id", append=True)
        coefficients = coefficients.reorder_levels([-1, 0, 1, 2, 3, 4, 5])
        coefficients.index = coefficients.index.map(
            lambda x: (
                x[0],
                x[1],
                x[2],
                self.str_mapper.prettify_str(x[3]),
                x[4],
                self.str_mapper.prettify_str(x[5]),
                x[6],
            )
        )
        return coefficients

    def _set_stats(self):
        stats = self.results.stats.copy(deep=True)
        if self.specs.group is not None:
            stats.columns = [self.specs.group]
        stats.loc[
            stats.index.get_level_values("Stats") == "Dependent Var", stats.columns[0]
        ] = self.str_mapper.prettify_strs(
            stats.loc[
                stats.index.get_level_values("Stats") == "Dependent Var",
                stats.columns[0],
            ]
        )
        stats.columns = pd.MultiIndex.from_tuples(
            [(self.specs.regression_id, c) for c in stats.columns]
        )
        return stats

    def _set_residuals(self):
        return pd.concat(self.results.residuals, axis=1)

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
            + "\\end{adjustbox}\n"
        )
        table_text = (
            table_text
            + "{\\centering\\tiny Note: * p\\textless0.05, ** p\\textless0.01, *** p\\textless0.001\\par}"
            if note
            else table_text
        )
        table_text = table_text.replace("\\_square", r"\textsuperscript{2}")
        table_text = table_text.replace("\\_log", r"\textsubscript{log}")
        logger.info(table_text)

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
        logger.info(f"${model_specification}$")

    def quantile_model_equation(self):
        logger.info(
            "$\\min_{\\beta} \\sum_{i:y_g \\geq x_g^T\\beta} q |y_g - x_g^T\\beta| + \\sum_{g:y_g < x_g^T\\beta} (1-q) |y_g - x_g^T\\beta|$"
        )
