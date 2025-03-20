from typing import Dict, Optional, Union

import pandas as pd
import statsmodels.api as sm
from linearmodels.system import SUR
from pandas import DataFrame, Series

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.linear_models import BaseRegressionModel


class SURModel(BaseRegressionModel):
    def __init__(
        self,
        equations_data: Dict[str, Dict[str, Union[Series, DataFrame]]],
        *args,
        **kwargs,
    ):
        super().__init__(
            data=DataFrame(),
            formula=None,
            dependent_variable="dummy",
            independent_variables=["dummy"],
            *args,
            **kwargs,
        )
        if not equations_data or not isinstance(equations_data, dict):
            raise ArgumentValueError("You must provide a dictionary of equation data.")
        for eq_name, eq_dict in equations_data.items():
            if "dependent" not in eq_dict or "exog" not in eq_dict:
                raise ArgumentValueError(
                    f"Equation {eq_name} must have 'dependent' and 'exog' keys."
                )
            dep = eq_dict["dependent"]
            exog = eq_dict["exog"]
            if not (
                isinstance(dep, (Series, DataFrame)) and isinstance(exog, DataFrame)
            ):
                raise ArgumentValueError(
                    f"Equation {eq_name} 'dependent' must be Series/DataFrame and 'exog' must be DataFrame."
                )
            if isinstance(dep, DataFrame) and dep.shape[1] != 1:
                raise ArgumentValueError(
                    f"Equation {eq_name} dependent variable DataFrame must have exactly one column."
                )
        self.equations_data = equations_data
        self.model_name = "SUR"
        self.fitted = False
        self.variables = None

    def fit(
        self,
        add_constant: bool = True,
        constraints: Optional[DataFrame] = None,
        *args,
        **kwargs,
    ):
        eq_data = {}
        for eq_name, eq_dict in self.equations_data.items():
            dep = eq_dict["dependent"]
            exog = eq_dict["exog"]
            if add_constant:
                exog = sm.add_constant(exog)
            if isinstance(dep, DataFrame):
                if dep.shape[1] != 1:
                    raise ArgumentValueError(
                        f"Equation {eq_name} dependent variable DataFrame must have exactly one column."
                    )
                dep = dep.iloc[:, 0]
            eq_data[eq_name] = (dep, exog)
        self.model = SUR(eq_data, *self.args, **self.kwargs)
        if constraints is not None:
            self.model.add_constraints(constraints)
        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    @classmethod
    def multivariate_ls(cls, *args, **kwargs):
        return SUR.multivariate_ls(*args, **kwargs)

    def predict(self, new_data: Optional[Dict[str, DataFrame]] = None):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")

        if new_data is None:
            pred_data = {}
            for eq_name, eq_dict in self.equations_data.items():
                exog = eq_dict["exog"].copy()
                if any(
                    param.startswith(f"{eq_name}_const")
                    for param in self.results.params.index
                ):
                    exog = sm.add_constant(exog)
                    exog = exog.rename(columns={"const": f"{eq_name}_const"})
                pred_data[eq_name] = {"exog": exog}
        else:
            if not isinstance(new_data, dict):
                raise ArgumentValueError(
                    "new_data must be a dictionary keyed by equation names."
                )
            if set(new_data.keys()) != set(self.equations_data.keys()):
                raise ArgumentValueError(
                    "new_data must contain all equations from the original model."
                )
            pred_data = {}
            for eq_name, exog in new_data.items():
                if not isinstance(exog, DataFrame):
                    raise ArgumentValueError(
                        f"Equation {eq_name} data must be a DataFrame."
                    )
                expected_vars = set(self.equations_data[eq_name]["exog"].columns)
                actual_vars = set(exog.columns)
                if expected_vars != actual_vars:
                    raise ArgumentValueError(
                        f"Equation {eq_name} must have the same variables as the original model."
                    )
                exog = exog.copy()
                if any(
                    param.startswith(f"{eq_name}_const")
                    for param in self.results.params.index
                ):
                    exog = sm.add_constant(exog)
                    exog = exog.rename(columns={"const": f"{eq_name}_const"})
                pred_data[eq_name] = {"exog": exog}

        predictions_df = self.results.predict(pred_data, dataframe=True)
        predictions = {col: predictions_df[col] for col in predictions_df.columns}
        return predictions
