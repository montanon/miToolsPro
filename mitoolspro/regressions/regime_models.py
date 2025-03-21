from typing import List, Literal, Optional

from pandas import DataFrame
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.base_models import BaseRegressionModel


class MarkovRegressionModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: str,
        independent_variables: Optional[List[str]] = None,
        k_regimes: int = 2,
        trend: Literal["n", "c", "t", "ct"] = "c",
        switching_trend: bool = True,
        switching_exog: bool = True,
        switching_variance: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=None,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=None,
            *args,
            **kwargs,
        )

        if self.formula is not None:
            raise ArgumentValueError(
                "MarkovRegression does not support a formula interface."
            )

        if self.dependent_variable is None:
            raise ArgumentValueError(
                "You must provide a dependent_variable for MarkovRegressionModel."
            )
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        self.model_name = "MarkovRegression"

    def fit(self, *args, **kwargs):
        endog = self.data[self.dependent_variable].values
        exog = None
        if self.independent_variables:
            exog = self.data[self.independent_variables].values

        self.model = MarkovRegression(
            endog,
            k_regimes=self.k_regimes,
            trend=self.trend,
            exog=exog,
            switching_variance=self.switching_variance,
            switching_trend=self.switching_trend,
            switching_exog=self.switching_exog,
            *self.args,
            **self.kwargs,
        )

        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, start=None, end=None, probabilities=None, conditional=False):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        return self.results.predict(
            start=start, end=end, probabilities=probabilities, conditional=conditional
        )


class MarkovAutoregressionModel(BaseRegressionModel):
    def __init__(
        self,
        data: DataFrame,
        dependent_variable: str,
        order: int,
        independent_variables: Optional[List[str]] = None,
        k_regimes: int = 2,
        trend: Literal["n", "c", "t", "ct"] = "c",
        switching_trend: bool = True,
        switching_exog: bool = True,
        switching_ar: bool = True,
        switching_variance: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            data=data,
            formula=None,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            control_variables=None,
            *args,
            **kwargs,
        )

        if self.formula is not None:
            raise ArgumentValueError(
                "MarkovAutoregression does not support a formula interface."
            )

        if self.dependent_variable is None:
            raise ArgumentValueError(
                "You must provide a dependent_variable for MarkovAutoregressionModel."
            )
        self.order = order
        self.k_regimes = k_regimes
        self.trend = trend
        self.switching_ar = switching_ar
        self.switching_trend = switching_trend
        self.switching_exog = switching_exog
        self.switching_variance = switching_variance
        self.model_name = "MarkovAutoregression"

    def fit(self, *args, **kwargs):
        endog = self.data[self.dependent_variable].values
        exog = None
        if self.independent_variables:
            exog = self.data[self.independent_variables].values

        self.model = MarkovAutoregression(
            endog,
            k_regimes=self.k_regimes,
            order=self.order,
            trend=self.trend,
            exog=exog,
            switching_variance=self.switching_variance,
            switching_trend=self.switching_trend,
            switching_exog=self.switching_exog,
            switching_ar=self.switching_ar,
            *self.args,
            **self.kwargs,
        )

        self.results = self.model.fit(*args, **kwargs)
        self.fitted = True
        return self.results

    def predict(self, start=None, end=None, probabilities=None, conditional=False):
        if not self.fitted:
            raise ArgumentValueError("Model not fitted yet")
        return self.results.predict(
            start=start, end=end, probabilities=probabilities, conditional=conditional
        )
