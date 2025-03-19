import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.regime_models import (
    MarkovAutoregressionModel,
    MarkovRegressionModel,
)


class TestMarkovRegressionModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = MarkovRegressionModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.k_regimes, 2)
        self.assertEqual(model.trend, "c")
        self.assertTrue(model.switching_trend)
        self.assertTrue(model.switching_exog)
        self.assertFalse(model.switching_variance)
        self.assertEqual(model.model_name, "MarkovRegression")
        self.assertFalse(model.fitted)
        self.assertIsNone(model.formula)

    def test_init_with_custom_regimes(self):
        model = MarkovRegressionModel(self.data, dependent_variable="y", k_regimes=3)
        self.assertEqual(model.k_regimes, 3)

    def test_init_with_different_trends(self):
        for trend in ["n", "c", "t", "ct"]:
            model = MarkovRegressionModel(
                self.data, dependent_variable="y", trend=trend
            )
            self.assertEqual(model.trend, trend)

    def test_init_with_switching_options(self):
        model = MarkovRegressionModel(
            self.data,
            dependent_variable="y",
            switching_trend=False,
            switching_exog=False,
            switching_variance=True,
        )
        self.assertFalse(model.switching_trend)
        self.assertFalse(model.switching_exog)
        self.assertTrue(model.switching_variance)

    def test_init_with_independent_variables(self):
        model = MarkovRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        self.assertEqual(model.independent_variables, ["x1", "x2"])

    def test_fit(self):
        model = MarkovRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertIsNotNone(results.smoothed_marginal_probabilities)

    def test_fit_with_different_regimes(self):
        model = MarkovRegressionModel(self.data, dependent_variable="y", k_regimes=3)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertEqual(results.k_regimes, 3)

    def test_predict_before_fit(self):
        model = MarkovRegressionModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = MarkovRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_probabilities(self):
        model = MarkovRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        model.fit()
        predictions = model.predict(probabilities="filtered")

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_conditional(self):
        model = MarkovRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        model.fit()
        predictions = model.predict(conditional=True)

        self.assertEqual(predictions.shape[0], model.k_regimes)
        self.assertEqual(predictions.shape[1], len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestMarkovAutoregressionModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = MarkovAutoregressionModel(self.data, dependent_variable="y", order=1)
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.order, 1)
        self.assertEqual(model.k_regimes, 2)
        self.assertEqual(model.trend, "c")
        self.assertTrue(model.switching_trend)
        self.assertTrue(model.switching_exog)
        self.assertTrue(model.switching_ar)
        self.assertFalse(model.switching_variance)
        self.assertEqual(model.model_name, "MarkovAutoregression")
        self.assertFalse(model.fitted)
        self.assertIsNone(model.formula)

    def test_init_with_custom_regimes(self):
        model = MarkovAutoregressionModel(
            self.data, dependent_variable="y", order=1, k_regimes=3
        )
        self.assertEqual(model.k_regimes, 3)

    def test_init_with_different_orders(self):
        for order in [1, 2, 3]:
            model = MarkovAutoregressionModel(
                self.data, dependent_variable="y", order=order
            )
            self.assertEqual(model.order, order)

    def test_init_with_different_trends(self):
        for trend in ["n", "c", "t", "ct"]:
            model = MarkovAutoregressionModel(
                self.data, dependent_variable="y", order=1, trend=trend
            )
            self.assertEqual(model.trend, trend)

    def test_init_with_switching_options(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            switching_trend=False,
            switching_exog=False,
            switching_ar=False,
            switching_variance=True,
        )
        self.assertFalse(model.switching_trend)
        self.assertFalse(model.switching_exog)
        self.assertFalse(model.switching_ar)
        self.assertTrue(model.switching_variance)

    def test_init_with_independent_variables(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            independent_variables=["x1", "x2"],
        )
        self.assertEqual(model.independent_variables, ["x1", "x2"])

    def test_fit(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertIsNotNone(results.smoothed_marginal_probabilities)

    def test_fit_with_different_regimes(self):
        model = MarkovAutoregressionModel(
            self.data, dependent_variable="y", order=1, k_regimes=3
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertEqual(results.k_regimes, 3)

    def test_fit_with_different_orders(self):
        for order in [1, 2, 3]:
            model = MarkovAutoregressionModel(
                self.data, dependent_variable="y", order=order
            )
            results = model.fit()
            self.assertTrue(model.fitted)
            self.assertIsNotNone(results)

    def test_predict_before_fit(self):
        model = MarkovAutoregressionModel(self.data, dependent_variable="y", order=1)
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data) - model.order)
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_probabilities(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict(probabilities="filtered")

        self.assertEqual(len(predictions), len(self.data) - model.order)
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_conditional(self):
        model = MarkovAutoregressionModel(
            self.data,
            dependent_variable="y",
            order=1,
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict(conditional=True)

        self.assertEqual(predictions.shape[0], model.k_regimes)
        self.assertEqual(predictions.shape[1], len(self.data) - model.order)
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
