import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.linear_models import (
    OLSModel,
    QuantileRegressionModel,
    RLMModel,
    RollingOLSModel,
)


class TestOLSModel(TestCase):
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
        model = OLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "OLS")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = OLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = OLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = OLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = OLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = OLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = OLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = OLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = OLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestQuantileRegressionModel(TestCase):
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
        model = QuantileRegressionModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "QuantReg")
        self.assertEqual(model.quantiles, [0.5])
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = QuantileRegressionModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])

    def test_init_with_multiple_quantiles(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            quantiles=quantiles,
        )
        self.assertEqual(model.quantiles, quantiles)

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = QuantileRegressionModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_multiple_quantiles(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            quantiles=quantiles,
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), len(quantiles))
        for q in quantiles:
            self.assertIn(q, results)
            self.assertIsNotNone(results[q].params)

    def test_predict_before_fit(self):
        model = QuantileRegressionModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), 1)  # One quantile (0.5)
        self.assertEqual(len(predictions[0.5]), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions[0.5])))

    def test_predict_with_new_data(self):
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), 1)  # One quantile (0.5)
        self.assertEqual(len(predictions[0.5]), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions[0.5])))

    def test_predict_with_specific_quantile(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            quantiles=quantiles,
        )
        model.fit()

        predictions = model.predict(quantiles=0.5)
        self.assertEqual(len(predictions), 1)  # One quantile (0.5)
        self.assertEqual(len(predictions[0.5]), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions[0.5])))

    def test_predict_with_invalid_quantile(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            quantiles=quantiles,
        )
        model.fit()

        with self.assertRaises(ArgumentValueError):
            model.predict(quantiles=0.3)

    def test_summary_with_multiple_quantiles(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            quantiles=quantiles,
        )
        model.fit()

        summaries = model.summary()
        self.assertEqual(len(summaries), len(quantiles))
        for q in quantiles:
            self.assertIn(q, summaries)

    def test_summary_with_specific_quantile(self):
        quantiles = [0.25, 0.5, 0.75]
        model = QuantileRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            quantiles=quantiles,
        )
        model.fit()

        summary = model.summary(quantiles=0.5)
        self.assertIsNotNone(summary)


class TestRLMModel(TestCase):
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
        model = RLMModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "RLM")
        self.assertIsInstance(model.M, HuberT)
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = RLMModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = RLMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])

    def test_init_with_custom_norm(self):
        custom_norm = HuberT(t=2.0)
        model = RLMModel(
            self.data,
            dependent_variable="y",
            M=custom_norm,
        )
        self.assertEqual(model.M, custom_norm)

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = RLMModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = RLMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = RLMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = RLMModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = RLMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = RLMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestRollingOLSModel(TestCase):
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
        model = RollingOLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertIsNone(model.independent_variables)
        self.assertIsNone(model.control_variables)
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "RollingOLS")
        self.assertEqual(model.window, 30)
        self.assertIsNone(model.min_nobs)
        self.assertFalse(model.expanding)
        self.assertEqual(model.missing, "drop")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = RollingOLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertIsNone(model.dependent_variable)
        self.assertIsNone(model.independent_variables)
        self.assertIsNone(model.control_variables)

    def test_init_with_variables(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertIsNone(model.control_variables)

    def test_init_without_dependent_variable(self):
        with self.assertRaises(ArgumentValueError):
            RollingOLSModel(self.data)

    def test_init_with_custom_window(self):
        window = 50
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            window=window,
        )
        self.assertEqual(model.window, window)

    def test_init_with_min_nobs(self):
        min_nobs = 20
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            min_nobs=min_nobs,
        )
        self.assertEqual(model.min_nobs, min_nobs)

    def test_init_with_expanding(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            expanding=True,
        )
        self.assertTrue(model.expanding)

    def test_init_with_missing_options(self):
        for missing in ["drop", "skip", "raise"]:
            model = RollingOLSModel(
                self.data,
                dependent_variable="y",
                missing=missing,
            )
            self.assertEqual(model.missing, missing)

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = RollingOLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = RollingOLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_missing_variables(self):
        model = RollingOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
            }
        )
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)


if __name__ == "__main__":
    unittest.main()
