import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.linearfactor_models import (
    LinearFactorGMMModel,
    LinearFactorModel,
    TradedFactorModel,
)


class TestTradedFactorModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        n_portfolios = 5
        n_factors = 2

        self.data = pd.DataFrame(
            {
                "portfolio1": np.random.normal(0, 1, n_samples),
                "portfolio2": np.random.normal(0, 1, n_samples),
                "portfolio3": np.random.normal(0, 1, n_samples),
                "portfolio4": np.random.normal(0, 1, n_samples),
                "portfolio5": np.random.normal(0, 1, n_samples),
                "factor1": np.random.normal(0, 1, n_samples),
                "factor2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = TradedFactorModel(self.data, portfolios="portfolio1")
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(
            model.independent_variables,
            [
                "factor1",
                "factor2",
                "portfolio2",
                "portfolio3",
                "portfolio4",
                "portfolio5",
            ],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "TradedFactorModel")
        self.assertFalse(model.fitted)

    def test_init_with_multiple_portfolios(self):
        portfolios = ["portfolio1", "portfolio2"]
        model = TradedFactorModel(
            self.data,
            portfolios=portfolios,
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, portfolios)
        self.assertEqual(model.independent_variables, ["factor1"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "TradedFactorModel")
        self.assertFalse(model.fitted)

    def test_init_with_specific_factors(self):
        model = TradedFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(model.independent_variables, ["factor1"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "TradedFactorModel")
        self.assertFalse(model.fitted)

    def test_fit_with_single_portfolio(self):
        model = TradedFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 1)  # Only one factor coefficient

    def test_fit_with_multiple_portfolios(self):
        model = TradedFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2"],
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)

    def test_predict_before_fit(self):
        model = TradedFactorModel(self.data, portfolios="portfolio1")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = TradedFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = TradedFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
                "factor2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_missing_factors(self):
        model = TradedFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
            }
        )
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)


class TestLinearFactorModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100

        self.data = pd.DataFrame(
            {
                "portfolio1": np.random.normal(0, 1, n_samples),
                "portfolio2": np.random.normal(0, 1, n_samples),
                "portfolio3": np.random.normal(0, 1, n_samples),
                "portfolio4": np.random.normal(0, 1, n_samples),
                "portfolio5": np.random.normal(0, 1, n_samples),
                "factor1": np.random.normal(0, 1, n_samples),
                "factor2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = LinearFactorModel(self.data, portfolios="portfolio1")
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(
            model.independent_variables,
            [
                "factor1",
                "factor2",
                "portfolio2",
                "portfolio3",
                "portfolio4",
                "portfolio5",
            ],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_multiple_portfolios(self):
        portfolios = ["portfolio1", "portfolio2"]
        model = LinearFactorModel(
            self.data,
            portfolios=portfolios,
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, portfolios)
        self.assertEqual(
            model.independent_variables,
            ["factor1"],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_specific_factors(self):
        model = LinearFactorModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(
            model.independent_variables,
            ["factor1"],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_risk_free(self):
        model = LinearFactorModel(
            self.data,
            portfolios="portfolio1",
            risk_free=True,
        )
        self.assertTrue(model.risk_free)

    def test_fit_with_single_portfolio(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + factor1 + factor2

    def test_fit_with_multiple_portfolios(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2"],
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)

    def test_fit_with_risk_free(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
            risk_free=True,
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)

    def test_predict_before_fit(self):
        model = LinearFactorModel(self.data, portfolios="portfolio1")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertEqual(len(predictions.columns), 3)  # One column per portfolio
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
                "factor2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_missing_factors(self):
        model = LinearFactorModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
            }
        )
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)


class TestLinearFactorGMMModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100

        self.data = pd.DataFrame(
            {
                "portfolio1": np.random.normal(0, 1, n_samples),
                "portfolio2": np.random.normal(0, 1, n_samples),
                "portfolio3": np.random.normal(0, 1, n_samples),
                "portfolio4": np.random.normal(0, 1, n_samples),
                "portfolio5": np.random.normal(0, 1, n_samples),
                "factor1": np.random.normal(0, 1, n_samples),
                "factor2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = LinearFactorGMMModel(self.data, portfolios="portfolio1")
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(
            model.independent_variables,
            [
                "factor1",
                "factor2",
                "portfolio2",
                "portfolio3",
                "portfolio4",
                "portfolio5",
            ],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorGMMModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_multiple_portfolios(self):
        portfolios = ["portfolio1", "portfolio2"]
        model = LinearFactorGMMModel(
            self.data,
            portfolios=portfolios,
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, portfolios)
        self.assertEqual(
            model.independent_variables,
            ["factor1"],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorGMMModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_specific_factors(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1"],
        )
        self.assertEqual(model.dependent_variable, "portfolio1")
        self.assertEqual(
            model.independent_variables,
            ["factor1"],
        )
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "LinearFactorGMMModel")
        self.assertFalse(model.risk_free)
        self.assertFalse(model.fitted)

    def test_init_with_risk_free(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios="portfolio1",
            risk_free=True,
        )
        self.assertTrue(model.risk_free)

    def test_fit_with_single_portfolio(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + factor1 + factor2

    def test_fit_with_multiple_portfolios(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios=["portfolio1", "portfolio2"],
            factors=["factor1", "factor2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)

    def test_fit_with_risk_free(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios=["portfolio1", "portfolio2", "portfolio3"],
            factors=["factor1", "factor2"],
            risk_free=True,
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)

    def test_predict_before_fit(self):
        model = LinearFactorGMMModel(self.data, portfolios="portfolio1")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
                "factor2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_missing_factors(self):
        model = LinearFactorGMMModel(
            self.data,
            portfolios="portfolio1",
            factors=["factor1", "factor2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "factor1": np.random.normal(0, 1, 10),
            }
        )
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)


if __name__ == "__main__":
    unittest.main()
