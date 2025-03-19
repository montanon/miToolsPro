import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.seasonality_models import MSTLModel


class TestMSTLModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 100)
        self.data = pd.DataFrame(
            {"y": np.sin(t) + 0.5 * np.sin(2 * t) + np.random.normal(0, 0.1, 100)}
        )

    def test_init_with_defaults(self):
        model = MSTLModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertIsNone(model.periods)
        self.assertIsNone(model.windows)
        self.assertEqual(model.model_name, "MSTL")
        self.assertFalse(model.fitted)
        self.assertEqual(model.kwargs, {})

    def test_init_with_periods(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=12)
        self.assertEqual(model.periods, 12)

    def test_init_with_multiple_periods(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=[12, 24])
        self.assertEqual(model.periods, [12, 24])

    def test_init_with_windows(self):
        model = MSTLModel(self.data, dependent_variable="y", windows=7)
        self.assertEqual(model.windows, 7)

    def test_init_with_multiple_windows(self):
        model = MSTLModel(self.data, dependent_variable="y", windows=[7, 13])
        self.assertEqual(model.windows, [7, 13])

    def test_init_with_kwargs(self):
        kwargs = {"seasonal_deg": 1, "trend_deg": 2}
        model = MSTLModel(self.data, dependent_variable="y", **kwargs)
        self.assertEqual(model.kwargs, kwargs)

    def test_fit(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=12)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.trend)
        self.assertIsNotNone(results.seasonal)
        self.assertIsNotNone(results.resid)

    def test_predict_before_fit(self):
        model = MSTLModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=12)
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_multiple_periods(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=[12, 24])
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_summary_before_fit(self):
        model = MSTLModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.summary()

    def test_summary_after_fit(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=12)
        model.fit()
        summary = model.summary()

        self.assertIsInstance(summary, str)
        self.assertIn("MSTL Decomposition", summary)
        self.assertIn("Observations:", summary)
        self.assertIn("Number of seasonal components:", summary)
        self.assertIn("Trend shape:", summary)
        self.assertIn("Seasonal shape:", summary)
        self.assertIn("Remainder shape:", summary)

    def test_summary_with_multiple_periods(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=[12, 24])
        model.fit()
        summary = model.summary()

        self.assertIsInstance(summary, str)
        self.assertIn("MSTL Decomposition", summary)
        self.assertIn("Observations:", summary)
        self.assertIn("Number of seasonal components: 2", summary)
        self.assertIn("Trend shape:", summary)
        self.assertIn("Seasonal shape:", summary)
        self.assertIn("Remainder shape:", summary)

    def test_fit_predict_consistency(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=12)
        model.fit()
        predictions = model.predict()

        # Check that predictions are close to actual values
        mse = np.mean((predictions - self.data["y"].values) ** 2)
        self.assertLess(mse, 1.0)  # MSE should be less than 1 given our noise level

    def test_multiple_periods_decomposition(self):
        model = MSTLModel(self.data, dependent_variable="y", periods=[12, 24])
        model.fit()
        predictions = model.predict()

        # Check that predictions are close to actual values
        mse = np.mean((predictions - self.data["y"].values) ** 2)
        self.assertLess(mse, 1.0)  # MSE should be less than 1 given our noise level


if __name__ == "__main__":
    unittest.main()
