import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import DataFrame, Series

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.system_models import SURModel


class TestSURModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100

        y1 = np.random.normal(0, 1, n_samples)
        y2 = np.random.normal(0, 1, n_samples)
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        x3 = np.random.normal(0, 1, n_samples)

        self.equations_data = {
            "eq1": {
                "dependent": pd.Series(y1),
                "exog": pd.DataFrame({"x1": x1, "x2": x2}),
            },
            "eq2": {
                "dependent": pd.Series(y2),
                "exog": pd.DataFrame({"x2": x2, "x3": x3}),
            },
        }

    def test_init_with_valid_data(self):
        model = SURModel(self.equations_data)
        self.assertEqual(model.model_name, "SUR")
        self.assertFalse(model.fitted)
        self.assertIsNone(model.variables)
        self.assertEqual(model.equations_data, self.equations_data)

    def test_init_with_empty_dict(self):
        with self.assertRaises(ArgumentValueError):
            SURModel({})

    def test_init_with_invalid_dict_type(self):
        with self.assertRaises(ArgumentValueError):
            SURModel("not_a_dict")

    def test_init_with_missing_dependent_key(self):
        invalid_data = {
            "eq1": {
                "exog": pd.DataFrame({"x1": np.random.normal(0, 1, 100)}),
            }
        }
        with self.assertRaises(ArgumentValueError):
            SURModel(invalid_data)

    def test_init_with_missing_exog_key(self):
        invalid_data = {
            "eq1": {
                "dependent": pd.Series(np.random.normal(0, 1, 100)),
            }
        }
        with self.assertRaises(ArgumentValueError):
            SURModel(invalid_data)

    def test_init_with_invalid_dependent_type(self):
        invalid_data = {
            "eq1": {
                "dependent": "not_a_series",
                "exog": pd.DataFrame({"x1": np.random.normal(0, 1, 100)}),
            }
        }
        with self.assertRaises(ArgumentValueError):
            SURModel(invalid_data)

    def test_init_with_invalid_exog_type(self):
        invalid_data = {
            "eq1": {
                "dependent": pd.Series(np.random.normal(0, 1, 100)),
                "exog": "not_a_dataframe",
            }
        }
        with self.assertRaises(ArgumentValueError):
            SURModel(invalid_data)

    def test_init_with_multi_column_dependent(self):
        invalid_data = {
            "eq1": {
                "dependent": pd.DataFrame(
                    {
                        "y1": np.random.normal(0, 1, 100),
                        "y2": np.random.normal(0, 1, 100),
                    }
                ),
                "exog": pd.DataFrame({"x1": np.random.normal(0, 1, 100)}),
            }
        }
        with self.assertRaises(ArgumentValueError):
            SURModel(invalid_data)

    def test_fit_with_defaults(self):
        model = SURModel(self.equations_data)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 6)  # 2 equations * (const + 2 vars)

    def test_fit_without_constant(self):
        model = SURModel(self.equations_data)
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 4)  # 2 equations * 2 vars

    def test_fit_with_constraints(self):
        model = SURModel(self.equations_data)
        constraints = pd.DataFrame(
            {
                "const": [0.0],
                "x1": [1.0],
                "x2": [0.0],
                "const.1": [0.0],
                "x2.1": [0.0],
                "x3": [1.0],
            }
        )
        results = model.fit(constraints=constraints)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)

    def test_predict_before_fit(self):
        model = SURModel(self.equations_data)
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_with_default_data(self):
        model = SURModel(self.equations_data)
        model.fit()
        predictions = model.predict()

        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), 2)  # One prediction per equation
        for eq_name, pred in predictions.items():
            self.assertEqual(len(pred), len(self.equations_data[eq_name]["dependent"]))

    def test_predict_with_new_data(self):
        model = SURModel(self.equations_data)
        model.fit()

        new_data = {
            "eq1": pd.DataFrame(
                {
                    "x1": np.random.normal(0, 1, 10),
                    "x2": np.random.normal(0, 1, 10),
                }
            ),
            "eq2": pd.DataFrame(
                {
                    "x2": np.random.normal(0, 1, 10),
                    "x3": np.random.normal(0, 1, 10),
                }
            ),
        }
        predictions = model.predict(new_data)

        self.assertIsInstance(predictions, dict)
        self.assertEqual(len(predictions), 2)
        for eq_name, pred in predictions.items():
            self.assertEqual(len(pred), len(new_data[eq_name]))

    def test_predict_with_invalid_new_data_type(self):
        model = SURModel(self.equations_data)
        model.fit()

        with self.assertRaises(ArgumentValueError):
            model.predict("not_a_dict")

    def test_predict_with_missing_equation(self):
        model = SURModel(self.equations_data)
        model.fit()

        new_data = {
            "eq1": pd.DataFrame(
                {
                    "x1": np.random.normal(0, 1, 10),
                    "x2": np.random.normal(0, 1, 10),
                }
            ),
        }
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)

    def test_predict_with_wrong_variables(self):
        model = SURModel(self.equations_data)
        model.fit()

        new_data = {
            "eq1": pd.DataFrame(
                {
                    "x1": np.random.normal(0, 1, 10),
                    "x3": np.random.normal(0, 1, 10),
                }
            ),
            "eq2": pd.DataFrame(
                {
                    "x2": np.random.normal(0, 1, 10),
                    "x3": np.random.normal(0, 1, 10),
                }
            ),
        }
        with self.assertRaises(ArgumentValueError):
            model.predict(new_data)


if __name__ == "__main__":
    unittest.main()
