import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from mitoolspro.exceptions import ArgumentStructureError, ArgumentValueError
from mitoolspro.regressions.base_models import (
    BasePanelRegressionModel,
    BaseRegressionModel,
)


class TestRegressionModel(BaseRegressionModel):
    def fit(self, *args, **kwargs):
        self.fitted = True
        self.results = type(
            "Results",
            (),
            {
                "predict": lambda x: np.array([1, 2, 3]),
                "summary": lambda: "Test Summary",
            },
        )()


class TestBaseRegressionModel(TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5],
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 4, 6, 8, 10],
                "c1": [1, 1, 1, 1, 1],
            }
        )

    def test_init_with_dependent_and_independent(self):
        model = TestRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        model = TestRegressionModel(self.data, formula="y ~ x1 + x2")
        self.assertEqual(model.formula, "y ~ x1 + x2")
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.variables)

    def test_init_with_control_variables(self):
        model = TestRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=["c1"],
        )
        self.assertEqual(model.control_variables, ["c1"])

    def test_init_with_auto_independent_variables(self):
        model = TestRegressionModel(self.data, dependent_variable="y")
        expected_vars = ["c1", "x1", "x2"]
        self.assertEqual(sorted(model.independent_variables), expected_vars)

    def test_init_validation_errors(self):
        with self.assertRaises(ArgumentValueError):
            TestRegressionModel(self.data)

        with self.assertRaises(ArgumentValueError):
            TestRegressionModel(
                self.data,
                dependent_variable="y",
                independent_variables=["x1"],
                formula="y ~ x1",
            )

    def test_predict_before_fit(self):
        model = TestRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1"]
        )
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = TestRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1"]
        )
        model.fit()
        predictions = model.predict()
        np.testing.assert_array_equal(predictions, np.array([1, 2, 3]))

    def test_summary_before_fit(self):
        model = TestRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1"]
        )
        with self.assertRaises(ArgumentValueError):
            model.summary()

    def test_summary_after_fit(self):
        model = TestRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1"]
        )
        model.fit()
        self.assertEqual(model.summary(), "Test Summary")

    def test_from_arrays(self):
        y = np.array([1, 2, 3])
        X = np.array([[1, 2], [2, 4], [3, 6]])
        controls = np.array([[1], [1], [1]])

        model = TestRegressionModel.from_arrays(y, X, controls)
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, ["c1"])

    def test_from_arrays_validation(self):
        y = np.array([[1, 2], [2, 4]])
        X = np.array([[1, 2], [2, 4]])
        with self.assertRaises(ArgumentValueError):
            TestRegressionModel.from_arrays(y, X)

        y = np.array([1, 2])
        X = np.array([1, 2])
        with self.assertRaises(ArgumentValueError):
            TestRegressionModel.from_arrays(y, X)

        y = np.array([1, 2])
        X = np.array([[1, 2], [2, 4]])
        controls = np.array([1, 2])
        with self.assertRaises(ArgumentValueError):
            TestRegressionModel.from_arrays(y, X, controls)


class TestBasePanelRegressionModel(TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_product(
            [["A", "B"], [1, 2, 3]], names=["entity", "time"]
        )
        self.data = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5, 6],
                "x1": [1, 2, 3, 4, 5, 6],
                "x2": [2, 4, 6, 8, 10, 12],
                "c1": [1, 1, 1, 1, 1, 1],
            },
            index=index,
        )

    def test_init_with_valid_panel_data(self):
        model = BasePanelRegressionModel(
            self.data, dependent_variable="y", independent_variables=["x1", "x2"]
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertFalse(model.fitted)

    def test_init_with_invalid_panel_data(self):
        invalid_data = pd.DataFrame({"y": [1, 2, 3], "x1": [1, 2, 3]})
        with self.assertRaises(ArgumentValueError):
            BasePanelRegressionModel(
                invalid_data, dependent_variable="y", independent_variables=["x1"]
            )

    def test_init_with_formula(self):
        model = BasePanelRegressionModel(self.data, formula="y ~ x1 + x2")
        self.assertEqual(model.formula, "y ~ x1 + x2")
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.variables)

    def test_init_with_control_variables(self):
        model = BasePanelRegressionModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=["c1"],
        )
        self.assertEqual(model.control_variables, ["c1"])

    def test_validate_data(self):
        invalid_data = pd.DataFrame({"y": [1, 2, 3], "x1": [1, 2, 3]})
        with self.assertRaises(ArgumentValueError):
            BasePanelRegressionModel(
                invalid_data, dependent_variable="y", independent_variables=["x1"]
            )


if __name__ == "__main__":
    unittest.main()
