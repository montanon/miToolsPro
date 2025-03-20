import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.ivars_models import (
    IV2SLSModel,
    IVGMMCUEModel,
    IVGMMModel,
    IVLIMLModel,
)


class TestIV2SLSModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
                "z1": np.random.normal(0, 1, n_samples),
                "z2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_model_attributes(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()
        print("\nAvailable attributes in IV2SLS model:")
        print(dir(model.results.model))
        print("\nAvailable attributes in IV2SLS results:")
        print(dir(model.results))

    def test_init_with_defaults(self):
        model = IV2SLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2", "z1", "z2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "IV2SLS")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IV2SLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])

    def test_init_with_variables(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, ["x1"])
        self.assertEqual(model.instrument_variables, ["z1", "z2"])

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IV2SLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = IV2SLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = IV2SLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestIVGMMModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
                "z1": np.random.normal(0, 1, n_samples),
                "z2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = IVGMMModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2", "z1", "z2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "IVGMM")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVGMMModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])

    def test_init_with_variables(self):
        model = IVGMMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, ["x1"])
        self.assertEqual(model.instrument_variables, ["z1", "z2"])

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVGMMModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = IVGMMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = IVGMMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = IVGMMModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = IVGMMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = IVGMMModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestIVGMMCUEModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
                "z1": np.random.normal(0, 1, n_samples),
                "z2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = IVGMMCUEModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2", "z1", "z2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "IVGMMCUE")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVGMMCUEModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])

    def test_init_with_variables(self):
        model = IVGMMCUEModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, ["x1"])
        self.assertEqual(model.instrument_variables, ["z1", "z2"])

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVGMMCUEModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = IVGMMCUEModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = IVGMMCUEModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = IVGMMCUEModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = IVGMMCUEModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = IVGMMCUEModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestIVLIMLModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
                "z1": np.random.normal(0, 1, n_samples),
                "z2": np.random.normal(0, 1, n_samples),
            }
        )

    def test_init_with_defaults(self):
        model = IVLIMLModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2", "z1", "z2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "IVLIML")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVLIMLModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, [])
        self.assertEqual(model.instrument_variables, [])

    def test_init_with_variables(self):
        model = IVLIMLModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertEqual(model.endogenous_variables, ["x1"])
        self.assertEqual(model.instrument_variables, ["z1", "z2"])

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = IVLIMLModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = IVLIMLModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_without_constant(self):
        model = IVLIMLModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        results = model.fit(add_constant=False)

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = IVLIMLModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = IVLIMLModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = IVLIMLModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            endogenous_variables=["x1"],
            instrument_variables=["z1", "z2"],
        )
        model.fit()

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, 10),
                "x2": np.random.normal(0, 1, 10),
            }
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
