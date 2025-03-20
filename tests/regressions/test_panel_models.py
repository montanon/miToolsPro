import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas import MultiIndex

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.panel_models import (
    BetweenOLSModel,
    FirstDifferenceOLSModel,
    PanelOLSModel,
    PooledOLSModel,
    RandomEffectsModel,
)


class TestPanelOLSModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_samples = n_entities * n_time

        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            },
            index=index,
        )

    def test_init_with_defaults(self):
        model = PanelOLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "PanelOLS")
        self.assertFalse(model.entity_effects)
        self.assertFalse(model.time_effects)
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = PanelOLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            control_variables=None,
        )
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])

    def test_init_with_effects(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            entity_effects=True,
            time_effects=True,
        )
        self.assertTrue(model.entity_effects)
        self.assertTrue(model.time_effects)

    def test_fit_with_formula(self):
        formula = "y ~ x1 + x2"
        model = PanelOLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_effects(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)

    def test_fit_without_constant(self):
        model = PanelOLSModel(
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
        model = PanelOLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = PanelOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        n_new = 10
        n_entities = 2
        n_time = 5
        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_new),
                "x2": np.random.normal(0, 1, n_new),
            },
            index=index,
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestPooledOLSModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_samples = n_entities * n_time

        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            },
            index=index,
        )

    def test_init_with_defaults(self):
        model = PooledOLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "PooledOLS")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = PooledOLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = PooledOLSModel(
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
        model = PooledOLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = PooledOLSModel(
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
        model = PooledOLSModel(
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
        model = PooledOLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = PooledOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = PooledOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        n_new = 10
        n_entities = 2
        n_time = 5
        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_new),
                "x2": np.random.normal(0, 1, n_new),
            },
            index=index,
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestRandomEffectsModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_samples = n_entities * n_time

        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            },
            index=index,
        )

    def test_init_with_defaults(self):
        model = RandomEffectsModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "RandomEffects")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = RandomEffectsModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = RandomEffectsModel(
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
        model = RandomEffectsModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = RandomEffectsModel(
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
        model = RandomEffectsModel(
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
        model = RandomEffectsModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = RandomEffectsModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = RandomEffectsModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        n_new = 10
        n_entities = 2
        n_time = 5
        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_new),
                "x2": np.random.normal(0, 1, n_new),
            },
            index=index,
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestBetweenOLSModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_samples = n_entities * n_time

        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            },
            index=index,
        )

    def test_init_with_defaults(self):
        model = BetweenOLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "BetweenOLS")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = BetweenOLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = BetweenOLSModel(
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
        model = BetweenOLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 3)  # const + x1 + x2

    def test_fit_with_variables(self):
        model = BetweenOLSModel(
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
        model = BetweenOLSModel(
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
        model = BetweenOLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = BetweenOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = BetweenOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        n_new = 10
        n_entities = 2
        n_time = 5
        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_new),
                "x2": np.random.normal(0, 1, n_new),
            },
            index=index,
        )
        new_data = sm.add_constant(new_data)
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestFirstDifferenceOLSModel(TestCase):
    def setUp(self):
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_samples = n_entities * n_time

        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        self.data = pd.DataFrame(
            {
                "y": np.random.normal(0, 1, n_samples),
                "x1": np.random.normal(0, 1, n_samples),
                "x2": np.random.normal(0, 1, n_samples),
            },
            index=index,
        )

    def test_init_with_defaults(self):
        model = FirstDifferenceOLSModel(self.data, dependent_variable="y")
        self.assertEqual(model.dependent_variable, "y")
        self.assertEqual(model.independent_variables, ["x1", "x2"])
        self.assertEqual(model.control_variables, [])
        self.assertIsNone(model.formula)
        self.assertEqual(model.model_name, "FirstDifferenceOLS")
        self.assertFalse(model.fitted)

    def test_init_with_formula(self):
        formula = "y ~ x1 + x2"
        model = FirstDifferenceOLSModel(self.data, formula=formula)
        self.assertEqual(model.formula, formula)
        self.assertEqual(model.dependent_variable, "")
        self.assertEqual(model.independent_variables, [])
        self.assertEqual(model.control_variables, [])

    def test_init_with_variables(self):
        model = FirstDifferenceOLSModel(
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
        model = FirstDifferenceOLSModel(self.data, formula=formula)
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_fit_with_variables(self):
        model = FirstDifferenceOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()

        self.assertTrue(model.fitted)
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.params)
        self.assertEqual(len(results.params), 2)  # x1 + x2

    def test_predict_before_fit(self):
        model = FirstDifferenceOLSModel(self.data, dependent_variable="y")
        with self.assertRaises(ArgumentValueError):
            model.predict()

    def test_predict_after_fit(self):
        model = FirstDifferenceOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()
        predictions = model.predict()

        self.assertEqual(len(predictions), len(self.data))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_predict_with_new_data(self):
        model = FirstDifferenceOLSModel(
            self.data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        model.fit()

        n_new = 10
        n_entities = 2
        n_time = 5
        entity_ids = np.repeat(range(n_entities), n_time)
        time_ids = np.tile(range(n_time), n_entities)
        index = MultiIndex.from_arrays([entity_ids, time_ids], names=["entity", "time"])

        new_data = pd.DataFrame(
            {
                "x1": np.random.normal(0, 1, n_new),
                "x2": np.random.normal(0, 1, n_new),
            },
            index=index,
        )
        predictions = model.predict(new_data)

        self.assertEqual(len(predictions), len(new_data))
        self.assertTrue(np.all(np.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
