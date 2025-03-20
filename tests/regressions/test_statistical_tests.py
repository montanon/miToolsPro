import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.stats import anderson as scipy_anderson
from scipy.stats import norm
from scipy.stats import shapiro as scipy_shapiro
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.regressions.linear_models import OLSModel
from mitoolspro.regressions.statistical_tests import (
    StatisticalTests,
    adf_test,
    adf_tests,
    anderson_test,
    anderson_tests,
    breusch_pagan_test,
    calculate_vif,
    durbin_watson_test,
    shapiro_test,
    shapiro_tests,
    white_test,
)


class TestStatisticalTestFunctions(TestCase):
    def setUp(self):
        np.random.seed(42)
        self.normal_data = pd.Series(norm.rvs(size=1000))
        self.normal_df = pd.DataFrame(
            {
                "var1": norm.rvs(size=1000),
                "var2": norm.rvs(size=1000),
                "var3": norm.rvs(size=1000),
            }
        )
        n = 1000
        x1 = norm.rvs(size=n)
        x2 = norm.rvs(size=n)
        y = 2 * x1 + 3 * x2 + norm.rvs(size=n)
        self.regression_data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def test_shapiro_test(self):
        result = shapiro_test(self.normal_data)
        self.assertIsInstance(result, dict)
        self.assertIn("statistic", result)
        self.assertIn("p-value", result)
        self.assertTrue(0 <= result["p-value"] <= 1)
        self.assertTrue(result["statistic"] > 0)
        expected_stat, expected_p = scipy_shapiro(self.normal_data)
        np.testing.assert_almost_equal(result["statistic"], expected_stat, decimal=10)
        np.testing.assert_almost_equal(result["p-value"], expected_p, decimal=10)

    def test_shapiro_tests(self):
        result = shapiro_tests(self.normal_df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ["statistic", "p-value"]))
        self.assertTrue(all(0 <= result["p-value"] <= 1))
        self.assertTrue(all(result["statistic"] > 0))
        expected_stats, expected_pvals = np.array(
            [scipy_shapiro(self.normal_df[col]) for col in self.normal_df.columns]
        ).T
        np.testing.assert_almost_equal(
            result["statistic"].values, expected_stats, decimal=10
        )
        np.testing.assert_almost_equal(
            result["p-value"].values, expected_pvals, decimal=10
        )

    def test_anderson_test(self):
        result = anderson_test(self.normal_data, criteria=0.01)
        self.assertIsInstance(result, dict)
        self.assertIn("statistic", result)
        self.assertIn("critical_value", result)
        self.assertTrue(result["statistic"] > 0)
        self.assertTrue(result["critical_value"] > 0)
        expected_result = scipy_anderson(self.normal_data, dist="norm")
        np.testing.assert_almost_equal(
            result["statistic"], expected_result.statistic, decimal=10
        )
        np.testing.assert_almost_equal(
            result["critical_value"], expected_result.critical_values[2], decimal=10
        )

    def test_anderson_tests(self):
        result = anderson_tests(self.normal_df, criteria=0.01)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(col in result.columns for col in ["statistic", "critical_value"])
        )
        self.assertTrue(all(result["statistic"] > 0))
        self.assertTrue(all(result["critical_value"] > 0))
        expected_stats = np.array(
            [
                scipy_anderson(self.normal_df[col], dist="norm").statistic
                for col in self.normal_df.columns
            ]
        )
        expected_crits = np.array(
            [
                scipy_anderson(self.normal_df[col], dist="norm").critical_values[2]
                for col in self.normal_df.columns
            ]
        )
        np.testing.assert_almost_equal(
            result["statistic"].values, expected_stats, decimal=10
        )
        np.testing.assert_almost_equal(
            result["critical_value"].values, expected_crits, decimal=10
        )

    def test_adf_test(self):
        result = adf_test(self.normal_data, critical_value=5)
        self.assertIsInstance(result, dict)
        self.assertIn("statistic", result)
        self.assertIn("p-value", result)
        self.assertIn("critical_value_5%", result)
        self.assertTrue(0 <= result["p-value"] <= 1)
        expected_result = adfuller(self.normal_data, autolag="AIC", regression="c")
        np.testing.assert_almost_equal(
            result["statistic"], expected_result[0], decimal=10
        )
        np.testing.assert_almost_equal(
            result["p-value"], expected_result[1], decimal=10
        )
        np.testing.assert_almost_equal(
            result["critical_value_5%"], expected_result[4]["5%"], decimal=10
        )

    def test_adf_tests(self):
        result = adf_tests(self.normal_df, critical_value=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["statistic", "p-value", "critical_value_5%"]
            )
        )
        self.assertTrue(all(0 <= result["p-value"] <= 1))
        expected_results = np.array(
            [
                adfuller(self.normal_df[col], autolag="AIC", regression="c")
                for col in self.normal_df.columns
            ]
        )
        np.testing.assert_almost_equal(
            result["statistic"].values, expected_results[:, 0], decimal=10
        )
        np.testing.assert_almost_equal(
            result["p-value"].values, expected_results[:, 1], decimal=10
        )
        np.testing.assert_almost_equal(
            result["critical_value_5%"].values, expected_results[:, 4]["5%"], decimal=10
        )

    def test_calculate_vif(self):
        result = calculate_vif(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("VIF", result.columns)
        self.assertIn("hypothesis", result.columns)
        self.assertTrue(all(result["VIF"] > 0))
        self.assertTrue(all(result["hypothesis"].isin(["Accept", "Reject"])))
        X = add_constant(self.regression_data[["x1", "x2"]])
        expected_vifs = np.array(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        )
        np.testing.assert_almost_equal(result["VIF"].values, expected_vifs, decimal=10)

    def test_durbin_watson_test(self):
        result = durbin_watson_test(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("DW Statistic", result.columns)
        self.assertIn("Hypothesis", result.columns)
        self.assertTrue(result["DW Statistic"].iloc[0] > 0)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)
        model = OLSModel(
            data=self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()
        expected_dw = durbin_watson(results.resid)
        np.testing.assert_almost_equal(
            result["DW Statistic"].iloc[0], expected_dw, decimal=10
        )

    def test_breusch_pagan_test(self):
        result = breusch_pagan_test(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["BP Statistic", "p-value", "Hypothesis"]
            )
        )
        self.assertTrue(0 <= result["p-value"].iloc[0] <= 1)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)
        model = OLSModel(
            data=self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()
        expected_stat, expected_p, _, _ = het_breuschpagan(
            results.resid, results.model.exog
        )
        np.testing.assert_almost_equal(
            result["BP Statistic"].iloc[0], expected_stat, decimal=10
        )
        np.testing.assert_almost_equal(
            result["p-value"].iloc[0], expected_p, decimal=10
        )

    def test_white_test(self):
        result = white_test(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["White Statistic", "p-value", "Hypothesis"]
            )
        )
        self.assertTrue(0 <= result["p-value"].iloc[0] <= 1)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)
        model = OLSModel(
            data=self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        results = model.fit()
        expected_stat, expected_p, _, _ = het_white(results.resid, results.model.exog)
        np.testing.assert_almost_equal(
            result["White Statistic"].iloc[0], expected_stat, decimal=10
        )
        np.testing.assert_almost_equal(
            result["p-value"].iloc[0], expected_p, decimal=10
        )


class TestStatisticalTests(TestCase):
    def setUp(self):
        np.random.seed(42)
        self.normal_df = pd.DataFrame(
            {
                "var1": norm.rvs(size=1000),
                "var2": norm.rvs(size=1000),
                "var3": norm.rvs(size=1000),
            }
        )
        n = 1000
        x1 = norm.rvs(size=n)
        x2 = norm.rvs(size=n)
        y = 2 * x1 + 3 * x2 + norm.rvs(size=n)
        self.regression_data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def test_initialization(self):
        st = StatisticalTests(self.normal_df)
        self.assertTrue(st.data.equals(self.normal_df))
        self.assertIsNone(st.dependent_variable)
        self.assertEqual(st.independent_variables, ["var1", "var2", "var3"])
        self.assertEqual(st.control_variables, [])

    def test_initialization_with_dependent(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        self.assertTrue(st.data.equals(self.regression_data))
        self.assertEqual(st.dependent_variable, "y")
        self.assertEqual(st.independent_variables, ["x1", "x2"])
        self.assertEqual(st.control_variables, [])

    def test_initialization_with_control(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1"],
            control_variables=["x2"],
        )
        self.assertTrue(st.data.equals(self.regression_data))
        self.assertEqual(st.dependent_variable, "y")
        self.assertEqual(st.independent_variables, ["x1"])
        self.assertEqual(st.control_variables, ["x2"])

    def test_invalid_dependent_variable(self):
        with self.assertRaises(ArgumentValueError):
            StatisticalTests(self.normal_df, dependent_variable="invalid")

    def test_invalid_independent_variable(self):
        with self.assertRaises(ArgumentValueError):
            StatisticalTests(self.normal_df, independent_variables=["invalid"])

    def test_invalid_control_variable(self):
        with self.assertRaises(ArgumentValueError):
            StatisticalTests(self.normal_df, control_variables=["invalid"])

    def test_shapiro_test(self):
        st = StatisticalTests(self.normal_df)
        result = st.shapiro_test()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(all(col in result.columns for col in ["statistic", "p-value"]))
        self.assertTrue(all(0 <= result["p-value"] <= 1))
        self.assertTrue(all(result["statistic"] > 0))

    def test_anderson_test(self):
        st = StatisticalTests(self.normal_df)
        result = st.anderson_test(criteria=0.01)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(col in result.columns for col in ["statistic", "critical_value"])
        )
        self.assertTrue(all(result["statistic"] > 0))
        self.assertTrue(all(result["critical_value"] > 0))

    def test_adf_test(self):
        st = StatisticalTests(self.normal_df)
        result = st.adf_test(critical_value=5)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["statistic", "p-value", "critical_value_5%"]
            )
        )
        self.assertTrue(all(0 <= result["p-value"] <= 1))

    def test_calculate_vif(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        result = st.calculate_vif(threshold=5.0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("VIF", result.columns)
        self.assertIn("hypothesis", result.columns)
        self.assertTrue(all(result["VIF"] > 0))
        self.assertTrue(all(result["hypothesis"].isin(["Accept", "Reject"])))

    def test_durbin_watson_test(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        result = st.durbin_watson_test()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("DW Statistic", result.columns)
        self.assertIn("Hypothesis", result.columns)
        self.assertTrue(result["DW Statistic"].iloc[0] > 0)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)

    def test_breusch_pagan_test(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        result = st.breusch_pagan_test()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["BP Statistic", "p-value", "Hypothesis"]
            )
        )
        self.assertTrue(0 <= result["p-value"].iloc[0] <= 1)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)

    def test_white_test(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        result = st.white_test()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            all(
                col in result.columns
                for col in ["White Statistic", "p-value", "Hypothesis"]
            )
        )
        self.assertTrue(0 <= result["p-value"].iloc[0] <= 1)
        self.assertIsInstance(result["Hypothesis"].iloc[0], str)

    def test_repr(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        repr_str = repr(st)
        self.assertIsInstance(repr_str, str)
        self.assertIn("data.shape", repr_str)
        self.assertIn("dependent_variable", repr_str)
        self.assertIn("independent_variables", repr_str)

    def test_str(self):
        st = StatisticalTests(
            self.regression_data,
            dependent_variable="y",
            independent_variables=["x1", "x2"],
        )
        str_rep = str(st)
        self.assertIsInstance(str_rep, str)
        self.assertIn("Data Shape", str_rep)
        self.assertIn("Dependent Variable", str_rep)
        self.assertIn("Independent Variables", str_rep)


if __name__ == "__main__":
    unittest.main()
