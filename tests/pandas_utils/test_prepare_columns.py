import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.pandas_utils.prepare_columns import (
    prepare_bin_columns,
    prepare_bool_columns,
    prepare_categorical_columns,
    prepare_date_columns,
    prepare_int_columns,
    prepare_normalized_columns,
    prepare_quantile_columns,
    prepare_rank_columns,
    prepare_standardized_columns,
    prepare_str_columns,
    validate_columns,
)


class TestValidateColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            }
        )

    def test_single_string_column(self):
        result = validate_columns(self.df, "col1")
        self.assertEqual(list(result), ["col1"])

    def test_multiple_string_columns(self):
        result = validate_columns(self.df, ["col1", "col2"])
        self.assertEqual(list(result), ["col1", "col2"])

    def test_all_columns(self):
        result = validate_columns(self.df, ["col1", "col2", "col3"])
        self.assertEqual(list(result), ["col1", "col2", "col3"])

    def test_empty_list(self):
        result = validate_columns(self.df, [])
        self.assertEqual(list(result), [])

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            validate_columns(self.df, "nonexistent")
        self.assertIn(
            "Columns ['nonexistent'] not found in DataFrame", str(context.exception)
        )

    def test_nonexistent_columns(self):
        with self.assertRaises(ArgumentValueError) as context:
            validate_columns(self.df, ["col1", "nonexistent"])
        self.assertIn(
            "Columns ['nonexistent'] not found in DataFrame", str(context.exception)
        )

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            validate_columns(self.df, 123)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_invalid_column_elements(self):
        with self.assertRaises(ArgumentTypeError) as context:
            validate_columns(self.df, ["col1", 123])
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            validate_columns(empty_df, "col1")
        self.assertIn("Columns ['col1'] not found in DataFrame", str(context.exception))

    def test_none_input(self):
        with self.assertRaises(ArgumentTypeError) as context:
            validate_columns(self.df, None)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_tuple_input(self):
        result = validate_columns(self.df, ("col1", "col2"))
        self.assertEqual(list(result), ["col1", "col2"])

    def test_set_input(self):
        result = validate_columns(self.df, {"col1", "col2"})
        self.assertEqual(set(result), {"col1", "col2"})


class TestPrepareIntCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )

    def test_single_column_conversion(self):
        result = prepare_int_columns(self.df.copy(), columns="col1", nan_placeholder=0)
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 0],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        assert_frame_equal(result, expected)

    def test_multiple_columns_conversion(self):
        result = prepare_int_columns(
            self.df.copy(), columns=["col1", "col2"], nan_placeholder=0
        )
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 0],
                "col2": [4, 0, 6, 0],
                "col3": [None, None, None, None],
            }
        )
        assert_frame_equal(result, expected)

    def test_column_with_only_nans(self):
        result = prepare_int_columns(self.df.copy(), columns="col3", nan_placeholder=99)
        expected = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [99, 99, 99, 99],
            }
        )
        assert_frame_equal(result, expected)

    def test_column_not_in_dataframe(self):
        with self.assertRaises(ArgumentValueError):
            prepare_int_columns(
                self.df.copy(), columns="nonexistent_col", nan_placeholder=0
            )

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_int_columns(self.df.copy(), columns=123, nan_placeholder=0)

    def test_invalid_iterable_column_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_int_columns(self.df.copy(), columns=[1, 2, 3], nan_placeholder=0)

    def test_ignore_errors(self):
        result = prepare_int_columns(
            self.df.copy(), columns="col2", nan_placeholder=0, errors="ignore"
        )
        expected = DataFrame(
            {
                "col1": ["1", "2", "3", None],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        assert_frame_equal(result, expected)

    def test_invalid_error_handling(self):
        with self.assertRaises(ArgumentValueError):
            prepare_int_columns(
                self.df.copy(),
                columns="col2",
                nan_placeholder=0,
                errors="invalid_option",
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = prepare_int_columns(empty_df, columns=[], nan_placeholder=0)
        assert_frame_equal(result, empty_df)

    def test_empty_columns(self):
        result = prepare_int_columns(self.df, columns=[], nan_placeholder=0)
        assert_frame_equal(result, self.df)

    def test_no_conversion_needed(self):
        df = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = prepare_int_columns(df.copy(), columns="col1", nan_placeholder=0)
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_columns(
            df.copy(), columns="col1", nan_placeholder=0, errors="ignore"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)
        result = prepare_int_columns(
            df.copy(), columns="col1", nan_placeholder=0, errors="raise"
        )
        expected = DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        assert_frame_equal(result, expected, check_dtype=False)

    def test_custom_nan_placeholder(self):
        result = prepare_int_columns(
            self.df.copy(), columns="col1", nan_placeholder=999
        )
        expected = DataFrame(
            {
                "col1": [1, 2, 3, 999],
                "col2": ["4.5", "invalid", "6", None],
                "col3": [None, None, None, None],
            }
        )
        assert_frame_equal(result, expected)


class TestPrepareCategoricalColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": ["a", "b", "c", "a"],
                "B": ["x", "y", "z", "x"],
                "C": [1, 2, 3, 4],  # Non-categorical numeric column
            }
        )

    def test_single_column_no_categories(self):
        result = prepare_categorical_columns(self.df.copy(), columns="A")
        expected = self.df.copy()
        expected["A"] = pd.Categorical(expected["A"])
        assert_frame_equal(result, expected)

    def test_multiple_columns_no_categories(self):
        result = prepare_categorical_columns(self.df.copy(), columns=["A", "B"])
        expected = self.df.copy()
        expected["A"] = pd.Categorical(expected["A"])
        expected["B"] = pd.Categorical(expected["B"])
        assert_frame_equal(result, expected)

    def test_single_column_with_categories(self):
        result = prepare_categorical_columns(
            self.df.copy(), columns="A", categories=["c", "b", "a"], ordered=True
        )
        expected = self.df.copy()
        expected["A"] = pd.Categorical(
            expected["A"], categories=["c", "b", "a"], ordered=True
        )
        assert_frame_equal(result, expected)

    def test_multiple_columns_with_categories(self):
        result = prepare_categorical_columns(
            self.df.copy(), columns=["A", "B"], categories=["c", "b", "a"], ordered=True
        )
        expected = self.df.copy()
        expected["A"] = pd.Categorical(
            expected["A"], categories=["c", "b", "a"], ordered=True
        )
        expected["B"] = pd.Categorical(
            expected["B"], categories=["c", "b", "a"], ordered=True
        )
        assert_frame_equal(result, expected)

    def test_missing_columns(self):
        with self.assertRaises(ArgumentValueError):
            prepare_categorical_columns(self.df.copy(), columns="D")

    def test_non_string_column(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_categorical_columns(
                self.df.copy(), columns=10, categories=["1", "2", "3", "4"]
            )

    def test_inferred_categories(self):
        result = prepare_categorical_columns(self.df.copy(), columns="B")
        expected_categories = sorted(self.df["B"].unique())
        self.assertListEqual(result["B"].cat.categories.tolist(), expected_categories)

    def test_ordered_flag(self):
        result = prepare_categorical_columns(
            self.df.copy(), columns="A", categories=["a", "b", "c"], ordered=True
        )
        self.assertTrue(result["A"].cat.ordered)

    def test_unordered_flag(self):
        result = prepare_categorical_columns(
            self.df.copy(), columns="A", categories=["a", "b", "c"], ordered=False
        )
        self.assertFalse(result["A"].cat.ordered)

    def test_no_modification_to_untouched_columns(self):
        result = prepare_categorical_columns(self.df.copy(), columns="A")
        self.assertTrue(pd.api.types.is_numeric_dtype(result["C"]))


class TestPrepareRankColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": [10, 20, 20, 30],
                "B": [100, 50, 50, 25],
                "C": ["x", "y", "z", "w"],  # Non-numeric column
            }
        )

    def test_single_column_default_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns="A")
        expected = self.df.copy()
        expected["A"] = [1.0, 2.5, 2.5, 4.0]  # Default "average" ranking
        assert_frame_equal(result, expected)

    def test_multiple_columns_default_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns=["A", "B"])
        expected = self.df.copy()
        expected["A"] = [1.0, 2.5, 2.5, 4.0]
        expected["B"] = [4.0, 2.5, 2.5, 1.0]
        assert_frame_equal(result, expected)

    def test_single_column_descending_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns="A", ascending=False)
        expected = self.df.copy()
        expected["A"] = [4.0, 2.5, 2.5, 1.0]
        assert_frame_equal(result, expected)

    def test_single_column_min_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns="A", method="min")
        expected = self.df.copy()
        expected["A"] = [
            1.0,
            2.0,
            2.0,
            4.0,
        ]  # "min" ranking assigns lowest rank for ties
        assert_frame_equal(result, expected)

    def test_single_column_max_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns="A", method="max")
        expected = self.df.copy()
        expected["A"] = [
            1.0,
            3.0,
            3.0,
            4.0,
        ]  # "max" ranking assigns highest rank for ties
        assert_frame_equal(result, expected)

    def test_single_column_dense_ranking(self):
        result = prepare_rank_columns(self.df.copy(), columns="A", method="dense")
        expected = self.df.copy()
        expected["A"] = [1.0, 2.0, 2.0, 3.0]  # "dense" ranking skips no ranks
        assert_frame_equal(result, expected)

    def test_single_column_ordinal_ranking(self):
        with self.assertRaises(ArgumentValueError):
            prepare_rank_columns(self.df.copy(), columns="A", method="ordinal")

    def test_invalid_column(self):
        with self.assertRaises(ArgumentValueError):
            prepare_rank_columns(self.df.copy(), columns="D")

    def test_non_numeric_column(self):
        result = prepare_rank_columns(self.df.copy(), columns="C")
        expected = DataFrame({"C": [2.0, 3.0, 4.0, 1.0]})
        assert_frame_equal(result[["C"]], expected)

    def test_inferred_numeric_column(self):
        df = self.df.copy()
        df["D"] = [1.1, 2.2, 3.3, 4.4]
        result = prepare_rank_columns(df, columns="D")
        expected = df.copy()
        expected["D"] = [1.0, 2.0, 3.0, 4.0]
        assert_frame_equal(result, expected)

    def test_no_modification_to_untouched_columns(self):
        result = prepare_rank_columns(self.df.copy(), columns="A")
        self.assertTrue(result["B"].equals(self.df["B"]))


class TestPrepareStandardizedColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "w", "v"],  # Non-numeric column
            }
        )

    def test_single_column_standardization(self):
        result = prepare_standardized_columns(self.df.copy(), columns="A")
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].mean()) / self.df["A"].std()
        assert_frame_equal(result, expected)

    def test_multiple_columns_standardization(self):
        result = prepare_standardized_columns(self.df.copy(), columns=["A", "B"])
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].mean()) / self.df["A"].std()
        expected["B"] = (self.df["B"] - self.df["B"].mean()) / self.df["B"].std()
        assert_frame_equal(result, expected)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            prepare_standardized_columns(self.df.copy(), columns="D")

    def test_non_numeric_column(self):
        with self.assertRaises(TypeError):
            prepare_standardized_columns(self.df.copy(), columns="C")

    def test_inferred_numeric_column(self):
        df = self.df.copy()
        df["D"] = [1.1, 2.2, 3.3, 4.4, 5.5]
        result = prepare_standardized_columns(df, columns="D")
        expected = df.copy()
        expected["D"] = (df["D"] - df["D"].mean()) / df["D"].std()
        assert_frame_equal(result, expected)

    def test_standardized_column_mean(self):
        result = prepare_standardized_columns(self.df.copy(), columns="A")
        self.assertAlmostEqual(result["A"].mean(), 0, places=6)

    def test_standardized_column_std(self):
        result = prepare_standardized_columns(self.df.copy(), columns="A")
        self.assertAlmostEqual(result["A"].std(), 1, places=6)

    def test_no_modification_to_untouched_columns(self):
        result = prepare_standardized_columns(self.df.copy(), columns="A")
        self.assertTrue(result["B"].equals(self.df["B"]))

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = prepare_standardized_columns(empty_df, columns=[])
        assert_frame_equal(result, empty_df)

    def test_all_columns_standardized(self):
        numeric_columns = self.df.select_dtypes(include=np.number).columns.tolist()
        result = prepare_standardized_columns(self.df.copy(), columns=numeric_columns)
        expected = self.df.copy()
        for col in numeric_columns:
            expected[col] = (self.df[col] - self.df[col].mean()) / self.df[col].std()
        assert_frame_equal(result, expected)


class TestPrepareNormalizedColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "w", "v"],  # Non-numeric column
            }
        )

    def test_single_column_default_range(self):
        result = prepare_normalized_columns(self.df.copy(), columns="A")
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].min()) / (
            self.df["A"].max() - self.df["A"].min()
        )
        assert_frame_equal(result, expected)

    def test_multiple_columns_default_range(self):
        result = prepare_normalized_columns(self.df.copy(), columns=["A", "B"])
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].min()) / (
            self.df["A"].max() - self.df["A"].min()
        )
        expected["B"] = (self.df["B"] - self.df["B"].min()) / (
            self.df["B"].max() - self.df["B"].min()
        )
        assert_frame_equal(result, expected)

    def test_single_column_custom_range(self):
        result = prepare_normalized_columns(
            self.df.copy(), columns="A", range_min=-1, range_max=1
        )
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].min()) / (
            self.df["A"].max() - self.df["A"].min()
        ) * (1 - (-1)) + (-1)
        assert_frame_equal(result, expected)

    def test_multiple_columns_custom_range(self):
        result = prepare_normalized_columns(
            self.df.copy(), columns=["A", "B"], range_min=10, range_max=20
        )
        expected = self.df.copy()
        expected["A"] = (self.df["A"] - self.df["A"].min()) / (
            self.df["A"].max() - self.df["A"].min()
        ) * (20 - 10) + 10
        expected["B"] = (self.df["B"] - self.df["B"].min()) / (
            self.df["B"].max() - self.df["B"].min()
        ) * (20 - 10) + 10
        assert_frame_equal(result, expected)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            prepare_normalized_columns(self.df.copy(), columns="D")

    def test_non_numeric_column(self):
        with self.assertRaises(TypeError):
            prepare_normalized_columns(self.df.copy(), columns="C")

    def test_no_modification_to_untouched_columns(self):
        result = prepare_normalized_columns(self.df.copy(), columns="A")
        self.assertTrue(result["B"].equals(self.df["B"]))

    def test_all_columns_normalized(self):
        numeric_columns = self.df.select_dtypes(include=np.number).columns.tolist()
        result = prepare_normalized_columns(self.df.copy(), columns=numeric_columns)
        expected = self.df.copy()
        for col in numeric_columns:
            expected[col] = (self.df[col] - self.df[col].min()) / (
                self.df[col].max() - self.df[col].min()
            )
        assert_frame_equal(result, expected)

    def test_constant_column(self):
        constant_df = self.df.copy()
        constant_df["A"] = 5
        result = prepare_normalized_columns(constant_df.copy(), columns="A")
        expected = constant_df.copy()
        expected["A"] = 0.0  # All values should normalize to range_min since min == max
        assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        result = prepare_normalized_columns(empty_df, columns=[])
        assert_frame_equal(result, empty_df)


class TestPrepareBinColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "w", "v"],  # Non-numeric column
            }
        )

    def test_single_column_equal_bins(self):
        result = prepare_bin_columns(self.df.copy(), columns="A", bins=2)
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=2)
        assert_frame_equal(result, expected)

    def test_multiple_columns_equal_bins(self):
        result = prepare_bin_columns(self.df.copy(), columns=["A", "B"], bins=3)
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=3)
        expected["B"] = pd.cut(self.df["B"], bins=3)
        assert_frame_equal(result, expected)

    def test_single_column_custom_bins(self):
        custom_bins = [0, 2, 4, 6]
        result = prepare_bin_columns(self.df.copy(), columns="A", bins=custom_bins)
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=custom_bins)
        assert_frame_equal(result, expected)

    def test_multiple_columns_custom_bins(self):
        custom_bins = [0, 20, 40, 60]
        result = prepare_bin_columns(
            self.df.copy(), columns=["A", "B"], bins=custom_bins
        )
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=custom_bins)
        expected["B"] = pd.cut(self.df["B"], bins=custom_bins)
        assert_frame_equal(result, expected)

    def test_single_column_custom_labels(self):
        labels = ["Low", "High"]
        result = prepare_bin_columns(self.df.copy(), columns="A", bins=2, labels=labels)
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=2, labels=labels)
        assert_frame_equal(result, expected)

    def test_multiple_columns_custom_labels(self):
        labels = ["Low", "Medium", "High"]
        result = prepare_bin_columns(
            self.df.copy(), columns=["A", "B"], bins=3, labels=labels
        )
        expected = self.df.copy()
        expected["A"] = pd.cut(self.df["A"], bins=3, labels=labels)
        expected["B"] = pd.cut(self.df["B"], bins=3, labels=labels)
        assert_frame_equal(result, expected)

    def test_missing_column(self):
        with self.assertRaises(ArgumentValueError):
            prepare_bin_columns(self.df.copy(), columns="D", bins=3)

    def test_non_numeric_column(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_bin_columns(self.df.copy(), columns="C", bins=3)

    def test_bins_mismatch_with_labels(self):
        with self.assertRaises(ArgumentValueError):
            prepare_bin_columns(
                self.df.copy(), columns="A", bins=3, labels=["Low", "High"]
            )

    def test_no_modification_to_untouched_columns(self):
        result = prepare_bin_columns(self.df.copy(), columns="A", bins=3)
        self.assertTrue(result["B"].equals(self.df["B"]))


class TestPrepareQuantileColumns(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["x", "y", "z", "w", "v"],  # Non-numeric column
            }
        )

    def test_single_column_default_quantiles(self):
        result = prepare_quantile_columns(self.df.copy(), columns="A", quantiles=2)
        expected = self.df.copy()
        expected["A"] = pd.qcut(self.df["A"], q=2)
        assert_frame_equal(result, expected)

    def test_multiple_columns_default_quantiles(self):
        result = prepare_quantile_columns(
            self.df.copy(), columns=["A", "B"], quantiles=3
        )
        expected = self.df.copy()
        expected["A"] = pd.qcut(self.df["A"], q=3)
        expected["B"] = pd.qcut(self.df["B"], q=3)
        assert_frame_equal(result, expected)

    def test_single_column_custom_labels(self):
        labels = ["Low", "Medium", "High"]
        result = prepare_quantile_columns(
            self.df.copy(), columns="A", quantiles=3, labels=labels
        )
        expected = self.df.copy()
        expected["A"] = pd.qcut(self.df["A"], q=3, labels=labels)
        assert_frame_equal(result, expected)

    def test_multiple_columns_custom_labels(self):
        labels = ["Low", "Medium", "High"]
        result = prepare_quantile_columns(
            self.df.copy(), columns=["A", "B"], quantiles=3, labels=labels
        )
        expected = self.df.copy()
        expected["A"] = pd.qcut(self.df["A"], q=3, labels=labels)
        expected["B"] = pd.qcut(self.df["B"], q=3, labels=labels)
        assert_frame_equal(result, expected)

    def test_invalid_quantiles(self):
        with self.assertRaises(ArgumentValueError):
            prepare_quantile_columns(self.df.copy(), columns="A", quantiles=1)

    def test_invalid_labels_length(self):
        with self.assertRaises(ArgumentValueError):
            prepare_quantile_columns(
                self.df.copy(), columns="A", quantiles=3, labels=["Low", "High"]
            )

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            prepare_quantile_columns(self.df.copy(), columns="D", quantiles=3)

    def test_non_numeric_column(self):
        with self.assertRaises(TypeError):
            prepare_quantile_columns(self.df.copy(), columns="C", quantiles=3)

    def test_no_modification_to_untouched_columns(self):
        result = prepare_quantile_columns(self.df.copy(), columns="A", quantiles=3)
        self.assertTrue(result["B"].equals(self.df["B"]))


class TestPrepareStrCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {"col1": [1, 2, 3], "col2": [4.5, 5.5, None], "col3": ["a", "b", "c"]}
        )

    def test_single_column_conversion(self):
        result = prepare_str_columns(self.df.copy(), columns="col1")
        self.assertTrue((result["col1"] == ["1", "2", "3"]).all())
        self.assertEqual(result["col1"].dtype, "object")

    def test_multiple_column_conversion(self):
        result = prepare_str_columns(self.df.copy(), columns=["col1", "col2"])
        self.assertTrue((result["col1"] == ["1", "2", "3"]).all())
        self.assertTrue((result["col2"] == ["4.5", "5.5", "nan"]).all())
        self.assertEqual(result["col1"].dtype, "object")
        self.assertEqual(result["col2"].dtype, "object")

    def test_no_conversion_needed(self):
        result = prepare_str_columns(self.df.copy(), columns="col3")
        self.assertTrue((result["col3"] == ["a", "b", "c"]).all())
        self.assertEqual(result["col3"].dtype, "object")

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_columns(self.df.copy(), columns="nonexistent_col")
        self.assertIn("Columns ['nonexistent_col'] not found", str(context.exception))

    def test_mixed_column_input(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_columns(self.df.copy(), columns=["col1", "nonexistent_col"])
        self.assertIn("Columns ['nonexistent_col'] not found", str(context.exception))

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_str_columns(self.df.copy(), columns=123)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_invalid_cols_elements(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_str_columns(self.df.copy(), columns=["col1", 123])
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings",
            str(context.exception),
        )

    def test_empty_cols_list(self):
        result = prepare_str_columns(self.df.copy(), columns=[])
        assert_frame_equal(result, self.df)

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            prepare_str_columns(empty_df, columns="col1")
        self.assertIn("Columns ['col1'] not found in DataFrame", str(context.exception))

    def test_preserves_other_columns(self):
        result = prepare_str_columns(self.df.copy(), columns="col1")
        assert_frame_equal(result[["col2", "col3"]], self.df[["col2", "col3"]])

    def test_large_dataframe(self):
        large_df = DataFrame(
            {
                "col1": range(10000),
                "col2": [str(x) for x in range(10000)],
            }
        )
        result = prepare_str_columns(large_df, columns="col1")
        self.assertTrue((result["col1"] == large_df["col1"].astype(str)).all())


class TestPrepareDateCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "valid_dates": ["2021-01-01", "2021-02-01", None],
                "invalid_dates": ["invalid_date", "2021-01-01", None],
                "mixed_dates": ["2021-01-01", "invalid_date", "2021-02-01"],
                "custom_format": ["01-01-2021", "02-01-2021", None],
                "mixed_formats": ["2021-01-01", "01-02-2021", None],
            }
        )

    def test_basic_conversion(self):
        result = prepare_date_columns(
            self.df.copy(), columns="valid_dates", nan_placeholder="2000-01-01"
        )
        self.assertTrue(isinstance(result["valid_dates"].iloc[0], pd.Timestamp))
        self.assertEqual(result["valid_dates"].iloc[2], pd.Timestamp("2000-01-01"))

    def test_multiple_column_conversion(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns=["valid_dates", "mixed_dates"],
            nan_placeholder="2000-01-01",
        )
        self.assertTrue(isinstance(result["valid_dates"].iloc[0], pd.Timestamp))
        self.assertTrue(isinstance(result["mixed_dates"].iloc[0], pd.Timestamp))
        self.assertEqual(result["mixed_dates"].iloc[1], pd.Timestamp("2000-01-01"))

    def test_invalid_date_handling_coerce(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns="invalid_dates",
            nan_placeholder="2000-01-01",
            errors="coerce",
        )
        self.assertEqual(result["invalid_dates"].iloc[0], pd.Timestamp("2000-01-01"))

    def test_invalid_date_handling_ignore(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns="invalid_dates",
            nan_placeholder="2000-01-01",
            errors="ignore",
        )
        self.assertEqual(result["invalid_dates"].iloc[0], "invalid_date")
        self.assertTrue(pd.isna(result["invalid_dates"].iloc[2]))

    def test_invalid_date_handling_raise(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_date_columns(
                self.df.copy(),
                columns="invalid_dates",
                nan_placeholder="2000-01-01",
                errors="raise",
            )

    def test_custom_date_format(self):
        df = DataFrame({"custom_dates": ["01-01-2021", "02-01-2021", None]})
        result = prepare_date_columns(
            df,
            columns="custom_dates",
            nan_placeholder="2000-01-01",
            date_format="%d-%m-%Y",
        )
        self.assertEqual(result["custom_dates"].iloc[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result["custom_dates"].iloc[2], pd.Timestamp("2000-01-01"))

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError):
            prepare_date_columns(
                self.df.copy(), columns=123, nan_placeholder="2000-01-01"
            )

    def test_nonexistent_column(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_columns(
                self.df.copy(), columns="nonexistent", nan_placeholder="2000-01-01"
            )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError):
            prepare_date_columns(
                empty_df, columns="any_column", nan_placeholder="2000-01-01"
            )

    def test_missing_placeholder(self):
        df = DataFrame({"dates": ["2021-01-01", None]})
        result = prepare_date_columns(df, columns="dates", nan_placeholder=None)
        self.assertTrue(pd.isna(result["dates"].iloc[1]))

    def test_preserves_other_columns(self):
        result = prepare_date_columns(
            self.df.copy(), columns="valid_dates", nan_placeholder="2000-01-01"
        )
        assert_frame_equal(result[["invalid_dates"]], self.df[["invalid_dates"]])

    def test_large_dataframe(self):
        large_df = DataFrame({"dates": ["2021-01-01"] * 100000 + [None] * 100000})
        result = prepare_date_columns(
            large_df, columns="dates", nan_placeholder="2000-01-01"
        )
        self.assertEqual(result["dates"].iloc[-1], pd.Timestamp("2000-01-01"))

    def test_single_format_none(self):
        result = prepare_date_columns(
            self.df.copy(), columns="valid_dates", nan_placeholder="2000-01-01"
        )
        self.assertTrue(isinstance(result["valid_dates"].iloc[0], pd.Timestamp))
        self.assertEqual(result["valid_dates"].iloc[0], pd.Timestamp("2021-01-01"))

    def test_single_format_string(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns="custom_format",
            nan_placeholder="2000-01-01",
            date_format="%d-%m-%Y",
        )
        self.assertTrue(isinstance(result["custom_format"].iloc[0], pd.Timestamp))
        self.assertEqual(result["custom_format"].iloc[0], pd.Timestamp("2021-01-01"))

    def test_multiple_formats(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns=["valid_dates", "custom_format", "mixed_formats"],
            nan_placeholder="2000-01-01",
            date_format=[None, "%d-%m-%Y", "%Y-%m-%d"],
        )
        self.assertEqual(result["valid_dates"].iloc[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result["custom_format"].iloc[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result["mixed_formats"].iloc[0], pd.Timestamp("2021-01-01"))

    def test_invalid_format_length(self):
        with self.assertRaises(ArgumentValueError):
            prepare_date_columns(
                self.df.copy(),
                columns=["valid_dates", "custom_format"],
                nan_placeholder="2000-01-01",
                date_format=["%Y-%m-%d"],
            )

    def test_mixed_formats_with_none(self):
        result = prepare_date_columns(
            self.df.copy(),
            columns=["valid_dates", "custom_format"],
            nan_placeholder="2000-01-01",
            date_format=[None, "%d-%m-%Y"],
        )
        self.assertEqual(result["valid_dates"].iloc[0], pd.Timestamp("2021-01-01"))
        self.assertEqual(result["custom_format"].iloc[0], pd.Timestamp("2021-01-01"))


class TestPrepareBoolCols(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "col1": [1, 0, None],
                "col2": [True, False, None],
                "col3": ["yes", "no", None],
            }
        )

    def test_single_column_conversion(self):
        result = prepare_bool_columns(self.df.copy(), columns="col1")
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue((result["col1"] == [True, False, False]).all())

    def test_multiple_column_conversion(self):
        result = prepare_bool_columns(self.df.copy(), columns=["col1", "col2"])
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue(result["col2"].dtype == bool)
        self.assertTrue((result["col1"] == [True, False, False]).all())
        self.assertTrue((result["col2"] == [True, False, False]).all())

    def test_with_nan_placeholder_true(self):
        result = prepare_bool_columns(
            self.df.copy(), columns=["col1"], nan_placeholder=True
        )
        self.assertTrue((result["col1"] == [True, False, True]).all())

    def test_with_nan_placeholder_false(self):
        result = prepare_bool_columns(
            self.df.copy(), columns=["col1"], nan_placeholder=False
        )
        self.assertTrue((result["col1"] == [True, False, False]).all())

    def test_preserves_other_columns(self):
        result = prepare_bool_columns(self.df.copy(), columns=["col1"])
        assert_frame_equal(result[["col2", "col3"]], self.df[["col2", "col3"]])

    def test_invalid_cols_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_bool_columns(self.df.copy(), columns=123)
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings.",
            str(context.exception),
        )

    def test_invalid_cols_elements(self):
        with self.assertRaises(ArgumentTypeError) as context:
            prepare_bool_columns(self.df.copy(), columns=["col1", 123])
        self.assertIn(
            "Argument 'cols' must be a string or an iterable of strings.",
            str(context.exception),
        )

    def test_nonexistent_columns(self):
        with self.assertRaises(ArgumentValueError) as context:
            prepare_bool_columns(self.df.copy(), columns=["nonexistent"])
        self.assertIn(
            "Columns ['nonexistent'] not found in DataFrame.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame()
        with self.assertRaises(ArgumentValueError) as context:
            prepare_bool_columns(empty_df, columns="col1")
        self.assertIn(
            "Columns ['col1'] not found in DataFrame.", str(context.exception)
        )

    def test_empty_cols_list(self):
        result = prepare_bool_columns(self.df.copy(), columns=[])
        assert_frame_equal(result, self.df)

    def test_large_dataframe(self):
        large_df = DataFrame({"col1": [1, 0, None] * 100000})
        result = prepare_bool_columns(large_df, columns="col1", nan_placeholder=False)
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue((result["col1"].iloc[2] == False))

    def test_mixed_data_types(self):
        df = DataFrame(
            {
                "col1": [1, "yes", None],
                "col2": [0, "no", "yes"],
            }
        )
        result = prepare_bool_columns(
            df.copy(), columns=["col1", "col2"], nan_placeholder=True
        )
        self.assertTrue(result["col1"].dtype == bool)
        self.assertTrue(result["col2"].dtype == bool)
        self.assertTrue((result["col1"] == [True, True, True]).all())
        self.assertTrue((result["col2"] == [False, True, True]).all())

    def test_preserves_column_order(self):
        result = prepare_bool_columns(self.df.copy(), columns=["col1"])
        self.assertTrue(list(result.columns) == ["col1", "col2", "col3"])

    def test_no_changes_for_all_boolean_columns(self):
        df = DataFrame({"bool_col": [True, False, True]})
        result = prepare_bool_columns(df.copy(), columns="bool_col")
        self.assertTrue((result["bool_col"] == df["bool_col"]).all())


if __name__ == "__main__":
    unittest.main()
