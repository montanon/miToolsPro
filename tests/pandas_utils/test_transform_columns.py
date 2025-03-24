import unittest
from unittest import TestCase

from numpy import log
from pandas import DataFrame, MultiIndex
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal

from mitoolspro.exceptions import (
    ArgumentTypeError,
    ArgumentValueError,
)
from mitoolspro.pandas_utils.transform_columns import (
    ADDED_COLUMN_NAME,
    DIVIDED_COLUMN_NAME,
    GROWTH_COLUMN_NAME,
    GROWTH_PCT_COLUMN_NAME,
    MULTIPLIED_COLUMN_NAME,
    SHIFTED_COLUMN_NAME,
    SUBTRACTED_COLUMN_NAME,
    add_columns,
    divide_columns,
    growth_columns,
    multiply_columns,
    shift_columns,
    subtract_columns,
    transform_columns,
)


class TestTransformColumns(TestCase):
    def setUp(self):
        self.multiidx_df = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_transform_multiidx_log(self):
        result = transform_columns(self.multiidx_df, log, ["one", "three"], level=-1)
        expected_columns = [("A", "one_log"), ("B", "three_log")]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = (col[0], col[1].replace("_log", ""))
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(result[col] == log(self.multiidx_df[original_col].replace(0, 1e-6)))
            )

    def test_transform_singleidx_log(self):
        result = transform_columns(self.singleidx_df, log, ["one", "three"])
        expected_columns = ["one_log", "three_log"]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = col.replace("_log", "")
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(
                    result[col] == log(self.singleidx_df[original_col].replace(0, 1e-6))
                )
            )

    def test_transform_with_custom_rename(self):
        result = transform_columns(self.singleidx_df, log, ["one"], rename="custom")
        expected_columns = ["one_custom"]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_transform_with_invalid_column(self):
        with self.assertRaises(ArgumentValueError):
            transform_columns(self.multiidx_df, log, ["four"])

    def test_transform_with_invalid_function(self):
        def invalid_function(x):
            return x + "invalid"

        with self.assertRaises(ArgumentValueError):
            transform_columns(self.multiidx_df, invalid_function, ["one"])

    def test_transform_multiidx_with_level_positional(self):
        result = transform_columns(self.multiidx_df, log, ["one", "three"], level=1)
        expected_columns = [("A", "one_log"), ("B", "three_log")]
        self.assertListEqual(list(result.columns), expected_columns)
        for col in result.columns:
            original_col = (col[0], col[1].replace("_log", ""))
            self.assertTrue(is_numeric_dtype(result[col]))
            self.assertTrue(
                all(result[col] == log(self.multiidx_df[original_col].replace(0, 1e-6)))
            )

    def test_transform_with_non_callable_transformation(self):
        with self.assertRaises(ArgumentTypeError):
            transform_columns(self.singleidx_df, "not_callable", ["one"])

    def test_transform_with_tuple_column_not_matching_multiindex(self):
        with self.assertRaises(ValueError):
            transform_columns(self.multiidx_df, log, [("A", "invalid")])


class TestGrowthColumns(TestCase):
    def setUp(self):
        self.multiidx_df = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_variation_singleidx_absolute(self):
        result = growth_columns(self.singleidx_df, ["one", "three"], t=1)
        expected_columns = [
            f"one_{GROWTH_COLUMN_NAME.format(1)}",
            f"three_{GROWTH_COLUMN_NAME.format(1)}",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"one_{GROWTH_COLUMN_NAME.format(1)}": [None, 1, 1],
                f"three_{GROWTH_COLUMN_NAME.format(1)}": [None, 1, 1],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_singleidx_pct(self):
        result = growth_columns(self.singleidx_df, ["one", "three"], t=1, pct=True)
        expected_columns = [
            f"one_{GROWTH_PCT_COLUMN_NAME.format(1)}",
            f"three_{GROWTH_PCT_COLUMN_NAME.format(1)}",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"one_{GROWTH_PCT_COLUMN_NAME.format(1)}": [None, 50.0, 33.333333],
                f"three_{GROWTH_PCT_COLUMN_NAME.format(1)}": [
                    None,
                    12.5,
                    11.111111,
                ],  # Rounded for clarity
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_multiidx_absolute(self):
        result = growth_columns(self.multiidx_df, ["one", "three"], t=1, level=-1)
        expected_columns = [
            ("A", f"one_{GROWTH_COLUMN_NAME.format(1)}"),
            ("B", f"three_{GROWTH_COLUMN_NAME.format(1)}"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{GROWTH_COLUMN_NAME.format(1)}"): [None, 1, 1],
                ("B", f"three_{GROWTH_COLUMN_NAME.format(1)}"): [None, 1, 1],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_multiidx_pct(self):
        result = growth_columns(
            self.multiidx_df, ["one", "three"], t=1, pct=True, level=-1
        )
        expected_columns = [
            ("A", f"one_{GROWTH_PCT_COLUMN_NAME.format(1)}"),
            ("B", f"three_{GROWTH_PCT_COLUMN_NAME.format(1)}"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{GROWTH_PCT_COLUMN_NAME.format(1)}"): [
                    None,
                    50.0,
                    33.333333,
                ],
                ("B", f"three_{GROWTH_PCT_COLUMN_NAME.format(1)}"): [
                    None,
                    12.5,
                    11.111111,
                ],  # Rounded for clarity
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_variation_with_invalid_t(self):
        with self.assertRaises(ArgumentTypeError):
            growth_columns(self.singleidx_df, ["one"], t="invalid")

    def test_variation_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            growth_columns(self.singleidx_df, ["invalid_column"], t=1)

    def test_variation_with_custom_rename(self):
        result = growth_columns(self.singleidx_df, ["one"], t=1, rename="custom_name")
        expected_columns = ["one_custom_name"]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_variation_multiidx_with_positional_level(self):
        result = growth_columns(self.multiidx_df, ["one", "three"], t=1, level=1)
        expected_columns = [
            ("A", f"one_{GROWTH_COLUMN_NAME.format(1)}"),
            ("B", f"three_{GROWTH_COLUMN_NAME.format(1)}"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)

    def test_variation_non_numeric_data(self):
        df_non_numeric = DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
        with self.assertRaises(ArgumentValueError):
            growth_columns(df_non_numeric, ["A"], t=1)


class TestShiftColumns(TestCase):
    def setUp(self):
        self.multiidx_df = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.multiidx_df.columns = MultiIndex.from_tuples(self.multiidx_df.columns)
        self.singleidx_df = DataFrame(
            {
                "one": [1, 2, 3],
                "two": [4, 5, 6],
                "three": [7, 8, 9],
            }
        )

    def test_shift_singleidx(self):
        result = shift_columns(self.singleidx_df, ["one", "three"], t=1)
        expected_columns = [
            f"one_{SHIFTED_COLUMN_NAME.format(1)}",
            f"three_{SHIFTED_COLUMN_NAME.format(1)}",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"one_{SHIFTED_COLUMN_NAME.format(1)}": [None, 1, 2],
                f"three_{SHIFTED_COLUMN_NAME.format(1)}": [None, 7, 8],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_multiidx(self):
        result = shift_columns(self.multiidx_df, ["one", "three"], t=1, level=-1)
        expected_columns = [
            ("A", f"one_{SHIFTED_COLUMN_NAME.format(1)}"),
            ("B", f"three_{SHIFTED_COLUMN_NAME.format(1)}"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{SHIFTED_COLUMN_NAME.format(1)}"): [None, 1, 2],
                ("B", f"three_{SHIFTED_COLUMN_NAME.format(1)}"): [None, 7, 8],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_singleidx_with_custom_rename(self):
        result = shift_columns(self.singleidx_df, ["one"], t=1, rename="custom_name")
        expected_columns = ["one_custom_name"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame({"one_custom_name": [None, 1, 2]})
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_multiidx_with_positional_level(self):
        result = shift_columns(self.multiidx_df, ["one", "three"], t=1, level=1)
        expected_columns = [
            ("A", f"one_{SHIFTED_COLUMN_NAME.format(1)}"),
            ("B", f"three_{SHIFTED_COLUMN_NAME.format(1)}"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{SHIFTED_COLUMN_NAME.format(1)}"): [None, 1, 2],
                ("B", f"three_{SHIFTED_COLUMN_NAME.format(1)}"): [None, 7, 8],
            }
        )
        assert_frame_equal(result.reset_index(drop=True), expected_values)

    def test_shift_with_invalid_t(self):
        with self.assertRaises(ArgumentTypeError):
            shift_columns(self.singleidx_df, ["one"], t="invalid")

    def test_shift_with_invalid_column(self):
        with self.assertRaises(ArgumentValueError):
            shift_columns(self.singleidx_df, ["invalid_column"], t=1)


class TestAddColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        arrays = [["A", "A", "B"], ["one", "two", "three"]]
        self.df_multi = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.df_multi.columns = MultiIndex.from_arrays(arrays)

    def test_add_singleidx(self):
        result = add_columns(self.df_single, ["A", "B"], "C")
        expected_columns = [
            f"A_{ADDED_COLUMN_NAME.format('C')}",
            f"B_{ADDED_COLUMN_NAME.format('C')}",
        ]

        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"A_{ADDED_COLUMN_NAME.format('C')}": [8, 10, 12],
                f"B_{ADDED_COLUMN_NAME.format('C')}": [11, 13, 15],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_add_multiidx(self):
        result = add_columns(
            self.df_multi, [("A", "one"), ("A", "two")], ("B", "three")
        )
        expected_columns = [
            ("A", f"one_{ADDED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{ADDED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{ADDED_COLUMN_NAME.format('B')},three"): [8, 10, 12],
                ("A", f"two_{ADDED_COLUMN_NAME.format('B')},three"): [11, 13, 15],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_add_with_custom_rename(self):
        result = add_columns(self.df_single, ["A"], "C", rename="added_custom")
        expected_columns = ["A_added_custom"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame({"A_added_custom": [8, 10, 12]})
        assert_frame_equal(result, expected_values)

    def test_add_multiidx_with_positional_level(self):
        result = add_columns(self.df_multi, ["one", "two"], "three", level=-1)
        expected_columns = [
            ("A", f"one_{ADDED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{ADDED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{ADDED_COLUMN_NAME.format('B')},three"): [8, 10, 12],
                ("A", f"two_{ADDED_COLUMN_NAME.format('B')},three"): [11, 13, 15],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_add_with_invalid_column_to_add(self):
        with self.assertRaises(ArgumentValueError):
            add_columns(self.df_single, ["A", "B"], ["C", "B"])

    def test_add_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            add_columns(self.df_single, ["A"], "D")

    def test_add_non_numeric_data(self):
        df_non_numeric = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        with self.assertRaises(ArgumentValueError):
            add_columns(df_non_numeric, ["B"], "A")


class TestSubtractColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        arrays = [["A", "A", "B"], ["one", "two", "three"]]
        self.df_multi = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [7, 8, 9],
            }
        )
        self.df_multi.columns = MultiIndex.from_arrays(arrays)

    def test_subtract_singleidx(self):
        result = subtract_columns(self.df_single, ["A", "B"], "C")
        expected_columns = [
            f"A_{SUBTRACTED_COLUMN_NAME.format('C')}",
            f"B_{SUBTRACTED_COLUMN_NAME.format('C')}",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"A_{SUBTRACTED_COLUMN_NAME.format('C')}": [-6, -6, -6],
                f"B_{SUBTRACTED_COLUMN_NAME.format('C')}": [-3, -3, -3],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_subtract_multiidx(self):
        result = subtract_columns(
            self.df_multi, [("A", "one"), ("A", "two")], ("B", "three")
        )
        expected_columns = [
            ("A", f"one_{SUBTRACTED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{SUBTRACTED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{SUBTRACTED_COLUMN_NAME.format('B')},three"): [-6, -6, -6],
                ("A", f"two_{SUBTRACTED_COLUMN_NAME.format('B')},three"): [-3, -3, -3],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_subtract_with_custom_rename(self):
        result = subtract_columns(
            self.df_single, ["A"], "C", rename="subtracted_custom"
        )
        expected_columns = ["A_subtracted_custom"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame({"A_subtracted_custom": [-6, -6, -6]})
        assert_frame_equal(result, expected_values)

    def test_subtract_multiidx_with_positional_level(self):
        result = subtract_columns(self.df_multi, ["one", "two"], "three", level=-1)
        expected_columns = [
            ("A", f"one_{SUBTRACTED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{SUBTRACTED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{SUBTRACTED_COLUMN_NAME.format('B')},three"): [-6, -6, -6],
                ("A", f"two_{SUBTRACTED_COLUMN_NAME.format('B')},three"): [-3, -3, -3],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_subtract_with_invalid_column_to_subtract(self):
        with self.assertRaises(ArgumentValueError):
            subtract_columns(self.df_single, ["A", "B"], ["C", "B"])

    def test_subtract_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            subtract_columns(self.df_single, ["A"], "D")

    def test_subtract_non_numeric_data(self):
        df_non_numeric = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        with self.assertRaises(ArgumentValueError):
            subtract_columns(df_non_numeric, ["B"], "A")


class TestDivideColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [2, 4, 6], "B": [8, 10, 12], "C": [2, 2, 2]})
        arrays = [["A", "A", "B"], ["one", "two", "three"]]
        self.df_multi = DataFrame(
            {
                ("A", "one"): [2, 4, 6],
                ("A", "two"): [8, 10, 12],
                ("B", "three"): [2, 2, 2],
            }
        )
        self.df_multi.columns = MultiIndex.from_arrays(arrays)

    def test_divide_singleidx(self):
        result = divide_columns(self.df_single, ["A", "B"], "C")
        expected_columns = [
            f"A_{DIVIDED_COLUMN_NAME.format('C')}",
            f"B_{DIVIDED_COLUMN_NAME.format('C')}",
        ]

        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"A_{DIVIDED_COLUMN_NAME.format('C')}": [1.0, 2.0, 3.0],
                f"B_{DIVIDED_COLUMN_NAME.format('C')}": [4.0, 5.0, 6.0],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_divide_multiidx(self):
        result = divide_columns(
            self.df_multi, [("A", "one"), ("A", "two")], ("B", "three")
        )
        expected_columns = [
            ("A", f"one_{DIVIDED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{DIVIDED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{DIVIDED_COLUMN_NAME.format('B')},three"): [1.0, 2.0, 3.0],
                ("A", f"two_{DIVIDED_COLUMN_NAME.format('B')},three"): [4.0, 5.0, 6.0],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_divide_with_custom_rename(self):
        result = divide_columns(self.df_single, ["A"], "C", rename="divided_custom")
        expected_columns = ["A_divided_custom"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame({"A_divided_custom": [1.0, 2.0, 3.0]})
        assert_frame_equal(result, expected_values)

    def test_divide_multiidx_with_positional_level(self):
        result = divide_columns(self.df_multi, ["one", "two"], "three", level=-1)
        expected_columns = [
            ("A", f"one_{DIVIDED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{DIVIDED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{DIVIDED_COLUMN_NAME.format('B')},three"): [1.0, 2.0, 3.0],
                ("A", f"two_{DIVIDED_COLUMN_NAME.format('B')},three"): [4.0, 5.0, 6.0],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_divide_with_invalid_column_to_divide(self):
        with self.assertRaises(ArgumentValueError):
            divide_columns(self.df_single, ["A", "B"], ["C", "B"])

    def test_divide_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            divide_columns(self.df_single, ["A"], "D")

    def test_divide_non_numeric_data(self):
        df_non_numeric = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        with self.assertRaises(ArgumentValueError):
            divide_columns(df_non_numeric, ["B"], "A")


class TestMultiplyColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [2, 2, 2]})
        arrays = [["A", "A", "B"], ["one", "two", "three"]]
        self.df_multi = DataFrame(
            {
                ("A", "one"): [1, 2, 3],
                ("A", "two"): [4, 5, 6],
                ("B", "three"): [2, 2, 2],
            }
        )
        self.df_multi.columns = MultiIndex.from_arrays(arrays)

    def test_multiply_singleidx(self):
        result = multiply_columns(self.df_single, ["A", "B"], "C")
        expected_columns = [
            f"A_{MULTIPLIED_COLUMN_NAME.format('C')}",
            f"B_{MULTIPLIED_COLUMN_NAME.format('C')}",
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                f"A_{MULTIPLIED_COLUMN_NAME.format('C')}": [2, 4, 6],
                f"B_{MULTIPLIED_COLUMN_NAME.format('C')}": [8, 10, 12],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_multiply_multiidx(self):
        result = multiply_columns(
            self.df_multi, [("A", "one"), ("A", "two")], ("B", "three")
        )
        expected_columns = [
            ("A", f"one_{MULTIPLIED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{MULTIPLIED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{MULTIPLIED_COLUMN_NAME.format('B')},three"): [2, 4, 6],
                ("A", f"two_{MULTIPLIED_COLUMN_NAME.format('B')},three"): [8, 10, 12],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_multiply_with_custom_rename(self):
        result = multiply_columns(
            self.df_single, ["A"], "C", rename="multiplied_custom"
        )
        expected_columns = ["A_multiplied_custom"]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame({"A_multiplied_custom": [2, 4, 6]})
        assert_frame_equal(result, expected_values)

    def test_multiply_multiidx_with_positional_level(self):
        result = multiply_columns(self.df_multi, ["one", "two"], "three", level=-1)
        expected_columns = [
            ("A", f"one_{MULTIPLIED_COLUMN_NAME.format('B')},three"),
            ("A", f"two_{MULTIPLIED_COLUMN_NAME.format('B')},three"),
        ]
        self.assertListEqual(list(result.columns), expected_columns)
        expected_values = DataFrame(
            {
                ("A", f"one_{MULTIPLIED_COLUMN_NAME.format('B')},three"): [2, 4, 6],
                ("A", f"two_{MULTIPLIED_COLUMN_NAME.format('B')},three"): [8, 10, 12],
            }
        )
        assert_frame_equal(result, expected_values)

    def test_multiply_with_invalid_column_to_multiply(self):
        with self.assertRaises(ArgumentValueError):
            multiply_columns(self.df_single, ["A", "B"], ["C", "B"])

    def test_multiply_with_invalid_columns(self):
        with self.assertRaises(ArgumentValueError):
            multiply_columns(self.df_single, ["A"], "D")


if __name__ == "__main__":
    unittest.main()
