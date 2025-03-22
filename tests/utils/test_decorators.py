import unittest
import warnings
from typing import Dict, List, Optional
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex, Series

from mitoolspro.exceptions import (
    ArgumentTypeError,
    ArgumentValueError,
    ColumnValidationError,
)
from mitoolspro.utils.decorators import (
    cached_property,
    parallel,
    store_signature_in_dev,
    suppress_user_warning,
    validate_args_types,
    validate_dataframe_structure,
)
from mitoolspro.utils.dev_object import Dev, get_dev_var
from mitoolspro.utils.functions import remove_characters_from_strings
from mitoolspro.utils.validation_templates import sankey_plot_validation


class TestStoreSignatureInDev(TestCase):
    def setUp(self):
        Dev().clear_vars()

    def test_basic_function_args(self):
        @store_signature_in_dev
        def example_func(x: int, y: str):
            return x + len(y)

        result = example_func(42, "test")
        stored_args = get_dev_var("example_func")

        self.assertEqual(stored_args["x"], 42)
        self.assertEqual(stored_args["y"], "test")
        self.assertEqual(result, 46)  # Verify function still works normally

    def test_function_with_defaults(self):
        @store_signature_in_dev
        def func_with_defaults(x: int = 10, y: str = "default"):
            return x + len(y)

        func_with_defaults()
        stored_args = get_dev_var("func_with_defaults")
        self.assertEqual(stored_args["x"], 10)
        self.assertEqual(stored_args["y"], "default")
        func_with_defaults(x=20)
        stored_args = get_dev_var("func_with_defaults")
        self.assertEqual(stored_args["x"], 20)
        self.assertEqual(stored_args["y"], "default")

    def test_function_with_kwargs(self):
        @store_signature_in_dev
        def func_with_kwargs(x: int, **kwargs):
            return x + len(kwargs)

        func_with_kwargs(10, extra="test", another="value")
        stored_args = get_dev_var("func_with_kwargs")
        self.assertEqual(stored_args["x"], 10)
        self.assertEqual(stored_args["kwargs"], {"extra": "test", "another": "value"})

    def test_function_with_args(self):
        @store_signature_in_dev
        def func_with_args(*args):
            return sum(args)

        func_with_args(1, 2, 3)
        stored_args = get_dev_var("func_with_args")
        self.assertEqual(stored_args["args"], (1, 2, 3))

    def test_method_in_class(self):
        class TestClass:
            @store_signature_in_dev
            def test_method(self, x: int, y: Optional[str] = None):
                return x + (len(y) if y else 0)

        obj = TestClass()
        obj.test_method(42, "test")
        stored_args = get_dev_var("test_method")
        self.assertIn("self", stored_args)
        self.assertEqual(stored_args["x"], 42)
        self.assertEqual(stored_args["y"], "test")

    def test_complex_types(self):
        @store_signature_in_dev
        def complex_func(numbers: List[int], text: str = "default"):
            return sum(numbers) + len(text)

        complex_func([1, 2, 3], "test")
        stored_args = get_dev_var("complex_func")
        self.assertEqual(stored_args["numbers"], [1, 2, 3])
        self.assertEqual(stored_args["text"], "test")

    def test_multiple_calls(self):
        @store_signature_in_dev
        def multi_call(x: int):
            return x * 2

        multi_call(10)
        stored_args = get_dev_var("multi_call")
        self.assertEqual(stored_args["x"], 10)
        multi_call(20)
        stored_args = get_dev_var("multi_call")
        self.assertEqual(stored_args["x"], 20)

    def test_nested_functions(self):
        @store_signature_in_dev
        def outer(x: int):
            @store_signature_in_dev
            def inner(y: int):
                return y * 2

            return inner(x + 1)

        result = outer(5)
        outer_args = get_dev_var("outer")
        inner_args = get_dev_var("inner")
        self.assertEqual(outer_args["x"], 5)
        self.assertEqual(inner_args["y"], 6)
        self.assertEqual(result, 12)

    def test_preserves_docstring(self):
        @store_signature_in_dev
        def documented_func(x: int):
            """This is a test docstring."""
            return x

        self.assertEqual(documented_func.__doc__, "This is a test docstring.")

    def test_preserves_function_name(self):
        @store_signature_in_dev
        def named_func(x: int):
            return x

        self.assertEqual(named_func.__name__, "named_func")


class TestValidateTypesDecorator(TestCase):
    def test_correct_types_positional_arguments(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        self.assertTrue(test_func(10, "hello"))

    def test_correct_types_keyword_arguments(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        self.assertTrue(test_func(x=10, y="hello"))

    def test_incorrect_type_positional_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(10, 20)  # y should be a str, not an int
        self.assertIn("Argument 'y' must be of type str", str(context.exception))

    def test_incorrect_type_keyword_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(x=10, y=20)  # y should be a str, not an int
        self.assertIn("Argument 'y' must be of type str", str(context.exception))

    def test_missing_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x):
            return True

        with self.assertRaises(ArgumentValueError):
            test_func(10)  # Missing argument 'y'

    def test_extra_argument(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y, z):
            return True

        self.assertTrue(test_func(10, "hello", "extra argument"))

    def test_multiple_arguments_different_types(self):
        @validate_args_types(a=int, b=float, c=str)
        def test_func(a, b, c):
            return True

        self.assertTrue(test_func(5, 3.14, "test"))

    def test_multiple_incorrect_arguments(self):
        @validate_args_types(a=int, b=float, c=str)
        def test_func(a, b, c):
            return True

        with self.assertRaises(ArgumentTypeError) as context:
            test_func(5, "not a float", 10)  # b is incorrect
        self.assertIn("Argument 'b' must be of type float", str(context.exception))

    def test_unexpected_argument_name(self):
        @validate_args_types(a=int, b=str)
        def test_func(x, y):
            return True

        with self.assertRaises(ArgumentValueError) as context:
            test_func(5, "hello")
        self.assertIn(
            "Argument 'a' not found in function signature", str(context.exception)
        )

    def test_with_default_values(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y="default"):
            return True

        self.assertTrue(test_func(5))  # y should use the default value, no TypeError

    def test_type_check_on_default_value(self):
        @validate_args_types(x=int, y=str)
        def test_func(x, y="default"):
            return True

        with self.assertRaises(TypeError):
            test_func(
                5, y=10
            )  # y should be a str, not an int, even with default values present

    def test_no_type_validation_when_not_specified(self):
        @validate_args_types(x=int)
        def test_func(x, y):
            return True

        self.assertTrue(
            test_func(5, "anything")
        )  # y has no specified type, so any type is allowed


def custom_validation(
    dataframe: DataFrame,
    required_columns: List[str] = None,
    column_types: Dict[str, str] = None,
) -> bool:
    if required_columns:
        missing_columns = [
            col for col in required_columns if col not in dataframe.columns
        ]
        if missing_columns:
            raise ArgumentValueError(
                f"DataFrame is missing required columns: {missing_columns}"
            )
    if column_types:
        for col, expected_type in column_types.items():
            if col in dataframe.columns and not pd.api.types.is_dtype_equal(
                dataframe[col].dtype, expected_type
            ):
                raise ArgumentTypeError(
                    f"Column '{col}' must be of type {expected_type}. Found {dataframe[col].dtype} instead."
                )


@validate_dataframe_structure(
    dataframe_name="data",
    validation=custom_validation,
    required_columns=["column1", "column2"],
    column_types={"column1": "int64", "column2": "float64"},
)
def process_data(data: DataFrame) -> str:
    return "Data processed successfully"


class TestValidateDataFrameStructureDecorator(TestCase):
    def setUp(self):
        self.correct_df = DataFrame(
            {
                "column1": Series([1, 2, 3], dtype="int64"),
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
            }
        )
        self.missing_column_df = DataFrame(
            {"column1": Series([1, 2, 3], dtype="int64")}
        )
        self.wrong_type_df = DataFrame(
            {
                "column1": Series([1.0, 2.0, 3.0], dtype="float64"),  # Should be int64
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
            }
        )
        self.extra_columns_df = DataFrame(
            {
                "column1": Series([1, 2, 3], dtype="int64"),
                "column2": Series([1.0, 2.0, 3.0], dtype="float64"),
                "extra_column": Series(["a", "b", "c"], dtype="object"),
            }
        )
        self.non_dataframe_input = "Not a DataFrame"

    def test_correct_dataframe_structure(self):
        result = process_data(data=self.correct_df)
        self.assertEqual(result, "Data processed successfully")

    def test_missing_required_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            process_data(data=self.missing_column_df)
        self.assertIn(
            "DataFrame is missing required columns: ['column2']", str(context.exception)
        )

    def test_incorrect_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            process_data(data=self.wrong_type_df)
        self.assertIn(
            "Column 'column1' must be of type int64. Found float64 instead.",
            str(context.exception),
        )

    def test_extra_columns(self):
        result = process_data(data=self.extra_columns_df)
        self.assertEqual(result, "Data processed successfully")

    def test_non_dataframe_input(self):
        with self.assertRaises(ArgumentTypeError) as context:
            process_data(data=self.non_dataframe_input)
        self.assertIn("must be a DataFrame.", str(context.exception))

    def test_empty_dataframe(self):
        empty_correct_df = DataFrame(
            {
                "column1": Series([], dtype="int64"),
                "column2": Series([], dtype="float64"),
            }
        )
        result = process_data(data=empty_correct_df)
        self.assertEqual(result, "Data processed successfully")

    def test_partial_column_types(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
            required_columns=["column1"],
            column_types={"column1": "int64"},
        )
        def partial_type_check(data: DataFrame) -> str:
            return "Data processed with partial column type check"

        result = partial_type_check(data=self.correct_df)
        self.assertEqual(result, "Data processed with partial column type check")

    def test_no_validation_criteria(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
        )
        def no_criteria_check(data: DataFrame) -> str:
            return "Data processed with no criteria"

        result = no_criteria_check(data=self.correct_df)
        self.assertEqual(result, "Data processed with no criteria")

    def test_missing_required_column_with_partial_check(self):
        @validate_dataframe_structure(
            dataframe_name="data",
            validation=custom_validation,
            required_columns=["column1", "column2"],
        )
        def partial_column_check(data: DataFrame) -> str:
            return "Data processed with partial column check"

        with self.assertRaises(ArgumentValueError) as context:
            partial_column_check(data=self.missing_column_df)
        self.assertIn(
            "DataFrame is missing required columns: ['column2']", str(context.exception)
        )


class TestValidateDataFrameColumns(TestCase):
    def setUp(self):
        correct_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Gram"),
                ("(2000, 2020)", "2_3-Gram", "Count"),
                ("(2010, 2020)", "1_2-Gram", "Gram"),
                ("(2010, 2020)", "1_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.correct_df = pd.DataFrame(
            [
                ["example1", 10, "example2", 15],
                [None, 20, "example3", np.nan],
            ],
            columns=correct_index,
        )
        invalid_level_0_index = MultiIndex.from_tuples(
            [
                ("2000-2020", "2_3-Gram", "Gram"),
                ("2000-2020", "2_3-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_0_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_0_index
        )
        invalid_level_1_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "3_2-Gram", "Gram"),
                ("(2000, 2020)", "3_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_1_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_1_index
        )
        invalid_level_2_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Frequency"),
                ("(2000, 2020)", "2_3-Gram", "Sum"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        self.invalid_level_2_df = pd.DataFrame(
            [["example", 5], [None, 10]], columns=invalid_level_2_index
        )
        self.invalid_gram_column_df = self.correct_df.copy()
        self.invalid_gram_column_df[("(2000, 2020)", "2_3-Gram", "Gram")] = [123, None]
        self.invalid_count_column_df = self.correct_df.copy()
        self.invalid_count_column_df[("(2000, 2020)", "2_3-Gram", "Count")] = [
            "not a number",
            np.nan,
        ]

    def test_correct_dataframe(self):
        self.assertTrue(sankey_plot_validation(self.correct_df))

    def test_invalid_level_0_format(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_0_df)
        self.assertIn("Level 0 column", str(context.exception))

    def test_invalid_level_1_format(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_1_df)
        self.assertIn("Level 1 column", str(context.exception))

    def test_invalid_level_2_names(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_level_2_df)
        self.assertIn("Level 2 column", str(context.exception))

    def test_invalid_gram_column_values(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_gram_column_df)
        self.assertIn(
            "Level 2 'Gram' columns must contain strings or NaN", str(context.exception)
        )

    def test_invalid_count_column_values(self):
        with self.assertRaises(ColumnValidationError) as context:
            sankey_plot_validation(self.invalid_count_column_df)
        self.assertIn(
            "Level 2 'Count' columns must contain numeric values or NaN",
            str(context.exception),
        )

    def test_empty_dataframe(self):
        empty_correct_index = MultiIndex.from_tuples(
            [
                ("(2000, 2020)", "2_3-Gram", "Gram"),
                ("(2000, 2020)", "2_3-Gram", "Count"),
                ("(2010, 2020)", "1_2-Gram", "Gram"),
                ("(2010, 2020)", "1_2-Gram", "Count"),
            ],
            names=["year_range", "n-gram", "values"],
        )
        empty_correct_df = pd.DataFrame(columns=empty_correct_index)
        self.assertTrue(sankey_plot_validation(empty_correct_df))


class TestCachedProperty(TestCase):
    def setUp(self):
        class Sample:
            def __init__(self, value):
                self._value = value
                self.compute_count = 0

            @cached_property
            def computed_property(self):
                "Sample computed property."
                self.compute_count += 1
                return self._value * 2

        self.Sample = Sample

    def test_cached_property_basic(self):
        obj = self.Sample(10)
        self.assertEqual(obj.computed_property, 20)
        self.assertEqual(obj.computed_property, 20)
        self.assertEqual(obj.compute_count, 1)

    def test_cached_property_caching(self):
        obj = self.Sample(5)
        first_access = obj.computed_property
        second_access = obj.computed_property
        self.assertEqual(first_access, 10)
        self.assertEqual(second_access, 10)
        self.assertEqual(obj.compute_count, 1)

    def test_cached_property_mutability(self):
        obj = self.Sample(3)
        self.assertEqual(obj.computed_property, 6)
        obj.__dict__["computed_property"] = 15
        self.assertEqual(obj.computed_property, 15)
        self.assertEqual(obj.compute_count, 1)

    def test_cached_property_exceptions(self):
        class ExceptionSample:
            def __init__(self, raise_exception):
                self.raise_exception = raise_exception

            @cached_property
            def error_property(self):
                if self.raise_exception:
                    raise ValueError("Intentional error")
                return 42

        obj = ExceptionSample(raise_exception=True)
        with self.assertRaises(ValueError):
            _ = obj.error_property
        obj.raise_exception = False
        self.assertEqual(obj.error_property, 42)

    def test_cached_property_docstring(self):
        obj = self.Sample(4)
        self.assertEqual(
            obj.__class__.computed_property.__doc__, "Sample computed property."
        )

    def test_cached_property_separate_instances(self):
        obj1 = self.Sample(7)
        obj2 = self.Sample(8)
        self.assertEqual(obj1.computed_property, 14)
        self.assertEqual(obj2.computed_property, 16)
        self.assertEqual(obj1.compute_count, 1)
        self.assertEqual(obj2.compute_count, 1)


def clean_string_chunk(chunk, characters=None):
    return list(remove_characters_from_strings(chunk, characters))


def invalid_chunk(chunk, characters=None):
    return 123  # Invalid return type (not iterable)


def fail_on_keyword(chunk, characters=None):
    for s in chunk:
        if "log" in s:
            raise ValueError("triggered error")
    return chunk


class TestParallelDecorator(unittest.TestCase):
    def setUp(self):
        self.filenames = [
            "report_01?.pdf",
            "data*log.txt",
            "summary<final>.doc",
            "image|backup.jpg",
            "file:name.csv",
            "presentation%.ppt",
        ]
        self.default_expected = [
            "report_01.pdf",
            "datalog.txt",
            "summaryfinal.doc",
            "imagebackup.jpg",
            "filename.csv",
            "presentation.ppt",
        ]
        self.custom_characters = r"[aeiou]"
        self.custom_expected = [
            "rprt_01?.pdf",
            "dt*lg.txt",
            "smmry<fnl>.dc",
            "mg|bckp.jpg",
            "fl:nm.csv",
            "prsnttn%.ppt",
        ]

    def test_parallel_default_characters(self):
        decorated = parallel(n_threads=2, chunk_size=2)(clean_string_chunk)
        result = decorated(self.filenames)
        self.assertEqual(result, self.default_expected)

    def test_parallel_custom_characters(self):
        decorated = parallel(n_threads=3, chunk_size=3)(clean_string_chunk)
        result = decorated(self.filenames, self.custom_characters)
        self.assertEqual(result, self.custom_expected)

    def test_parallel_empty_input(self):
        decorated = parallel(n_threads=2, chunk_size=1)(clean_string_chunk)
        result = decorated([])
        self.assertEqual(result, [])

    def test_parallel_chunk_size_larger_than_data(self):
        decorated = parallel(n_threads=4, chunk_size=100)(clean_string_chunk)
        result = decorated(self.filenames)
        self.assertEqual(result, self.default_expected)

    def test_parallel_one_thread(self):
        decorated = parallel(n_threads=1, chunk_size=2)(clean_string_chunk)
        result = decorated(self.filenames)
        self.assertEqual(result, self.default_expected)

    def test_parallel_thread_pool(self):
        decorated = parallel(n_threads=2, chunk_size=2, use_threads=True)(
            clean_string_chunk
        )
        result = decorated(self.filenames)
        self.assertEqual(result, self.default_expected)

    def test_parallel_invalid_return_type(self):
        decorated = parallel(n_threads=2, chunk_size=3)(invalid_chunk)
        with self.assertRaises(RuntimeError) as context:
            decorated(self.filenames)
        self.assertIn("Expected iterable return", str(context.exception))

    def test_parallel_function_raises(self):
        decorated = parallel(n_threads=2, chunk_size=2)(fail_on_keyword)
        with self.assertRaises(RuntimeError) as context:
            decorated(self.filenames)
        self.assertIn("triggered error", str(context.exception))

    def test_parallel_preserves_flattening(self):
        decorated = parallel(n_threads=2, chunk_size=2)(clean_string_chunk)
        result = decorated(self.filenames)
        self.assertEqual(len(result), len(self.filenames))
        self.assertTrue(all(isinstance(s, str) for s in result))


class TestSuppressUserWarningDecorator(TestCase):
    def test_suppresses_user_warning(self):
        @suppress_user_warning
        def warning_function():
            warnings.warn("This is a test warning", UserWarning)
            return True

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = warning_function()
            self.assertTrue(result)

    def test_does_not_suppress_other_warnings(self):
        @suppress_user_warning
        def warning_function():
            warnings.warn("This is a test warning", RuntimeWarning)
            return True

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertRaises(RuntimeWarning):
                warning_function()

    def test_multiple_warnings(self):
        @suppress_user_warning
        def warning_function():
            warnings.warn("First warning", UserWarning)
            warnings.warn("Second warning", UserWarning)
            return True

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = warning_function()
            self.assertTrue(result)

    def test_with_arguments(self):
        @suppress_user_warning
        def warning_function(x):
            warnings.warn("Test warning", UserWarning)
            return x * 2

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = warning_function(5)
            self.assertEqual(result, 10)

    def test_with_multiple_arguments(self):
        @suppress_user_warning
        def warning_function(x, y):
            warnings.warn("Test warning", UserWarning)
            return x + y

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = warning_function(3, 4)
            self.assertEqual(result, 7)

    def test_with_keyword_arguments(self):
        @suppress_user_warning
        def warning_function(x=1, y=2):
            warnings.warn("Test warning", UserWarning)
            return x + y

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = warning_function(x=5, y=3)
            self.assertEqual(result, 8)

    def test_preserves_function_name(self):
        @suppress_user_warning
        def test_func():
            return True

        self.assertEqual(test_func.__name__, "test_func")

    def test_preserves_docstring(self):
        @suppress_user_warning
        def test_func():
            """Test docstring."""
            return True

        self.assertEqual(test_func.__doc__, "Test docstring.")


if __name__ == "__main__":
    unittest.main()
