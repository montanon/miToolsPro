import sys
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import mock_open, patch

import numpy as np
from pandas import Series

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.utils.functions import (
    add_significance,
    all_can_be_ints,
    can_convert_to,
    check_symmetrical_matrix,
    dict_from_kwargs,
    display_env_variables,
    get_file_encoding,
    invert_dict,
    iterable_chunks,
    sort_dict_keys,
    unpack_list_of_lists,
)


class TestIterableChunks(TestCase):
    def test_list_input(self):
        iterable = [1, 2, 3, 4, 5, 6]
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [[1, 2], [3, 4], [5, 6]])

    def test_string_input(self):
        iterable = "123456"
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, ["12", "34", "56"])

    def test_tuple_input(self):
        iterable = (1, 2, 3, 4, 5, 6)
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [(1, 2), (3, 4), (5, 6)])

    def test_bytes_input(self):
        iterable = b"123456"
        chunk_size = 2
        result = list(iterable_chunks(iterable, chunk_size))
        self.assertEqual(result, [b"12", b"34", b"56"])

    def test_invalid_input(self):
        iterable = set([1, 2, 3, 4, 5, 6])
        chunk_size = 2
        with self.assertRaises(TypeError):
            list(iterable_chunks(iterable, chunk_size))


class TestDictFromKwargs(TestCase):
    def test_no_arguments(self):
        self.assertEqual(dict_from_kwargs(), {})

    def test_one_argument(self):
        self.assertEqual(dict_from_kwargs(a=1), {"a": 1})

    def test_multiple_arguments(self):
        self.assertEqual(dict_from_kwargs(a=1, b=2, c=3), {"a": 1, "b": 2, "c": 3})


class TestAddSignificance(TestCase):
    def test_add_significance_very_significant(self):
        row = Series(["Test (0.001)"])
        self.assertEqual(row.apply(add_significance)[0], "Test (0.001)***")

    def test_add_significance_significant(self):
        row = Series(["Test (0.03)"])
        self.assertEqual(row.apply(add_significance)[0], "Test (0.03)**")

    def test_add_significance_moderately_significant(self):
        row = Series(["Test (0.07)"])
        self.assertEqual(row.apply(add_significance)[0], "Test (0.07)*")

    def test_add_significance_not_significant(self):
        row = Series(["Test (0.2)"])
        self.assertEqual(row.apply(add_significance)[0], "Test (0.2)")


class TestCanConvertTo(TestCase):
    def test_can_convert_to_int_from_int(self):
        items = [1, 2, 3]
        self.assertTrue(can_convert_to(items, int))

    def test_can_convert_to_str_from_str(self):
        items = ["1", "2", "3"]
        self.assertTrue(can_convert_to(items, str))

    def test_can_convert_to_float_from_float(self):
        items = [1.0, 2.0, 3.0]
        self.assertTrue(can_convert_to(items, float))

    def test_can_convert_to_bool_from_bool(self):
        items = [True, False, True]
        self.assertTrue(can_convert_to(items, bool))

    def test_can_convert_to_int_from_str(self):
        items = ["1", "2", "3"]
        self.assertTrue(can_convert_to(items, int))

    def test_can_convert_to_str_from_int(self):
        items = [1, 2, 3]
        self.assertTrue(can_convert_to(items, str))

    def test_can_convert_to_float_from_str(self):
        items = ["1.0", "2.0", "3.0"]
        self.assertTrue(can_convert_to(items, float))

    def test_can_convert_to_bool_from_str(self):
        items = ["True", "False", "True"]
        self.assertTrue(can_convert_to(items, bool))

    def test_can_convert_to_int_from_int_fail(self):
        items = [1, 2, 3, "fail"]
        self.assertFalse(can_convert_to(items, int))

    def test_can_convert_to_float_from_float_fail(self):
        items = [1.0, 2.0, 3.0, "fail"]
        self.assertFalse(can_convert_to(items, float))


class TestInvertDict(TestCase):
    def test_invert_dict(self):
        dictionary = {"a": 1, "b": 2, "c": 3}
        inverted = {1: "a", 2: "b", 3: "c"}
        self.assertEqual(invert_dict(dictionary), inverted)

    def test_invert_dict_empty(self):
        dictionary = {}
        inverted = {}
        self.assertEqual(invert_dict(dictionary), inverted)

    def test_invert_dict_duplicates(self):
        dictionary = {"a": 1, "b": 1, "c": 2}
        inverted = {1: "b", 2: "c"}
        self.assertEqual(invert_dict(dictionary), inverted)


class TestCheckSymmetricalMatrix(TestCase):
    def test_check_symmetrical_matrix_symmetrical(self):
        a = np.array([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
        self.assertTrue(check_symmetrical_matrix(a))

    def test_check_symmetrical_matrix_not_symmetrical(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertFalse(check_symmetrical_matrix(a))

    def test_check_symmetrical_matrix_symmetrical_with_tolerance(self):
        a = np.array([[1, 2, 3], [2, 1, 4.0001], [3, 4, 1]])
        self.assertTrue(check_symmetrical_matrix(a, rtol=1e-04))

    def test_check_symmetrical_matrix_not_symmetrical_with_tolerance(self):
        a = np.array([[1, 2, 3], [2, 1, 4.1], [3, 4, 1]])
        self.assertFalse(check_symmetrical_matrix(a, rtol=1e-04))


class TestUnpackListOfLists(TestCase):
    def test_unpack_list_of_lists(self):
        list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        unpacked = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(unpack_list_of_lists(list_of_lists), unpacked)

    def test_unpack_list_of_lists_empty(self):
        list_of_lists = []
        unpacked = []
        self.assertEqual(unpack_list_of_lists(list_of_lists), unpacked)

    def test_unpack_list_of_lists_single(self):
        list_of_lists = [[1, 2, 3]]
        unpacked = [1, 2, 3]
        self.assertEqual(unpack_list_of_lists(list_of_lists), unpacked)


class TestDisplayEnvVariables(TestCase):
    def setUp(self):
        self.env_vars = [
            ("small_int", 1),
            ("large_list", list(range(10000))),
            ("string", "hello world"),
            ("large_dict", {i: i for i in range(1000)}),
        ]

    def test_no_large_variables(self):
        threshold_mb = sys.getsizeof(self.env_vars[1][1]) / (1024**2) + 1
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertTrue(df.empty)

    def test_large_variables(self):
        threshold_mb = 0
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertFalse(df.empty)
        self.assertTrue(all(df["Size (MB)"] > threshold_mb))

    def test_edge_cases(self):
        # Empty env_vars
        df_empty = display_env_variables([], 0)
        self.assertTrue(df_empty.empty)
        # Extremely high threshold
        df_high_threshold = display_env_variables(self.env_vars, 1000000)
        self.assertTrue(df_high_threshold.empty)

    def test_different_data_types(self):
        threshold_mb = 0
        df = display_env_variables(self.env_vars, threshold_mb)
        self.assertIn("large_list", df["Variable"].values)
        self.assertIn("large_dict", df["Variable"].values)


class TestSortDictKeys(TestCase):
    def test_sort_by_keys_ascending(self):
        input_dict = {"b": 2, "a": 3, "d": 1, "c": 4}
        expected_output = {"a": 3, "b": 2, "c": 4, "d": 1}
        self.assertEqual(sort_dict_keys(input_dict), expected_output)

    def test_sort_by_keys_descending(self):
        input_dict = {"b": 2, "a": 3, "d": 1, "c": 4}
        expected_output = {"d": 1, "c": 4, "b": 2, "a": 3}
        self.assertEqual(sort_dict_keys(input_dict, reverse=True), expected_output)

    def test_sort_by_values_ascending(self):
        input_dict = {"b": 2, "a": 3, "d": 1, "c": 4}
        expected_output = {"d": 1, "b": 2, "a": 3, "c": 4}
        self.assertEqual(
            sort_dict_keys(input_dict, key=lambda item: item[1]), expected_output
        )

    def test_sort_by_values_descending(self):
        input_dict = {"b": 2, "a": 3, "d": 1, "c": 4}
        expected_output = {"c": 4, "a": 3, "b": 2, "d": 1}
        self.assertEqual(
            sort_dict_keys(input_dict, key=lambda item: item[1], reverse=True),
            expected_output,
        )

    def test_empty_dict(self):
        input_dict = {}
        expected_output = {}
        self.assertEqual(sort_dict_keys(input_dict), expected_output)

    def test_single_element_dict(self):
        input_dict = {"a": 1}
        expected_output = {"a": 1}
        self.assertEqual(sort_dict_keys(input_dict), expected_output)

    def test_invalid_input_type(self):
        with self.assertRaises(ArgumentValueError):
            sort_dict_keys(None)

    def test_sort_with_custom_key_function(self):
        input_dict = {"b": "banana", "a": "apple", "c": "cherry"}
        # Custom key function: sort by the length of the values
        expected_output = {"a": "apple", "c": "cherry", "b": "banana"}
        self.assertEqual(
            sort_dict_keys(input_dict, key=lambda item: len(item[1])), expected_output
        )


class TestGetFileEncoding(TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.write("This is a test.".encode("utf-8"))
        self.temp_file.close()

    def tearDown(self):
        Path(self.temp_file.name).unlink()

    def test_detect_utf8_encoding(self):
        encoding = get_file_encoding(self.temp_file.name)
        self.assertEqual(encoding, "utf-8")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_file_encoding("non_existent_file.txt")

    @patch("builtins.open", new_callable=mock_open, read_data=b"\x80\x81\x82")
    def test_detect_low_confidence_encoding(self, mock_file):
        with patch(
            "chardet.detect", return_value={"encoding": "iso-8859-1", "confidence": 0.5}
        ):
            encoding = get_file_encoding("dummy_file.txt")
            self.assertEqual(encoding, "utf-8")

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_io_error(self, mock_file):
        with self.assertRaises(IOError):
            get_file_encoding("dummy_file.txt")


class TestAllCanBeInts(TestCase):
    def test_all_ints(self):
        self.assertTrue(all_can_be_ints([1, 2, 3]))

    def test_all_strings_representing_ints(self):
        self.assertTrue(all_can_be_ints(["1", "2", "3"]))

    def test_mixed_types(self):
        self.assertTrue(all_can_be_ints(["1", 2, 3.0]))

    def test_non_convertible_item(self):
        self.assertFalse(all_can_be_ints(["1", "a", "3"]))

    def test_with_empty_list(self):
        self.assertTrue(all_can_be_ints([]))

    def test_with_none(self):
        self.assertFalse(all_can_be_ints([None]))

    def test_with_non_numeric_types(self):
        self.assertFalse(all_can_be_ints([1, 2, "three"]))

    def test_with_nested_list(self):
        self.assertFalse(all_can_be_ints([[1, 2], 3]))

    def test_with_boolean_values(self):
        self.assertTrue(all_can_be_ints([True, False]))


if __name__ == "__main__":
    unittest.main()
