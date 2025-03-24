import unittest
from unittest import TestCase

from mitoolspro.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitoolspro.plotting.plots.validations import (
    validate_bins,
    validate_bool,
    validate_bool_sequence,
    validate_color,
    validate_color_sequence,
    validate_color_sequences,
    validate_consistent_len,
    validate_edgecolor,
    validate_edgecolor_sequence,
    validate_edgecolor_sequences,
    validate_length,
    validate_literal,
    validate_literal_sequence,
    validate_marker,
    validate_non_negative,
    validate_numeric,
    validate_numeric_sequence,
    validate_numeric_sequences,
    validate_numeric_tuple,
    validate_numeric_tuple_sequence,
    validate_same,
    validate_same_length,
    validate_sequence_length,
    validate_sequence_non_negative,
    validate_sequence_type,
    validate_subsequences_length,
    validate_type,
    validate_value_in_options,
    validate_value_in_range,
)


class TestValidations(TestCase):
    def test_validate_type(self):
        validate_type(1, (int, float), "value")
        validate_type(1.0, (int, float), "value")
        with self.assertRaises(ArgumentTypeError):
            validate_type("1", (int, float), "value")

    def test_validate_sequence_type(self):
        validate_sequence_type([1, 2, 3], (int, float), "sequence")
        validate_sequence_type([1.0, 2.0, 3.0], (int, float), "sequence")
        with self.assertRaises(ArgumentTypeError):
            validate_sequence_type([1, "2", 3], (int, float), "sequence")

    def test_validate_non_negative(self):
        validate_non_negative(0, "value")
        validate_non_negative(1, "value")
        with self.assertRaises(ArgumentValueError):
            validate_non_negative(-1, "value")

    def test_validate_sequence_non_negative(self):
        validate_sequence_non_negative([0, 1, 2], "sequence")
        with self.assertRaises(ArgumentValueError):
            validate_sequence_non_negative([0, -1, 2], "sequence")

    def test_validate_sequence_length(self):
        validate_sequence_length([1, 2, 3], 3, "sequence")
        validate_sequence_length([1, 2], (2, 3), "sequence")
        with self.assertRaises(ArgumentStructureError):
            validate_sequence_length([1, 2], 3, "sequence")

    def test_validate_subsequences_length(self):
        validate_subsequences_length([[1, 2], [3, 4]], 2, "sequence")
        with self.assertRaises(ArgumentStructureError):
            validate_subsequences_length([[1, 2], [3]], 2, "sequence")

    def test_validate_same_length(self):
        validate_same_length([1, 2], [3, 4], "seq1", "seq2")
        with self.assertRaises(ArgumentStructureError):
            validate_same_length([1, 2], [3], "seq1", "seq2")

    def test_validate_length(self):
        validate_length([1, 2, 3], 3, "sequence")
        with self.assertRaises(ArgumentStructureError):
            validate_length([1, 2], 3, "sequence")

    def test_validate_value_in_options(self):
        validate_value_in_options("a", ["a", "b", "c"], "value")
        with self.assertRaises(ArgumentValueError):
            validate_value_in_options("d", ["a", "b", "c"], "value")

    def test_validate_bool(self):
        validate_bool(True, "value")
        validate_bool(False, "value")
        with self.assertRaises(ArgumentTypeError):
            validate_bool(1, "value")

    def test_validate_bool_sequence(self):
        validate_bool_sequence([True, False, True], "sequence")
        with self.assertRaises(ArgumentTypeError):
            validate_bool_sequence([True, 1, False], "sequence")

    def test_validate_color(self):
        validate_color("red")
        validate_color("#FF0000")
        validate_color((1, 0, 0))
        validate_color((1, 0, 0, 0.5))
        with self.assertRaises(ArgumentTypeError):
            validate_color({"red": 1})

    def test_validate_color_sequence(self):
        validate_color_sequence(["red", "blue", "green"])
        validate_color_sequence([(1, 0, 0), (0, 1, 0)])
        with self.assertRaises(ArgumentTypeError):
            validate_color_sequence([{"red": 1}, "blue"])

    def test_validate_color_sequences(self):
        validate_color_sequences([["red", "blue"], ["green", "yellow"]])
        with self.assertRaises(ArgumentTypeError):
            validate_color_sequences([["red", {"blue": 1}], ["green", "yellow"]])

    def test_validate_edgecolor(self):
        validate_edgecolor("red")
        validate_edgecolor("face")
        validate_edgecolor("none")
        validate_edgecolor(None)
        with self.assertRaises(ArgumentTypeError):
            validate_edgecolor({"red": 1})

    def test_validate_edgecolor_sequence(self):
        validate_edgecolor_sequence(["red", "blue", "face"])
        with self.assertRaises(ArgumentTypeError):
            validate_edgecolor_sequence([{"red": 1}, "blue"])

    def test_validate_edgecolor_sequences(self):
        validate_edgecolor_sequences([["red", "blue"], ["face", "none"]])
        with self.assertRaises(ArgumentTypeError):
            validate_edgecolor_sequences([["red", {"blue": 1}], ["face", "none"]])

    def test_validate_marker(self):
        validate_marker("o")
        validate_marker(1)
        validate_marker({"marker": "o", "fillstyle": "full"})
        with self.assertRaises(ArgumentTypeError):
            validate_marker({"invalid": "o"})

    def test_validate_numeric(self):
        validate_numeric(1, "value")
        validate_numeric(1.0, "value")
        with self.assertRaises(ArgumentTypeError):
            validate_numeric("1", "value")

    def test_validate_numeric_sequence(self):
        validate_numeric_sequence([1, 2, 3], "sequence")
        validate_numeric_sequence([1.0, 2.0, 3.0], "sequence")
        with self.assertRaises(ArgumentTypeError):
            validate_numeric_sequence([1, "2", 3], "sequence")

    def test_validate_numeric_sequences(self):
        validate_numeric_sequences([[1, 2], [3, 4]], "sequence")
        with self.assertRaises(ArgumentTypeError):
            validate_numeric_sequences([[1, "2"], [3, 4]], "sequence")

    def test_validate_numeric_tuple(self):
        validate_numeric_tuple((1, 2), [2])
        validate_numeric_tuple((1, 2, 3), [3])
        with self.assertRaises(ArgumentTypeError):
            validate_numeric_tuple((1, "2"), [2])

    def test_validate_numeric_tuple_sequence(self):
        validate_numeric_tuple_sequence([(1, 2), (3, 4)], [2])
        with self.assertRaises(ArgumentTypeError):
            validate_numeric_tuple_sequence([(1, "2"), (3, 4)], [2])

    def test_validate_bins(self):
        validate_bins(10)
        validate_bins("auto")
        with self.assertRaises(ArgumentTypeError):
            validate_bins("invalid")

    def test_validate_consistent_len(self):
        validate_consistent_len([[1, 2, 3], [4, 5, 6]], "sequence")
        validate_consistent_len([[1, 2, 3], [4]], "sequence")
        with self.assertRaises(ArgumentStructureError):
            validate_consistent_len([[1, 2], [3, 4, 5]], "sequence")

    def test_validate_value_in_range(self):
        validate_value_in_range(1, 0, 2, "value")
        validate_value_in_range(1.0, 0, 2, "value")
        with self.assertRaises(ArgumentValueError):
            validate_value_in_range(3, 0, 2, "value")

    def test_validate_same(self):
        validate_same(1, 1, "value1", "value2")
        with self.assertRaises(ArgumentValueError):
            validate_same(1, 2, "value1", "value2")

    def test_validate_literal(self):
        validate_literal("a", ["a", "b", "c"])
        with self.assertRaises(ArgumentTypeError):
            validate_literal("d", ["a", "b", "c"])

    def test_validate_literal_sequence(self):
        validate_literal_sequence(["a", "b", "c"], ["a", "b", "c"])
        with self.assertRaises(ArgumentTypeError):
            validate_literal_sequence(["a", "d", "c"], ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
