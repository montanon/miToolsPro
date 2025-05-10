import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.validation.functions import (
    coerce_to_list,
    is_bins,
    is_color,
    is_color_none,
    is_color_numeric_scalar,
    is_indexable,
    is_literal,
    is_marker,
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_valid_model,
    is_value_in_range,
    normalize_rgb_tuple,
    standardize_sequences,
    validate_numeric,
    validate_range,
    validate_sequence,
    validate_sequence_range,
    validate_sequence_sizes,
    validate_sequences_range,
    validate_sequences_sizes,
    validate_single_color,
    validate_tuple_sequence,
    validate_tuple_sequence_sizes,
    validate_tuple_sequences,
    validate_tuple_sequences_sizes,
    validate_tuple_sizes,
)


class TestIsValidModel(TestCase):
    def test_valid_model(self):
        class TestModel(BaseModel):
            name: str
            age: int

        self.assertTrue(is_valid_model(TestModel, name="test", age=25))

    def test_invalid_model(self):
        class TestModel(BaseModel):
            name: str
            age: int

        self.assertFalse(is_valid_model(TestModel, name="test", age="invalid"))


class TestIsIndexable(TestCase):
    def test_list_indexable(self):
        self.assertTrue(is_indexable([1, 2, 3], 0))

    def test_dict_indexable(self):
        self.assertTrue(is_indexable({"a": 1}, "a"))

    def test_string_indexable(self):
        self.assertTrue(is_indexable("test", 0))

    def test_not_indexable(self):
        self.assertFalse(is_indexable(123, 0))


class TestIsNumeric(TestCase):
    def test_numeric_values(self):
        self.assertTrue(is_numeric(1))
        self.assertTrue(is_numeric(1.0))
        self.assertTrue(is_numeric(0.00005))
        self.assertTrue(is_numeric(100_000_000))
        self.assertTrue(is_numeric(1e-10))

    def test_non_numeric_values(self):
        self.assertFalse(is_numeric("1"))
        self.assertFalse(is_numeric([1]))
        self.assertFalse(is_numeric(None))


class TestIsNumericSequence(TestCase):
    def test_valid_sequences(self):
        self.assertTrue(is_numeric_sequence([1, 2, 3]))
        self.assertTrue(is_numeric_sequence((1.0, 2.0, 3.0)))
        self.assertTrue(is_numeric_sequence(np.array([1, 2, 3])))

    def test_invalid_sequences(self):
        self.assertFalse(is_numeric_sequence([1, "2", 3]))
        self.assertFalse(is_numeric_sequence("123"))
        self.assertFalse(is_numeric_sequence(None))


class TestIsNumericSequences(TestCase):
    def test_valid_sequences(self):
        self.assertTrue(is_numeric_sequences([[1, 2], [3, 4]]))
        self.assertTrue(is_numeric_sequences(([1.0, 2.0], [3.0, 4.0])))

    def test_invalid_sequences(self):
        self.assertFalse(is_numeric_sequences([[1, "2"], [3, 4]]))
        self.assertFalse(is_numeric_sequences("123"))
        self.assertFalse(is_numeric_sequences(None))


class TestIsValueInRange(TestCase):
    def test_in_range(self):
        self.assertTrue(is_value_in_range(5, 0, 10))
        self.assertTrue(is_value_in_range(0, 0, 10))
        self.assertTrue(is_value_in_range(10, 0, 10))

    def test_out_of_range(self):
        self.assertFalse(is_value_in_range(-1, 0, 10))
        self.assertFalse(is_value_in_range(11, 0, 10))


class TestIsLiteral(TestCase):
    def test_valid_literals(self):
        self.assertTrue(is_literal("test", ["test", "other"]))
        self.assertTrue(is_literal(None, ["test", "other"]))

    def test_invalid_literals(self):
        self.assertFalse(is_literal("invalid", ["test", "other"]))
        self.assertFalse(is_literal(123, ["test", "other"]))


class TestIsBins(TestCase):
    def test_valid_bins(self):
        self.assertTrue(is_bins(10))
        self.assertTrue(is_bins("auto"))
        self.assertTrue(is_bins("fd"))

    def test_invalid_bins(self):
        self.assertFalse(is_bins(-1))
        self.assertFalse(is_bins("invalid"))
        self.assertFalse(is_bins(1_000_001))


class TestCoerceToList(TestCase):
    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        self.assertEqual(coerce_to_list(arr), [1, 2, 3])

    def test_pandas_series(self):
        series = pd.Series([1, 2, 3])
        self.assertEqual(coerce_to_list(series), [1, 2, 3])

    def test_tuple(self):
        self.assertEqual(coerce_to_list((1, 2, 3)), [1, 2, 3])

    def test_other_types(self):
        self.assertEqual(coerce_to_list([1, 2, 3]), [1, 2, 3])
        self.assertEqual(coerce_to_list("test"), "test")


class TestNormalizeRgbTuple(TestCase):
    def test_valid_rgb_float(self):
        self.assertEqual(normalize_rgb_tuple((0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))

    def test_valid_rgb_int(self):
        self.assertEqual(normalize_rgb_tuple((255, 128, 0)), (1.0, 0.502, 0.0))

    def test_valid_rgba_float(self):
        self.assertEqual(
            normalize_rgb_tuple((0.1, 0.2, 0.3, 0.4)), (0.1, 0.2, 0.3, 0.4)
        )

    def test_valid_rgba_int(self):
        self.assertEqual(
            normalize_rgb_tuple((255, 128, 0, 0.5)), (1.0, 0.502, 0.0, 0.5)
        )

    def test_invalid_inputs(self):
        self.assertEqual(normalize_rgb_tuple("red"), "red")
        self.assertEqual(normalize_rgb_tuple((1, 2)), (1, 2))
        self.assertEqual(normalize_rgb_tuple((1, 2, 3, 4, 5)), (1, 2, 3, 4, 5))


class TestIsColorNone(TestCase):
    def test_none_values(self):
        self.assertTrue(is_color_none(None))
        self.assertTrue(is_color_none("none"))

    def test_non_none_values(self):
        self.assertFalse(is_color_none("red"))
        self.assertFalse(is_color_none(1))


class TestIsColorNumericScalar(TestCase):
    def test_valid_values(self):
        self.assertTrue(is_color_numeric_scalar(0.5))
        self.assertTrue(is_color_numeric_scalar(0))
        self.assertTrue(is_color_numeric_scalar(1))

    def test_invalid_values(self):
        self.assertFalse(is_color_numeric_scalar(-0.1))
        self.assertFalse(is_color_numeric_scalar(1.1))
        self.assertFalse(is_color_numeric_scalar("0.5"))


class TestIsColor(TestCase):
    def test_valid_colors(self):
        self.assertTrue(is_color("red"))
        self.assertTrue(is_color("#FF0000"))
        self.assertTrue(is_color((1, 0, 0)))
        self.assertTrue(is_color(None))
        self.assertTrue(is_color(0.5))

    def test_invalid_colors(self):
        self.assertFalse(is_color("invalid"))
        self.assertFalse(is_color((2, 0, 0)))


class TestIsMarker(TestCase):
    def test_valid_markers(self):
        self.assertTrue(is_marker("o"))
        self.assertTrue(is_marker(1))
        self.assertTrue(is_marker(Path("test.png")))
        self.assertTrue(is_marker({"marker": "o", "fillstyle": "full"}))

    def test_invalid_markers(self):
        self.assertFalse(is_marker("invalid"))
        self.assertFalse(is_marker({"invalid": "value"}))
        self.assertFalse(is_marker(12))


class TestValidateRange(TestCase):
    def test_valid_range(self):
        validate_range(5, min_value=0, max_value=10)
        validate_range(0, min_value=0, max_value=10)
        validate_range(10, min_value=0, max_value=10)
        validate_range(5, min_value=0, max_value=10, strict=True)
        validate_range(0, min_value=-10, max_value=10, strict=True)
        validate_range(10, min_value=0, max_value=11, strict=True)

    def test_invalid_range(self):
        with self.assertRaises(ArgumentValidationError):
            validate_range(-1, 0, 10)
        with self.assertRaises(ArgumentValidationError):
            validate_range(11, 0, 10)

    def test_strict_range(self):
        validate_range(5, min_value=0, max_value=10, strict=True)
        with self.assertRaises(ArgumentValidationError):
            validate_range(0, min_value=0, max_value=10, strict=True)
        with self.assertRaises(ArgumentValidationError):
            validate_range(10, min_value=0, max_value=10, strict=True)


class TestValidateSequenceRange(TestCase):
    def test_valid_range(self):
        validate_sequence_range([1, 2, 3], 0, 5)
        validate_sequence_range([0, 5], 0, 5)

    def test_invalid_range(self):
        with self.assertRaises(ArgumentValidationError):
            validate_sequence_range([-1, 2, 3], 0, 5)
        with self.assertRaises(ArgumentValidationError):
            validate_sequence_range([1, 6, 3], 0, 5)


class TestValidateSequencesRange(TestCase):
    def test_valid_range(self):
        validate_sequences_range([[1, 2], [3, 4]], 0, 5)
        validate_sequences_range([[0, 5], [2, 3]], 0, 5)

    def test_invalid_range(self):
        with self.assertRaises(ArgumentValidationError):
            validate_sequences_range([[-1, 2], [3, 4]], 0, 5)
        with self.assertRaises(ArgumentValidationError):
            validate_sequences_range([[1, 6], [3, 4]], 0, 5)


class TestValidateSequence(TestCase):
    def test_valid_sequence(self):
        validate_sequence([1, 2, 3])
        validate_sequence((1, 2, 3))

    def test_invalid_sequence(self):
        with self.assertRaises(ArgumentValidationError):
            validate_sequence("test")
        with self.assertRaises(ArgumentValidationError):
            validate_sequence(123)


class TestValidateNumeric(TestCase):
    def test_valid_numeric(self):
        validate_numeric(1)
        validate_numeric(1.0)

    def test_invalid_numeric(self):
        with self.assertRaises(ArgumentValidationError):
            validate_numeric("1")
        with self.assertRaises(ArgumentValidationError):
            validate_numeric([1])


class TestValidateSequenceSizes(TestCase):
    def test_valid_sizes(self):
        self.assertEqual(validate_sequence_sizes([1, 2, 3], 3, True), [3])
        self.assertEqual(validate_sequence_sizes([1, 2], [2, 3], False), [2, 3])

    def test_invalid_sizes(self):
        with self.assertRaises(ArgumentValidationError):
            validate_sequence_sizes([1, 2, 3], 2, True)
        with self.assertRaises(ArgumentValidationError):
            validate_sequence_sizes([1, 2], [3, 4], False)


class TestValidateSequencesSizes(TestCase):
    def test_valid_sizes(self):
        self.assertEqual(
            validate_sequences_sizes([[1, 2], [3, 4]], [2, 2], True), [2, 2]
        )
        self.assertEqual(validate_sequences_sizes([[1], [2]], [1, 1], True), [1, 1])

    def test_invalid_sizes(self):
        with self.assertRaises(ArgumentValidationError):
            validate_sequences_sizes([[1, 2], [3]], 2, True)
        with self.assertRaises(ArgumentValidationError):
            validate_sequences_sizes([[1], [2]], [2, 1], True)


class TestStandardizeSequences(TestCase):
    def test_valid_sequences(self):
        self.assertEqual(standardize_sequences([[1, 2], [3, 4]]), [[1, 2], [3, 4]])
        self.assertEqual(standardize_sequences([(1, 2), (3, 4)]), [[1, 2], [3, 4]])

    def test_invalid_sequences(self):
        with self.assertRaises(ArgumentValidationError):
            standardize_sequences([1, 2, 3])
        with self.assertRaises(ArgumentValidationError):
            standardize_sequences(["test"])


class TestValidateTupleSizes(TestCase):
    def test_valid_sizes(self):
        self.assertEqual(validate_tuple_sizes((1, 2), 2), [2])
        self.assertEqual(validate_tuple_sizes((1, 2, 3), [2, 3]), [2, 3])

    def test_invalid_sizes(self):
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sizes((1, 2), 3)
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sizes((1, 2), [3, 4])


class TestValidateTupleSequenceSizes(TestCase):
    def test_valid_sizes(self):
        self.assertEqual(
            validate_tuple_sequence_sizes([(1, 2), (3, 4)], [2, 2], True), [2, 2]
        )
        self.assertEqual(validate_tuple_sequence_sizes([(1, 2), (3, 4)], 2, True), [2])
        self.assertEqual(
            validate_tuple_sequence_sizes([(1,), (2,)], [1, 1], True), [1, 1]
        )

    def test_invalid_sizes(self):
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequence_sizes([(1, 2), (3,)], 2, True)
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequence_sizes([(1,), (2,)], [2, 1], True)


class TestValidateTupleSequence(TestCase):
    def test_valid_sequence(self):
        validate_tuple_sequence([(1, 2), (3, 4)])

    def test_invalid_sequence(self):
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequence([1, 2])
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequence(["test"])


class TestValidateTupleSequences(TestCase):
    def test_valid_sequences(self):
        validate_tuple_sequences([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])

    def test_invalid_sequences(self):
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequences([[1, 2], [3, 4]])
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequences([["test"], ["test"]])


class TestValidateTupleSequencesSizes(TestCase):
    def test_valid_sizes(self):
        self.assertEqual(validate_tuple_sequences_sizes([[(1, 2), (3, 4)]], 2), [2])
        self.assertEqual(validate_tuple_sequences_sizes([[(1,), (2,)]], [1, 1]), [1, 1])

    def test_invalid_sizes(self):
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequences_sizes([[(1, 2, 3), (3, 4)]], 2)
        with self.assertRaises(ArgumentValidationError):
            validate_tuple_sequences_sizes([[(1,), (2,)]], [2, 3])


class TestValidateSingleColor(TestCase):
    def test_valid_colors(self):
        self.assertEqual(validate_single_color("red"), "red")
        self.assertEqual(validate_single_color("#FF0000"), "#FF0000")
        self.assertEqual(validate_single_color((1, 0, 0)), (1, 0, 0))
        self.assertEqual(validate_single_color("face", allow_face_literal=True), "face")

    def test_invalid_colors(self):
        with self.assertRaises(ArgumentValidationError):
            validate_single_color("invalid")
        with self.assertRaises(ArgumentValidationError):
            validate_single_color((2, 0, 0))
        with self.assertRaises(ArgumentValidationError):
            validate_single_color("face", allow_face_literal=False)


if __name__ == "__main__":
    unittest.main()
