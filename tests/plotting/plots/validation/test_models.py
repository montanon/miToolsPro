import unittest
from unittest import TestCase

import numpy as np
import pandas as pd
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.validation.models import (
    BoolParam,
    BoolSequenceParam,
    BoolSequencesParam,
    ColorParam,
    ColorSequenceParam,
    ColorSequencesParam,
    DictParam,
    DictSequenceParam,
    DictSequencesParam,
    EdgeColorParam,
    EdgeColorSequenceParam,
    EdgeColorSequencesParam,
    LiteralParam,
    LiteralSequenceParam,
    LiteralSequencesParam,
    MarkerParam,
    MarkerSequenceParam,
    MarkerSequencesParam,
    NormalizationParam,
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
    NumericTupleParam,
    NumericTupleSequenceParam,
    NumericTupleSequencesParam,
    Param,
    RangeParam,
    SequenceParam,
    SequencesParam,
    StrParam,
    StrSequenceParam,
    StrSequencesParam,
)


class TestParam(TestCase):
    def test_init_with_value(self):
        param = Param[int](value=42)
        self.assertEqual(param.value, 42)

    def test_init_with_dict(self):
        param = Param[int].model_validate({"value": 42})
        self.assertEqual(param.value, 42)

    def test_init_with_invalid_type(self):
        with self.assertRaises(ValidationError):
            Param[int](value="not_an_int")

    def test_init_with_extra_fields(self):
        with self.assertRaises(ValidationError):
            Param[int](value=42, extra_field="should_fail")

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            Param[int](value=None)

    def test_init_with_complex_type(self):
        class ComplexType:
            pass

        param = Param[ComplexType](value=ComplexType())
        self.assertIsInstance(param.value, ComplexType)

    def test_init_with_sequence(self):
        param = Param[list](value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_type_sequence(self):
        param = Param[list[int]](value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_dict_type(self):
        param = Param[dict](value={"key": "value"})
        self.assertEqual(param.value, {"key": "value"})

    def test_init_with_custom_type(self):
        class CustomType:
            def __init__(self, x):
                self.x = x

        param = Param[CustomType](value=CustomType(42))
        self.assertEqual(param.value.x, 42)

    def test_init_with_float(self):
        param = Param[float](value=3.14)
        self.assertEqual(param.value, 3.14)

    def test_init_with_string(self):
        param = Param[str](value="test")
        self.assertEqual(param.value, "test")

    def test_init_with_boolean(self):
        param = Param[bool](value=True)
        self.assertEqual(param.value, True)

    def test_init_with_empty_list(self):
        param = Param[list](value=[])
        self.assertEqual(param.value, [])

    def test_init_with_empty_dict(self):
        param = Param[dict](value={})
        self.assertEqual(param.value, {})

    def test_init_with_union_type(self):
        param = Param[int | str](value=42)
        self.assertEqual(param.value, 42)
        param = Param[int | str](value="test")
        self.assertEqual(param.value, "test")

    def test_init_with_optional_type(self):
        from typing import Optional

        param = Param[Optional[int]](value=42)
        self.assertEqual(param.value, 42)
        param = Param[Optional[int]](value=None)
        self.assertEqual(param.value, None)


class TestRangeParam(TestCase):
    def test_init_with_valid_value(self):
        param = RangeParam(value=5)
        self.assertEqual(param.value, 5)
        self.assertEqual(param.min_value, float("-inf"))
        self.assertEqual(param.max_value, float("inf"))

    def test_init_with_custom_range(self):
        param = RangeParam(value=5, min_value=0, max_value=10)
        self.assertEqual(param.value, 5)
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 10)

    def test_init_with_float_value(self):
        param = RangeParam(value=3.14)
        self.assertEqual(param.value, 3.14)

    def test_init_with_value_at_min_boundary(self):
        param = RangeParam(value=0, min_value=0, max_value=10)
        self.assertEqual(param.value, 0)

    def test_init_with_value_at_max_boundary(self):
        param = RangeParam(value=10, min_value=0, max_value=10)
        self.assertEqual(param.value, 10)

    def test_init_with_value_below_min(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=-1, min_value=0, max_value=10)
        self.assertIn("Value -1 is not in range [0, 10]", str(context.exception))

    def test_init_with_value_above_max(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=11, min_value=0, max_value=10)
        self.assertIn("Value 11 is not in range [0, 10]", str(context.exception))

    def test_init_with_non_numeric_value(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value="not_a_number")

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=None)

    def test_init_with_none_min_value(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=5, min_value=None)

    def test_init_with_none_max_value(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=5, max_value=None)

    def test_init_with_negative_infinity(self):
        param = RangeParam(value=-float("inf"))
        self.assertEqual(param.value, float("-inf"))

    def test_init_with_positive_infinity(self):
        param = RangeParam(value=float("inf"))
        self.assertEqual(param.value, float("inf"))

    def test_init_with_nan(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=float("nan"))

    def test_init_with_invalid_min_max(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=5, min_value=10, max_value=0)
        self.assertIn("Value 5 is not in range [10, 0]", str(context.exception))

    def test_init_with_equal_min_max(self):
        param = RangeParam(value=5, min_value=5, max_value=5)
        self.assertEqual(param.value, 5)

    def test_init_with_float_min_max(self):
        param = RangeParam(value=5.5, min_value=0.0, max_value=10.0)
        self.assertEqual(param.value, 5.5)

    def test_init_with_mixed_numeric_types(self):
        param = RangeParam(value=5, min_value=0.0, max_value=10)
        self.assertEqual(param.value, 5)

    def test_init_with_very_large_numbers(self):
        param = RangeParam(value=1e100, min_value=-1e100, max_value=1e100)
        self.assertEqual(param.value, 1e100)

    def test_init_with_very_small_numbers(self):
        param = RangeParam(value=-1e100, min_value=-1e100, max_value=1e100)
        self.assertEqual(param.value, -1e100)

    def test_init_with_strict_range_inclusive(self):
        param = RangeParam(value=5, min_value=0, max_value=10, strict=False)
        self.assertEqual(param.value, 5)
        self.assertEqual(param.strict, False)

    def test_init_with_strict_range_exclusive(self):
        param = RangeParam(value=5, min_value=0, max_value=10, strict=True)
        self.assertEqual(param.value, 5)
        self.assertEqual(param.strict, True)

    def test_strict_range_at_min_boundary(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=0, min_value=0, max_value=10, strict=True)
        self.assertIn("Value 0 is not in range (0, 10)", str(context.exception))

    def test_strict_range_at_max_boundary(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=10, min_value=0, max_value=10, strict=True)
        self.assertIn("Value 10 is not in range (0, 10)", str(context.exception))

    def test_strict_range_with_equal_min_max(self):
        with self.assertRaises(ValidationError) as context:
            RangeParam(value=5, min_value=5, max_value=5, strict=True)
        self.assertIn("Value 5 is not in range (5, 5)", str(context.exception))

    def test_strict_range_with_infinity(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=float("inf"), strict=True)

    def test_strict_range_with_negative_infinity(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=float("-inf"), strict=True)

    def test_strict_range_with_very_small_interval(self):
        param = RangeParam(value=0.000001, min_value=0, max_value=0.000002, strict=True)
        self.assertEqual(param.value, 0.000001)

    def test_strict_range_with_very_large_interval(self):
        with self.assertRaises(ValidationError):
            RangeParam(value=1e100, min_value=-1e100, max_value=1e100, strict=True)

    def test_strict_range_with_float_precision(self):
        param = RangeParam(value=0.5, min_value=0.0, max_value=1.0, strict=True)
        self.assertEqual(param.value, 0.5)

    def test_strict_range_with_mixed_numeric_types(self):
        param = RangeParam(value=5, min_value=0.0, max_value=10, strict=True)
        self.assertEqual(param.value, 5)


class TestSpecializedParams(TestCase):
    def test_str_param_valid_values(self):
        param = StrParam(value="test")
        self.assertEqual(param.value, "test")
        param = StrParam(value="")
        self.assertEqual(param.value, "")
        param = StrParam(value="123")
        self.assertEqual(param.value, "123")

    def test_str_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            StrParam(value=123)
        with self.assertRaises(ValidationError):
            StrParam(value=None)
        with self.assertRaises(ValidationError):
            StrParam(value=[])

    def test_bool_param_valid_values(self):
        param = BoolParam(value=True)
        self.assertEqual(param.value, True)
        param = BoolParam(value=False)
        self.assertEqual(param.value, False)

    def test_bool_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            BoolParam(value=1)
        with self.assertRaises(ValidationError):
            BoolParam(value="true")
        with self.assertRaises(ValidationError):
            BoolParam(value=None)

    def test_numeric_param_valid_values(self):
        param = NumericParam(value=42)
        self.assertEqual(param.value, 42)
        param = NumericParam(value=3.14)
        self.assertEqual(param.value, 3.14)
        param = NumericParam(value=-1)
        self.assertEqual(param.value, -1)

    def test_numeric_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            NumericParam(value="42")
        with self.assertRaises(ValidationError):
            NumericParam(value=None)
        with self.assertRaises(ValidationError):
            NumericParam(value=[])

    def test_dict_param_valid_values(self):
        param = DictParam(value={})
        self.assertEqual(param.value, {})
        param = DictParam(value={"key": "value"})
        self.assertEqual(param.value, {"key": "value"})
        param = DictParam(value={"nested": {"key": "value"}})
        self.assertEqual(param.value, {"nested": {"key": "value"}})

    def test_dict_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            DictParam(value="not_a_dict")
        with self.assertRaises(ValidationError):
            DictParam(value=None)
        with self.assertRaises(ValidationError):
            DictParam(value=[])

    def test_str_param_with_extra_fields(self):
        with self.assertRaises(ValidationError):
            StrParam(value="test", extra_field="should_fail")

    def test_bool_param_with_extra_fields(self):
        with self.assertRaises(ValidationError):
            BoolParam(value=True, extra_field="should_fail")

    def test_numeric_param_with_extra_fields(self):
        with self.assertRaises(ValidationError):
            NumericParam(value=42, extra_field="should_fail")

    def test_dict_param_with_extra_fields(self):
        with self.assertRaises(ValidationError):
            DictParam(value={}, extra_field="should_fail")

    def test_str_param_with_dict_initialization(self):
        param = StrParam.model_validate({"value": "test"})
        self.assertEqual(param.value, "test")

    def test_bool_param_with_dict_initialization(self):
        param = BoolParam.model_validate({"value": True})
        self.assertEqual(param.value, True)

    def test_numeric_param_with_dict_initialization(self):
        param = NumericParam.model_validate({"value": 42})
        self.assertEqual(param.value, 42)

    def test_dict_param_with_dict_initialization(self):
        param = DictParam.model_validate({"value": {"key": "value"}})
        self.assertEqual(param.value, {"key": "value"})


class TestSequenceParam(TestCase):
    def test_init_with_list(self):
        param = SequenceParam[int](value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_tuple(self):
        param = SequenceParam[int](value=(1, 2, 3))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_ndarray(self):
        param = SequenceParam[int](value=np.array([1, 2, 3]))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_series(self):
        param = SequenceParam[int](value=pd.Series([1, 2, 3]))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_empty_sequence(self):
        param = SequenceParam[int](value=[])
        self.assertEqual(param.value, [])
        param = SequenceParam[int](value=())
        self.assertEqual(param.value, [])
        param = SequenceParam[int](value=np.array([]))
        self.assertEqual(param.value, [])
        param = SequenceParam[int](value=pd.Series([]))
        self.assertEqual(param.value, [])

    def test_init_with_single_value(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=42)

    def test_init_with_dict_initialization(self):
        param = SequenceParam[int].model_validate({"value": [1, 2, 3]})
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_nested_sequences(self):
        param = SequenceParam[list](value=[[1, 2], [3, 4]])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_nested_sequences_with_types(self):
        param = SequenceParam[list[int]](value=[[1, 2], [3, 4]])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_mixed_types(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=[1, "2", 3])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=None)

    def test_init_with_non_sequence(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=42.0)

    def test_init_with_dict_value(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value={"key": "value"})

    def test_init_with_set_value(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value={1, 2, 3})

    def test_init_with_str_sequence(self):
        param = SequenceParam[str](value=["a", "b", "c"])
        self.assertEqual(param.value, ["a", "b", "c"])

    def test_init_with_bool_sequence(self):
        param = SequenceParam[bool](value=[True, False, True])
        self.assertEqual(param.value, [True, False, True])

    def test_init_with_float_sequence(self):
        param = SequenceParam[float](value=[1.0, 2.0, 3.0])
        self.assertEqual(param.value, [1.0, 2.0, 3.0])

    def test_init_with_mixed_numeric_sequence(self):
        param = SequenceParam[float](value=[1, 2.0, 3])
        self.assertEqual(param.value, [1.0, 2.0, 3.0])

    def test_init_with_complex_sequence(self):
        class ComplexType:
            pass

        param = SequenceParam[ComplexType](value=[ComplexType(), ComplexType()])
        self.assertEqual(len(param.value), 2)
        self.assertIsInstance(param.value[0], ComplexType)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=[1, None, 3])

    def test_init_with_extra_fields(self):
        param = SequenceParam[int](value=[1, 2, 3], extra_field="should_fail")
        print(param)

    def test_init_with_single_none(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value=None)

    def test_init_with_empty_dict(self):
        with self.assertRaises(ValidationError):
            SequenceParam[int](value={})

    def test_init_with_dict_containing_sequence(self):
        param = SequenceParam[int].model_validate({"value": [1, 2, 3]})
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_nested_dict_sequence(self):
        param = SequenceParam[dict](value=[{"a": 1}, {"b": 2}])
        self.assertEqual(param.value, [{"a": 1}, {"b": 2}])

    def test_init_with_nested_sequence_param(self):
        param = SequenceParam[SequenceParam[int]](value=[[1, 2], [3, 4]])
        self.assertEqual([seq.value for seq in param.value], [[1, 2], [3, 4]])

    def test_init_with_very_large_sequence(self):
        large_sequence = list(range(1000))
        param = SequenceParam[int](value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_sequence_of_sequences(self):
        param = SequenceParam[list](value=[[1, 2], [3, 4], [5, 6]])
        self.assertEqual(param.value, [[1, 2], [3, 4], [5, 6]])

    def test_init_with_sequence_of_tuples(self):
        param = SequenceParam[tuple](value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [(1, 2), (3, 4)])

    def test_init_with_sequence_of_mixed_sequences(self):
        with self.assertRaises(ValidationError):
            SequenceParam[list](value=[[1, 2], (3, 4)])

    def test_init_with_sequence_of_none(self):
        with self.assertRaises(ValidationError):
            SequenceParam[list](value=[None, None])

    def test_init_with_sequence_of_empty_sequences(self):
        param = SequenceParam[list](value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_init_with_sequence_of_dicts(self):
        param = SequenceParam[dict](value=[{"a": 1}, {"b": 2}])
        self.assertEqual(param.value, [{"a": 1}, {"b": 2}])

    def test_init_with_sequence_of_custom_objects(self):
        class CustomObject:
            def __init__(self, x):
                self.x = x

        param = SequenceParam[CustomObject](value=[CustomObject(1), CustomObject(2)])
        self.assertEqual(len(param.value), 2)
        self.assertEqual(param.value[0].x, 1)
        self.assertEqual(param.value[1].x, 2)


class TestSpecializedSequenceParams(TestCase):
    def test_numeric_sequence_param_valid_values(self):
        param = NumericSequenceParam(value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])
        param = NumericSequenceParam(value=[1.0, 2.0, 3.0])
        self.assertEqual(param.value, [1.0, 2.0, 3.0])
        param = NumericSequenceParam(value=np.array([1, 2, 3]))
        self.assertEqual(param.value, [1, 2, 3])

    def test_numeric_sequence_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            NumericSequenceParam(value=["1", "2", "3"])
        with self.assertRaises(ValidationError):
            NumericSequenceParam(value=[True, False])
        with self.assertRaises(ValidationError):
            NumericSequenceParam(value=[None, None])

    def test_str_sequence_param_valid_values(self):
        param = StrSequenceParam(value=["a", "b", "c"])
        self.assertEqual(param.value, ["a", "b", "c"])
        param = StrSequenceParam(value=["1", "2", "3"])
        self.assertEqual(param.value, ["1", "2", "3"])
        param = StrSequenceParam(value=[""])
        self.assertEqual(param.value, [""])

    def test_str_sequence_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            StrSequenceParam(value=[1, 2, 3])
        with self.assertRaises(ValidationError):
            StrSequenceParam(value=[True, False])
        with self.assertRaises(ValidationError):
            StrSequenceParam(value=[None, None])

    def test_bool_sequence_param_valid_values(self):
        param = BoolSequenceParam(value=[True, False, True])
        self.assertEqual(param.value, [True, False, True])
        param = BoolSequenceParam(value=[False, False])
        self.assertEqual(param.value, [False, False])

    def test_bool_sequence_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            BoolSequenceParam(value=[1, 0])
        with self.assertRaises(ValidationError):
            BoolSequenceParam(value=["true", "false"])
        with self.assertRaises(ValidationError):
            BoolSequenceParam(value=[None, None])

    def test_dict_sequence_param_valid_values(self):
        param = DictSequenceParam(value=[{"a": 1}, {"b": 2}])
        self.assertEqual(param.value, [{"a": 1}, {"b": 2}])
        param = DictSequenceParam(value=[{}, {}])
        self.assertEqual(param.value, [{}, {}])
        param = DictSequenceParam(value=[{"nested": {"key": "value"}}])
        self.assertEqual(param.value, [{"nested": {"key": "value"}}])

    def test_dict_sequence_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            DictSequenceParam(value=["not_a_dict"])
        with self.assertRaises(ValidationError):
            DictSequenceParam(value=[None])
        with self.assertRaises(ValidationError):
            DictSequenceParam(value=[1, 2, 3])


class TestSequencesParam(TestCase):
    def test_init_with_list_of_lists(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_list_of_tuples(self):
        param = SequencesParam[int](value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_mixed_sequence_types(self):
        param = SequencesParam[int](value=[[1, 2], (3, 4)])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_numpy_arrays(self):
        param = SequencesParam[int](value=[np.array([1, 2]), np.array([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_pandas_series(self):
        param = SequencesParam[int](value=[pd.Series([1, 2]), pd.Series([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_sequence_param_instances(self):
        with self.assertRaises(ValidationError):
            inner_seq = SequenceParam[int](value=[1, 2])
            SequencesParam[int](value=[inner_seq])

    def test_init_with_mixed_sequence_param_instances(self):
        with self.assertRaises(ValidationError):
            inner_seq1 = SequenceParam[int](value=[1, 2])
            inner_seq2 = [3, 4]
            SequencesParam[int](value=[inner_seq1, inner_seq2])

    def test_init_with_empty_sequences(self):
        param = SequencesParam[int](value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_init_with_single_sequence(self):
        param = SequencesParam[int](value=[[1, 2, 3]])
        self.assertEqual(param.value, [[1, 2, 3]])

    def test_init_with_uneven_lengths(self):
        param = SequencesParam[int](value=[[1], [2, 3]])
        self.assertEqual(param.value, [[1], [2, 3]])

    def test_init_with_nested_sequences(self):
        param = SequencesParam[list](value=[[[1, 2]], [[3, 4]]])
        self.assertEqual(param.value, [[[1, 2]], [[3, 4]]])

    def test_init_with_nested_sequences_and_types(self):
        param = SequencesParam[list[int]](value=[[[1, 2]], [[3, 4]]])
        self.assertEqual(param.value, [[[1, 2]], [[3, 4]]])

    def test_init_with_very_large_sequences(self):
        large_sequence = [list(range(1000)), list(range(1000))]
        param = SequencesParam[int](value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_dict_initialization(self):
        param = SequencesParam[int].model_validate({"value": [[1, 2], [3, 4]]})
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_str_sequences(self):
        param = SequencesParam[str](value=[["a", "b"], ["c", "d"]])
        self.assertEqual(param.value, [["a", "b"], ["c", "d"]])

    def test_init_with_bool_sequences(self):
        param = SequencesParam[bool](value=[[True, False], [False, True]])
        self.assertEqual(param.value, [[True, False], [False, True]])

    def test_init_with_float_sequences(self):
        param = SequencesParam[float](value=[[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(param.value, [[1.0, 2.0], [3.0, 4.0]])

    def test_init_with_dict_sequences(self):
        param = SequencesParam[dict](value=[[{"a": 1}], [{"b": 2}]])
        self.assertEqual(param.value, [[{"a": 1}], [{"b": 2}]])

    def test_init_with_mixed_numeric_types(self):
        param = SequencesParam[float](value=[[1, 2.0], [3, 4.0]])
        self.assertEqual(param.value, [[1.0, 2.0], [3.0, 4.0]])

    def test_init_with_invalid_sequence_param(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=[42])

    def test_init_with_invalid_nested_type(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=[[1, 2], ["3", "4"]])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=[[1, 2], None])

    def test_init_with_none_in_nested_sequence(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=[[1, None], [3, 4]])

    def test_init_with_empty_dict(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value={})

    def test_init_with_dict_containing_invalid_value(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int].model_validate({"value": [42]})

    def test_init_with_single_value(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value=42)

    def test_init_with_non_sequence(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value="not_a_sequence")

    def test_init_with_set(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value={1, 2})

    def test_init_with_dict_value(self):
        with self.assertRaises(ValidationError):
            SequencesParam[int](value={"key": "value"})

    def test_init_with_complex_nested_sequences(self):
        param = SequencesParam[list](value=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertEqual(param.value, [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_init_with_very_deep_nesting(self):
        param = SequencesParam[list](value=[[[[1]]], [[[2]]]])
        self.assertEqual(param.value, [[[[1]]], [[[2]]]])


class TestSpecializedSequencesParams(TestCase):
    def test_numeric_sequences_param_valid_values(self):
        param = NumericSequencesParam(value=[[1, 2], [3, 4]])
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        param = NumericSequencesParam(value=[[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(param.value, [[1.0, 2.0], [3.0, 4.0]])
        param = NumericSequencesParam(value=[np.array([1, 2]), np.array([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_numeric_sequences_param_with_mixed_numeric_types(self):
        param = NumericSequencesParam(value=[[1, 2.0], [3, 4.0]])
        self.assertEqual(param.value, [[1.0, 2.0], [3.0, 4.0]])

    def test_numeric_sequences_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            NumericSequencesParam(value=[["1", "2"], ["3", "4"]])
        with self.assertRaises(ValidationError):
            NumericSequencesParam(value=[[True, False], [False, True]])
        with self.assertRaises(ValidationError):
            NumericSequencesParam(value=[[None, None], [None, None]])

    def test_str_sequences_param_valid_values(self):
        param = StrSequencesParam(value=[["a", "b"], ["c", "d"]])
        self.assertEqual(param.value, [["a", "b"], ["c", "d"]])
        param = StrSequencesParam(value=[["1", "2"], ["3", "4"]])
        self.assertEqual(param.value, [["1", "2"], ["3", "4"]])
        param = StrSequencesParam(value=[["", ""], ["", ""]])
        self.assertEqual(param.value, [["", ""], ["", ""]])

    def test_str_sequences_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            StrSequencesParam(value=[[1, 2], [3, 4]])
        with self.assertRaises(ValidationError):
            StrSequencesParam(value=[[True, False], [False, True]])
        with self.assertRaises(ValidationError):
            StrSequencesParam(value=[[None, None], [None, None]])

    def test_bool_sequences_param_valid_values(self):
        param = BoolSequencesParam(value=[[True, False], [False, True]])
        self.assertEqual(param.value, [[True, False], [False, True]])
        param = BoolSequencesParam(value=[[False, False], [True, True]])
        self.assertEqual(param.value, [[False, False], [True, True]])

    def test_bool_sequences_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            BoolSequencesParam(value=[[1, 0], [0, 1]])
        with self.assertRaises(ValidationError):
            BoolSequencesParam(value=[["true", "false"], ["false", "true"]])
        with self.assertRaises(ValidationError):
            BoolSequencesParam(value=[[None, None], [None, None]])

    def test_dict_sequences_param_valid_values(self):
        param = DictSequencesParam(value=[[{"a": 1}], [{"b": 2}]])
        self.assertEqual(param.value, [[{"a": 1}], [{"b": 2}]])
        param = DictSequencesParam(value=[[{}, {}], [{}, {}]])
        self.assertEqual(param.value, [[{}, {}], [{}, {}]])
        param = DictSequencesParam(value=[[{"nested": {"key": "value"}}]])
        self.assertEqual(param.value, [[{"nested": {"key": "value"}}]])

    def test_dict_sequences_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            DictSequencesParam(value=[["not_a_dict"], ["not_a_dict"]])
        with self.assertRaises(ValidationError):
            DictSequencesParam(value=[[None], [None]])
        with self.assertRaises(ValidationError):
            DictSequencesParam(value=[[1, 2], [3, 4]])

    def test_numeric_sequences_param_with_uneven_lengths(self):
        param = NumericSequencesParam(value=[[1], [2, 3]])
        self.assertEqual(param.value, [[1], [2, 3]])

    def test_str_sequences_param_with_uneven_lengths(self):
        param = StrSequencesParam(value=[["a"], ["b", "c"]])
        self.assertEqual(param.value, [["a"], ["b", "c"]])

    def test_bool_sequences_param_with_uneven_lengths(self):
        param = BoolSequencesParam(value=[[True], [False, True]])
        self.assertEqual(param.value, [[True], [False, True]])

    def test_dict_sequences_param_with_uneven_lengths(self):
        param = DictSequencesParam(value=[[{"a": 1}], [{"b": 2}, {"c": 3}]])
        self.assertEqual(param.value, [[{"a": 1}], [{"b": 2}, {"c": 3}]])

    def test_numeric_sequences_param_with_empty_sequences(self):
        param = NumericSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_str_sequences_param_with_empty_sequences(self):
        param = StrSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_bool_sequences_param_with_empty_sequences(self):
        param = BoolSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_dict_sequences_param_with_empty_sequences(self):
        param = DictSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_numeric_sequences_param_with_single_sequence(self):
        param = NumericSequencesParam(value=[[1, 2, 3]])
        self.assertEqual(param.value, [[1, 2, 3]])

    def test_str_sequences_param_with_single_sequence(self):
        param = StrSequencesParam(value=[["a", "b", "c"]])
        self.assertEqual(param.value, [["a", "b", "c"]])

    def test_bool_sequences_param_with_single_sequence(self):
        param = BoolSequencesParam(value=[[True, False, True]])
        self.assertEqual(param.value, [[True, False, True]])

    def test_dict_sequences_param_with_single_sequence(self):
        param = DictSequencesParam(value=[[{"a": 1}, {"b": 2}]])
        self.assertEqual(param.value, [[{"a": 1}, {"b": 2}]])

    def test_numeric_sequences_param_with_numpy_arrays(self):
        param = NumericSequencesParam(value=[np.array([1, 2]), np.array([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_numeric_sequences_param_with_pandas_series(self):
        param = NumericSequencesParam(value=[pd.Series([1, 2]), pd.Series([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_numeric_sequences_param_with_very_large_sequences(self):
        large_sequence = [list(range(1000)), list(range(1000))]
        param = NumericSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_str_sequences_param_with_very_large_sequences(self):
        large_sequence = [["a"] * 1000, ["b"] * 1000]
        param = StrSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_bool_sequences_param_with_very_large_sequences(self):
        large_sequence = [[True] * 1000, [False] * 1000]
        param = BoolSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_dict_sequences_param_with_very_large_sequences(self):
        large_sequence = [[{"a": 1}] * 1000, [{"b": 2}] * 1000]
        param = DictSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)


class TestNumericTupleParam(TestCase):
    def test_init_with_valid_tuple(self):
        param = NumericTupleParam(value=(1, 2, 3))
        self.assertEqual(param.value, (1, 2, 3))
        self.assertIsNone(param.sizes)

    def test_init_with_valid_tuple_and_size(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=3)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [3])

    def test_init_with_valid_tuple_and_sizes(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [2, 3, 4])

    def test_init_with_mixed_numeric_types(self):
        param = NumericTupleParam(value=(1, 2.0, 3))
        self.assertEqual(param.value, (1, 2.0, 3))

    def test_init_with_float_tuple(self):
        param = NumericTupleParam(value=(1.0, 2.0, 3.0))
        self.assertEqual(param.value, (1.0, 2.0, 3.0))

    def test_init_with_int_tuple(self):
        param = NumericTupleParam(value=(1, 2, 3))
        self.assertEqual(param.value, (1, 2, 3))

    def test_init_with_empty_tuple(self):
        param = NumericTupleParam(value=())
        self.assertEqual(param.value, ())

    def test_init_with_single_element_tuple(self):
        param = NumericTupleParam(value=(1,))
        self.assertEqual(param.value, (1,))

    def test_init_with_very_large_tuple(self):
        large_tuple = tuple(range(1000))
        param = NumericTupleParam(value=large_tuple)
        self.assertEqual(param.value, large_tuple)

    def test_init_with_invalid_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=[1, 2, 3])
        with self.assertRaises(ValidationError):
            NumericTupleParam(value="not_a_tuple")
        with self.assertRaises(ValidationError):
            NumericTupleParam(value={"key": "value"})

    def test_init_with_non_numeric_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=("1", "2", "3"))
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(True, False, True))
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(None, None, None))

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=None)

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=2)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[2, 4])

    def test_init_with_valid_size_range(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [2, 3, 4])

    def test_init_with_single_size(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=3)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [3])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[])

    def test_init_with_none_size(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=None)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertIsNone(param.sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes="not_a_size")
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes={"key": "value"})

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=-1)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[-1, -2])

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(), sizes=0)

    def test_init_with_very_large_size(self):
        param = NumericTupleParam(value=tuple(range(1000)), sizes=1000)
        self.assertEqual(len(param.value), 1000)
        self.assertEqual(param.sizes, [1000])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=3.0)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=np.array([2, 3, 4]))

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=pd.Series([2, 3, 4]))

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[2, None, 4])

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[2, -1, 4])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=[2, 3, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [2, 3, 3, 4])

    def test_init_with_single_size_matching(self):
        param = NumericTupleParam(value=(1, 2), sizes=2)
        self.assertEqual(param.value, (1, 2))
        self.assertEqual(param.sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleParam(value=(1, 2, 3), sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.sizes, [2, 3, 4])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), sizes=[4, 5, 6])

    def test_init_with_single_element_and_size(self):
        param = NumericTupleParam(value=(1,), sizes=1)
        self.assertEqual(param.value, (1,))
        self.assertEqual(param.sizes, [1])


class TestNumericTupleSequenceParam(TestCase):
    def test_init_with_valid_sequence(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertIsNone(param.sizes)

    def test_init_with_valid_sequence_and_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])

    def test_init_with_valid_sequence_and_sizes(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_mixed_numeric_types(self):
        param = NumericTupleSequenceParam(value=[(1, 2.0), (3.0, 4)])
        self.assertEqual(param.value, [(1, 2.0), (3.0, 4)])

    def test_init_with_float_tuples(self):
        param = NumericTupleSequenceParam(value=[(1.0, 2.0), (3.0, 4.0)])
        self.assertEqual(param.value, [(1.0, 2.0), (3.0, 4.0)])

    def test_init_with_int_tuples(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [(1, 2), (3, 4)])

    def test_init_with_empty_sequence(self):
        param = NumericTupleSequenceParam(value=[])
        self.assertEqual(param.value, [])

    def test_init_with_single_element_sequence(self):
        param = NumericTupleSequenceParam(value=[(1, 2)])
        self.assertEqual(param.value, [(1, 2)])

    def test_init_with_very_large_sequence(self):
        large_sequence = [(i, i + 1) for i in range(1000)]
        param = NumericTupleSequenceParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_invalid_sequence_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value="not_a_sequence")
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value={"key": "value"})

    def test_init_with_non_tuple_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[1, 2, 3])
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=["1", "2", "3"])
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[True, False, True])

    def test_init_with_non_numeric_tuple_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[("1", "2"), ("3", "4")])
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(True, False), (False, True)])
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(None, None), (None, None)])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), None])

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=3)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[3, 4])

    def test_init_with_valid_size_range(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_single_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[])

    def test_init_with_none_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=None)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertIsNone(param.sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes="not_a_size")
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes={"key": "value"})

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=-1)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[-1, -2])

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(), ()], sizes=0)

    def test_init_with_very_large_size(self):
        large_sequence = [(i,) for i in range(1000)]
        param = NumericTupleSequenceParam(value=large_sequence, sizes=1)
        self.assertEqual(len(param.value), 1000)
        self.assertEqual(param.sizes, [1])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=2.0)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=np.array([2, 3]))

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=pd.Series([2, 3]))

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[2, None, 3])

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[2, -1, 3])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[2, 2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2, 2, 3])

    def test_init_with_single_size_matching(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], sizes=[3, 4])

    def test_init_with_empty_tuple_sequence(self):
        param = NumericTupleSequenceParam(value=[(), ()])
        self.assertEqual(param.value, [(), ()])

    def test_init_with_single_element_sequence_and_size(self):
        param = NumericTupleSequenceParam(value=[(1,)], sizes=1)
        self.assertEqual(param.value, [(1,)])
        self.assertEqual(param.sizes, [1])

    def test_init_with_uneven_tuple_sizes(self):
        param = NumericTupleSequenceParam(
            value=[(1,), (2, 3), (4, 5, 6)], sizes=[1, 2, 3]
        )
        self.assertEqual(param.value, [(1,), (2, 3), (4, 5, 6)])
        self.assertEqual(param.sizes, [1, 2, 3])

    def test_init_with_numpy_arrays_in_tuples(self):
        param = NumericTupleSequenceParam(
            value=[(np.array(1), np.array(2)), (np.array(3), np.array(4))]
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])

    def test_init_with_pandas_series_in_tuples(self):
        param = NumericTupleSequenceParam(
            value=[(pd.Series([1]), pd.Series([2])), (pd.Series([3]), pd.Series([4]))]
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])

    def test_init_with_mixed_sequence_types(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5), (6,)])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5), (6,)])

    def test_init_with_nested_sequences(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[[[[[(1, 2)]]]]])

    def test_init_with_dict_initialization(self):
        param = NumericTupleSequenceParam.model_validate(
            {"value": [(1, 2), (3, 4)], "sizes": 2}
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam.model_validate(
                {"value": [(1, 2), (3, 4)], "sizes": "invalid"}
            )


class TestNumericTupleSequencesParam(TestCase):
    def test_init_with_valid_sequences(self):
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertIsNone(param.sizes)

    def test_init_with_valid_sequences_and_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])

    def test_init_with_valid_sequences_and_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_mixed_numeric_types(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2.0), (3.0, 4)], [(5.0, 6), (7, 8.0)]]
        )
        self.assertEqual(param.value, [[(1, 2.0), (3.0, 4)], [(5.0, 6), (7, 8.0)]])

    def test_init_with_float_tuples(self):
        param = NumericTupleSequencesParam(
            value=[[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0), (7.0, 8.0)]]
        )
        self.assertEqual(
            param.value, [[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0), (7.0, 8.0)]]
        )

    def test_init_with_int_tuples(self):
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])

    def test_init_with_empty_sequences(self):
        param = NumericTupleSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_init_with_single_element_sequences(self):
        param = NumericTupleSequencesParam(value=[[(1, 2)]])
        self.assertEqual(param.value, [[(1, 2)]])

    def test_init_with_very_large_sequences(self):
        large_sequence = [
            [(i, i + 1) for i in range(1000)],
            [(i, i + 1) for i in range(1000)],
        ]
        param = NumericTupleSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_invalid_sequence_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value="not_a_sequence")
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value={"key": "value"})

    def test_init_with_non_sequence_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[1, 2, 3])
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=["1", "2", "3"])
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[True, False, True])

    def test_init_with_non_tuple_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[1, 2], [3, 4]])
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[["1", "2"], ["3", "4"]])
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[True, False], [False, True]])

    def test_init_with_non_numeric_tuple_elements(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[("1", "2"), ("3", "4")], [("5", "6"), ("7", "8")]]
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(True, False), (False, True)], [(True, False), (False, True)]]
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(None, None), (None, None)], [(None, None), (None, None)]]
            )

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[(1, 2), None], [(3, 4), (5, 6)]])

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=3
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[3, 4]
            )

    def test_init_with_valid_size_range(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_single_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[]
            )

    def test_init_with_none_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=None
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertIsNone(param.sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes="not_a_size"
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes={"key": "value"}
            )

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=-1
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[-1, -2]
            )

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[(), ()], [(), ()]], sizes=0)

    def test_init_with_very_large_size(self):
        large_sequence = [[(i,) for i in range(1000)], [(i,) for i in range(1000)]]
        param = NumericTupleSequencesParam(value=large_sequence, sizes=1)
        self.assertEqual(len(param.value), 2)
        self.assertEqual(param.sizes, [1])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], sizes=2.0)
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=np.array([2, 3])
            )

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], sizes=pd.Series([2, 3]))

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[2, None, 3]
            )

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], sizes=[2, -1, 3])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[2, 2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2, 2, 3])

    def test_init_with_single_size_matching(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], sizes=[3, 4]
            )

    def test_init_with_empty_tuple_sequences(self):
        param = NumericTupleSequencesParam(value=[[(), ()], [(), ()]])
        self.assertEqual(param.value, [[(), ()], [(), ()]])

    def test_init_with_single_element_sequences_and_size(self):
        param = NumericTupleSequencesParam(value=[[(1,)], [(2,)]], sizes=1)
        self.assertEqual(param.value, [[(1,)], [(2,)]])
        self.assertEqual(param.sizes, [1])

    def test_init_with_uneven_tuple_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1,), (2, 3), (4, 5, 6)]], sizes=[1, 2, 3]
        )
        self.assertEqual(param.value, [[(1,), (2, 3), (4, 5, 6)]])
        self.assertEqual(param.sizes, [1, 2, 3])

    def test_init_with_numpy_arrays_in_tuples(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[(np.array(1), np.array(2)), (np.array(3), np.array(4))]
            )

    def test_init_with_pandas_series_in_tuples(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[
                    (pd.Series([1]), pd.Series([2])),
                    (pd.Series([3]), pd.Series([4])),
                ]
            )

    def test_init_with_mixed_sequence_types(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4, 5)], [(6,), (7, 8, 9, 10)]], sizes=[1, 2, 3, 4]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4, 5)], [(6,), (7, 8, 9, 10)]])
        self.assertEqual(param.sizes, [1, 2, 3, 4])

    def test_init_with_nested_sequences(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[[[[(1, 2)]]]]])

    def test_init_with_dict_initialization(self):
        param = NumericTupleSequencesParam.model_validate(
            {"value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]], "sizes": 2}
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam.model_validate(
                {"value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]], "sizes": "invalid"}
            )

    def test_init_with_uneven_outer_sequence_lengths(self):
        param = NumericTupleSequencesParam(value=[[(1, 2)], [(3, 4), (5, 6), (7, 8)]])
        self.assertEqual(param.value, [[(1, 2)], [(3, 4), (5, 6), (7, 8)]])

    def test_init_with_uneven_inner_sequence_lengths(self):
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4)], [(5, 6)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6)]])

    def test_init_with_empty_inner_sequences(self):
        param = NumericTupleSequencesParam(value=[[], [(1, 2), (3, 4)]])
        self.assertEqual(param.value, [[], [(1, 2), (3, 4)]])

    def test_init_with_empty_outer_sequence(self):
        param = NumericTupleSequencesParam(value=[])
        self.assertEqual(param.value, [])

    def test_init_with_single_outer_sequence(self):
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4)]])

    def test_init_with_single_inner_sequence(self):
        param = NumericTupleSequencesParam(value=[[(1, 2)]])
        self.assertEqual(param.value, [[(1, 2)]])

    def test_init_with_single_tuple(self):
        param = NumericTupleSequencesParam(value=[[(1, 2)]])
        self.assertEqual(param.value, [[(1, 2)]])

    def test_init_with_very_deep_nesting(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[[[[(1, 2)]]]]])

    def test_init_with_mixed_numeric_types_in_tuples(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2.0), (3.0, 4)], [(5, 6.0), (7.0, 8)]]
        )
        self.assertEqual(param.value, [[(1, 2.0), (3.0, 4)], [(5, 6.0), (7.0, 8)]])

    def test_init_with_very_large_numbers(self):
        param = NumericTupleSequencesParam(
            value=[[(1e100, 2e100), (3e100, 4e100)], [(5e100, 6e100), (7e100, 8e100)]]
        )
        self.assertEqual(
            param.value,
            [[(1e100, 2e100), (3e100, 4e100)], [(5e100, 6e100), (7e100, 8e100)]],
        )

    def test_init_with_very_small_numbers(self):
        param = NumericTupleSequencesParam(
            value=[
                [(1e-100, 2e-100), (3e-100, 4e-100)],
                [(5e-100, 6e-100), (7e-100, 8e-100)],
            ]
        )
        self.assertEqual(
            param.value,
            [
                [(1e-100, 2e-100), (3e-100, 4e-100)],
                [(5e-100, 6e-100), (7e-100, 8e-100)],
            ],
        )


class TestColorParam(TestCase):
    def test_init_with_valid_hex_string(self):
        param = ColorParam(value="#FF0000")
        self.assertEqual(param.value, "#FF0000")

    def test_init_with_valid_named_color(self):
        param = ColorParam(value="red")
        self.assertEqual(param.value, "red")

    def test_init_with_valid_rgb_tuple(self):
        param = ColorParam(value=(255, 0, 0))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_tuple(self):
        param = ColorParam(value=(255, 0, 0, 0.5))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_rgb_list(self):
        param = ColorParam(value=[255, 0, 0])
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_list(self):
        param = ColorParam(value=[255, 0, 0, 0.5])
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_single_float(self):
        param = ColorParam(value=0.5)
        self.assertEqual(param.value, 0.5)

    def test_init_with_valid_none(self):
        param = ColorParam(value=None)
        self.assertIsNone(param.value)

    def test_init_with_valid_numpy_array(self):
        param = ColorParam(value=np.array([255, 0, 0]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_pandas_series(self):
        param = ColorParam(value=pd.Series([255, 0, 0]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgb_numpy_array(self):
        param = ColorParam(value=np.array([255, 0, 0], dtype=np.uint8))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_numpy_array(self):
        param = ColorParam(value=np.array([255, 0, 0, 0.5]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_rgb_pandas_series(self):
        param = ColorParam(value=pd.Series([255, 0, 0]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_pandas_series(self):
        param = ColorParam(value=pd.Series([255, 0, 0, 0.5]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_long_hex(self):
        param = ColorParam(value="#FF0000")
        self.assertEqual(param.value, "#FF0000")

    def test_init_with_valid_hex_with_alpha(self):
        param = ColorParam(value="#FF000080")
        self.assertEqual(param.value, "#FF000080")

    def test_init_with_valid_rgb_float_tuple(self):
        param = ColorParam(value=(1.0, 0.0, 0.0))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_float_tuple(self):
        param = ColorParam(value=(1.0, 0.0, 0.0, 0.5))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_rgb_float_list(self):
        param = ColorParam(value=[1.0, 0.0, 0.0])
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_float_list(self):
        param = ColorParam(value=[1.0, 0.0, 0.0, 0.5])
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_rgb_numpy_float_array(self):
        param = ColorParam(value=np.array([1.0, 0.0, 0.0]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_numpy_float_array(self):
        param = ColorParam(value=np.array([1.0, 0.0, 0.0, 0.5]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_valid_rgb_pandas_float_series(self):
        param = ColorParam(value=pd.Series([1.0, 0.0, 0.0]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0))

    def test_init_with_valid_rgba_pandas_float_series(self):
        param = ColorParam(value=pd.Series([1.0, 0.0, 0.0, 0.5]))
        self.assertEqual(param.value, (1.0, 0.0, 0.0, 0.5))

    def test_init_with_invalid_rgb_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="rgb(256, 0, 0)")

    def test_init_with_invalid_hex_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="#FF000")

    def test_init_with_invalid_named_color(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="notacolor")

    def test_init_with_invalid_rgb_tuple(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(256, 0, 0))

    def test_init_with_invalid_rgba_tuple(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(255, 0, 0, 1.1))

    def test_init_with_invalid_rgb_list(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=[256, 0, 0])

    def test_init_with_invalid_rgba_list(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=[255, 0, 0, 1.1])

    def test_init_with_invalid_single_int(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=256)

    def test_init_with_invalid_single_float(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=1.1)

    def test_init_with_invalid_numpy_array_shape(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([255, 0]))

    def test_init_with_invalid_pandas_series_shape(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([255, 0]))

    def test_init_with_invalid_rgb_numpy_array_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([256, 0, 0]))

    def test_init_with_invalid_rgba_numpy_array_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([255, 0, 0, 256]))

    def test_init_with_invalid_rgb_pandas_series_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([256, 0, 0]))

    def test_init_with_invalid_rgba_pandas_series_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([255, 0, 0, 256]))

    def test_init_with_invalid_hex_length(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="#FF000")

    def test_init_with_invalid_hex_characters(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="#FF000G")

    def test_init_with_invalid_hsl_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="hsl(361, 100%, 50%)")

    def test_init_with_invalid_hsla_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="hsla(0, 101%, 50%, 0.5)")

    def test_init_with_invalid_rgb_percent_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="rgb(101%, 0%, 0%)")

    def test_init_with_invalid_rgba_percent_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="rgba(100%, 0%, 0%, 1.1)")

    def test_init_with_invalid_rgb_float_tuple(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(1.1, 0.0, 0.0))

    def test_init_with_invalid_rgba_float_tuple(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(1.0, 0.0, 0.0, 1.1))

    def test_init_with_invalid_rgb_float_list(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=[1.1, 0.0, 0.0])

    def test_init_with_invalid_rgba_float_list(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=[1.0, 0.0, 0.0, 1.1])

    def test_init_with_invalid_rgb_numpy_float_array(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([1.1, 0.0, 0.0]))

    def test_init_with_invalid_rgba_numpy_float_array(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([1.0, 0.0, 0.0, 1.1]))

    def test_init_with_invalid_rgb_pandas_float_series(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([1.1, 0.0, 0.0]))

    def test_init_with_invalid_rgba_pandas_float_series(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([1.0, 0.0, 0.0, 1.1]))

    def test_init_with_invalid_type(self):
        with self.assertRaises(ValidationError):
            ColorParam(value={"key": "value"})

    def test_init_with_invalid_empty_string(self):
        with self.assertRaises(ValidationError):
            ColorParam(value="")

    def test_init_with_invalid_empty_list(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=[])

    def test_init_with_invalid_empty_tuple(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=())

    def test_init_with_invalid_empty_numpy_array(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=np.array([]))

    def test_init_with_invalid_empty_pandas_series(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=pd.Series([]))

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            ColorParam.model_validate({"value": "notacolor"})

    def test_init_with_valid_very_small_numbers(self):
        param = ColorParam(value=(1e-100, 0, 0))
        self.assertEqual(param.value, (1e-100, 0, 0))

    def test_init_with_valid_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(float("inf"), 0, 0))

    def test_init_with_valid_negative_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorParam(value=(float("-inf"), 0, 0))


class TestColorSequenceParam(TestCase):
    def test_init_with_valid_hex_strings(self):
        param = ColorSequenceParam(value=["#FF0000", "#00FF00", "#0000FF"])
        self.assertEqual(param.value, ["#FF0000", "#00FF00", "#0000FF"])

    def test_init_with_valid_named_colors(self):
        param = ColorSequenceParam(value=["red", "green", "blue"])
        self.assertEqual(param.value, ["red", "green", "blue"])

    def test_init_with_valid_rgb_tuples(self):
        param = ColorSequenceParam(value=[(255, 0, 0), (0, 255, 0), (0, 0, 255)])
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_valid_rgba_tuples(self):
        param = ColorSequenceParam(
            value=[(255, 0, 0, 0.5), (0, 255, 0, 0.5), (0, 0, 255, 0.5)]
        )
        self.assertEqual(
            param.value,
            [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)],
        )

    def test_init_with_valid_rgb_lists(self):
        param = ColorSequenceParam(value=[[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_valid_rgba_lists(self):
        param = ColorSequenceParam(
            value=[[255, 0, 0, 0.5], [0, 255, 0, 0.5], [0, 0, 255, 0.5]]
        )
        self.assertEqual(
            param.value,
            [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)],
        )

    def test_init_with_valid_float_tuples(self):
        param = ColorSequenceParam(
            value=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_valid_float_lists(self):
        param = ColorSequenceParam(
            value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_valid_numpy_arrays(self):
        param = ColorSequenceParam(
            value=[np.array([255, 0, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
        )
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_valid_pandas_series(self):
        param = ColorSequenceParam(
            value=[
                pd.Series([255, 0, 0]),
                pd.Series([0, 255, 0]),
                pd.Series([0, 0, 255]),
            ]
        )
        self.assertEqual(
            param.value, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        )

    def test_init_with_mixed_valid_colors(self):
        param = ColorSequenceParam(
            value=["red", (255, 0, 0), "#FF0000", [1.0, 0.0, 0.0]]
        )
        self.assertEqual(
            param.value, ["red", (1.0, 0.0, 0.0), "#FF0000", (1.0, 0.0, 0.0)]
        )

    def test_init_with_empty_sequence(self):
        param = ColorSequenceParam(value=[])
        self.assertEqual(param.value, [])

    def test_init_with_single_color(self):
        param = ColorSequenceParam(value=["red"])
        self.assertEqual(param.value, ["red"])

    def test_init_with_very_large_sequence(self):
        large_sequence = ["red"] * 1000
        param = ColorSequenceParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_invalid_hex_strings(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=["#FF000", "#00FF00", "#0000FF"])

    def test_init_with_invalid_named_colors(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=["notacolor", "green", "blue"])

    def test_init_with_invalid_rgb_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[(256, 0, 0), (0, 255, 0), (0, 0, 255)])

    def test_init_with_invalid_rgba_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[(255, 0, 0, 1.1), (0, 255, 0, 0.5), (0, 0, 255, 0.5)]
            )

    def test_init_with_invalid_rgb_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[[256, 0, 0], [0, 255, 0], [0, 0, 255]])

    def test_init_with_invalid_rgba_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[[255, 0, 0, 1.1], [0, 255, 0, 0.5], [0, 0, 255, 0.5]]
            )

    def test_init_with_invalid_float_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[(1.1, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            )

    def test_init_with_invalid_float_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )

    def test_init_with_invalid_numpy_array_shape(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[np.array([255, 0]), np.array([0, 255, 0]), np.array([0, 0, 255])]
            )

    def test_init_with_invalid_pandas_series_shape(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(
                value=[
                    pd.Series([255, 0]),
                    pd.Series([0, 255, 0]),
                    pd.Series([0, 0, 255]),
                ]
            )

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=None)

    def test_init_with_none_in_sequence(self):
        param = ColorSequenceParam(value=["red", None, "blue"])
        self.assertEqual(param.value, ["red", None, "blue"])

    def test_init_with_invalid_type(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[{"key": "value"}])

    def test_init_with_empty_string(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[""])

    def test_init_with_empty_list(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[[]])

    def test_init_with_empty_tuple(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[()])

    def test_init_with_empty_numpy_array(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[np.array([])])

    def test_init_with_empty_pandas_series(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[pd.Series([])])

    def test_init_with_dict_initialization(self):
        param = ColorSequenceParam.model_validate({"value": ["red", "green", "blue"]})
        self.assertEqual(param.value, ["red", "green", "blue"])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam.model_validate({"value": ["notacolor"]})

    def test_init_with_very_small_numbers(self):
        param = ColorSequenceParam(
            value=[(1e-100, 0, 0), (0, 1e-100, 0), (0, 0, 1e-100)]
        )
        self.assertEqual(
            param.value, [(1e-100, 0.0, 0.0), (0.0, 1e-100, 0.0), (0.0, 0.0, 1e-100)]
        )

    def test_init_with_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[(float("inf"), 0, 0), (0, 255, 0), (0, 0, 255)])

    def test_init_with_negative_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorSequenceParam(value=[(float("-inf"), 0, 0), (0, 255, 0), (0, 0, 255)])


class TestColorSequencesParam(TestCase):
    def test_init_with_valid_hex_strings(self):
        param = ColorSequencesParam(
            value=[["#FF0000", "#00FF00"], ["#0000FF", "#FFFF00"]]
        )
        self.assertEqual(param.value, [["#FF0000", "#00FF00"], ["#0000FF", "#FFFF00"]])

    def test_init_with_valid_named_colors(self):
        param = ColorSequencesParam(value=[["red", "green"], ["blue", "yellow"]])
        self.assertEqual(param.value, [["red", "green"], ["blue", "yellow"]])

    def test_init_with_valid_rgb_tuples(self):
        param = ColorSequencesParam(
            value=[[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 0)]]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_valid_rgba_tuples(self):
        param = ColorSequencesParam(
            value=[
                [(255, 0, 0, 0.5), (0, 255, 0, 0.5)],
                [(0, 0, 255, 0.5), (255, 255, 0, 0.5)],
            ]
        )
        self.assertEqual(
            param.value,
            [
                [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5)],
                [(0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5)],
            ],
        )

    def test_init_with_valid_rgb_lists(self):
        param = ColorSequencesParam(
            value=[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_valid_rgba_lists(self):
        param = ColorSequencesParam(
            value=[
                [[255, 0, 0, 0.5], [0, 255, 0, 0.5]],
                [[0, 0, 255, 0.5], [255, 255, 0, 0.5]],
            ]
        )
        self.assertEqual(
            param.value,
            [
                [(1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.5)],
                [(0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.0, 0.5)],
            ],
        )

    def test_init_with_valid_float_tuples(self):
        param = ColorSequencesParam(
            value=[
                [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
            ]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_valid_float_lists(self):
        param = ColorSequencesParam(
            value=[
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_valid_numpy_arrays(self):
        param = ColorSequencesParam(
            value=[
                [np.array([255, 0, 0]), np.array([0, 255, 0])],
                [np.array([0, 0, 255]), np.array([255, 255, 0])],
            ]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_valid_pandas_series(self):
        param = ColorSequencesParam(
            value=[
                [pd.Series([255, 0, 0]), pd.Series([0, 255, 0])],
                [pd.Series([0, 0, 255]), pd.Series([255, 255, 0])],
            ]
        )
        self.assertEqual(
            param.value,
            [[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]],
        )

    def test_init_with_mixed_valid_colors(self):
        param = ColorSequencesParam(
            value=[
                ["red", (255, 0, 0), "#FF0000", [1.0, 0.0, 0.0]],
                ["blue", (0, 0, 255), "#0000FF", [0.0, 0.0, 1.0]],
            ]
        )
        self.assertEqual(
            param.value,
            [
                ["red", (1.0, 0.0, 0.0), "#FF0000", (1.0, 0.0, 0.0)],
                ["blue", (0.0, 0.0, 1.0), "#0000FF", (0.0, 0.0, 1.0)],
            ],
        )

    def test_init_with_empty_outer_sequence(self):
        param = ColorSequencesParam(value=[])
        self.assertEqual(param.value, [])

    def test_init_with_empty_inner_sequence(self):
        param = ColorSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_init_with_single_inner_sequence(self):
        param = ColorSequencesParam(value=[["red"]])
        self.assertEqual(param.value, [["red"]])

    def test_init_with_very_large_sequence(self):
        large_sequence = [["red"] * 1000, ["blue"] * 1000]
        param = ColorSequencesParam(value=large_sequence)
        self.assertEqual(param.value, large_sequence)

    def test_init_with_uneven_lengths(self):
        param = ColorSequencesParam(value=[["red"], ["blue", "green"]])
        self.assertEqual(param.value, [["red"], ["blue", "green"]])

    def test_init_with_invalid_hex_strings(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[["#FF000", "#00FF00"], ["#0000FF", "#FFFF00"]])

    def test_init_with_invalid_named_colors(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[["notacolor", "green"], ["blue", "yellow"]])

    def test_init_with_invalid_rgb_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[[(256, 0, 0), (0, 255, 0)], [(0, 0, 255), (255, 255, 0)]]
            )

    def test_init_with_invalid_rgba_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [(255, 0, 0, 1.1), (0, 255, 0, 0.5)],
                    [(0, 0, 255, 0.5), (255, 255, 0, 0.5)],
                ]
            )

    def test_init_with_invalid_rgb_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[[[256, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]]
            )

    def test_init_with_invalid_rgba_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [[255, 0, 0, 1.1], [0, 255, 0, 0.5]],
                    [[0, 0, 255, 0.5], [255, 255, 0, 0.5]],
                ]
            )

    def test_init_with_invalid_float_tuples(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [(1.1, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    [(0.0, 0.0, 1.0), (1.0, 1.0, 0.0)],
                ]
            )

    def test_init_with_invalid_float_lists(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [[1.1, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                ]
            )

    def test_init_with_invalid_numpy_array_shape(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [np.array([255, 0]), np.array([0, 255, 0])],
                    [np.array([0, 0, 255]), np.array([255, 255, 0])],
                ]
            )

    def test_init_with_invalid_pandas_series_shape(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [pd.Series([255, 0]), pd.Series([0, 255, 0])],
                    [pd.Series([0, 0, 255]), pd.Series([255, 255, 0])],
                ]
            )

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=None)

    def test_init_with_none_in_sequence(self):
        param = ColorSequencesParam(
            value=[["red", None, "blue"], ["green", None, "yellow"]]
        )
        self.assertEqual(
            param.value, [["red", None, "blue"], ["green", None, "yellow"]]
        )

    def test_init_with_invalid_type(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[{"key": "value"}]])

    def test_init_with_empty_string(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[""]])

    def test_init_with_empty_list(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[[]]])

    def test_init_with_empty_tuple(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[()]])

    def test_init_with_empty_numpy_array(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[np.array([])]])

    def test_init_with_empty_pandas_series(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[pd.Series([])]])

    def test_init_with_dict_initialization(self):
        param = ColorSequencesParam.model_validate(
            {"value": [["red", "green"], ["blue", "yellow"]]}
        )
        self.assertEqual(param.value, [["red", "green"], ["blue", "yellow"]])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam.model_validate({"value": [["notacolor"]]})

    def test_init_with_very_small_numbers(self):
        param = ColorSequencesParam(
            value=[
                [(1e-100, 0, 0), (0, 1e-100, 0)],
                [(0, 0, 1e-100), (1e-100, 1e-100, 0)],
            ]
        )
        self.assertEqual(
            param.value,
            [
                [(1e-100, 0.0, 0.0), (0.0, 1e-100, 0.0)],
                [(0.0, 0.0, 1e-100), (1e-100, 1e-100, 0.0)],
            ],
        )

    def test_init_with_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [(float("inf"), 0, 0), (0, 255, 0)],
                    [(0, 0, 255), (255, 255, 0)],
                ]
            )

    def test_init_with_negative_inf_values(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(
                value=[
                    [(float("-inf"), 0, 0), (0, 255, 0)],
                    [(0, 0, 255), (255, 255, 0)],
                ]
            )

    def test_init_with_non_sequence_outer(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value="not_a_sequence")

    def test_init_with_non_sequence_inner(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[["red", "green"], "not_a_sequence"])

    def test_init_with_very_deep_nesting(self):
        with self.assertRaises(ValidationError):
            ColorSequencesParam(value=[[[[["red"]]]]])

    def test_init_with_mixed_numeric_types_in_tuples(self):
        param = ColorSequencesParam(
            value=[[(100, 2.0, 30), (40.0, 5, 6.0)], [(7, 80.0, 9), (10.0, 11, 12.0)]]
        )
        self.assertEqual(
            param.value,
            [
                [(0.3922, 0.0078, 0.1176), (0.1569, 0.0196, 0.0235)],
                [(0.0275, 0.3137, 0.0353), (0.0392, 0.0431, 0.0471)],
            ],
        )


class TestEdgeColorParams(TestCase):
    def test_edgecolor_param_valid_values(self):
        EdgeColorParam(value="face")
        EdgeColorParam(value="#FF0000")
        EdgeColorParam(value="red")
        EdgeColorParam(value=(255, 0, 0))
        EdgeColorParam(value=(255, 0, 0, 0.5))

    def test_edgecolor_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            EdgeColorParam(value="invalid")
        with self.assertRaises(ValidationError):
            EdgeColorParam(value=(300, 0, 0))
        with self.assertRaises(ValidationError):
            EdgeColorParam(value="not_face")

    def test_edgecolor_sequence_param_valid_values(self):
        EdgeColorSequenceParam(value=["face", "#FF0000", "red", (255, 0, 0)])
        EdgeColorSequenceParam(value=[(255, 0, 0, 0.5), "face", "#00FF00"])
        EdgeColorSequenceParam(value=[])

    def test_edgecolor_sequence_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            EdgeColorSequenceParam(value=["face", "invalid"])
        with self.assertRaises(ValidationError):
            EdgeColorSequenceParam(value=["not_face", "#FF0000"])

    def test_edgecolor_sequences_param_valid_values(self):
        EdgeColorSequencesParam(value=[["face", "#FF0000"], ["red", (255, 0, 0)]])
        EdgeColorSequencesParam(value=[[(255, 0, 0, 0.5)], ["face", "#00FF00"]])
        EdgeColorSequencesParam(value=[[], []])

    def test_edgecolor_sequences_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            EdgeColorSequencesParam(value=[["face", "invalid"], ["#FF0000"]])
        with self.assertRaises(ValidationError):
            EdgeColorSequencesParam(value=[["not_face", "#FF0000"], ["red"]])

    def test_edgecolor_sequences_param_mixed_valid_invalid(self):
        with self.assertRaises(ValidationError):
            EdgeColorSequencesParam(value=[["face", "#FF0000"], ["invalid", "red"]])


class TestMarkerParam(TestCase):
    def test_init_with_valid_string_markers(self):
        MarkerParam(value="o")
        MarkerParam(value=".")
        MarkerParam(value="s")
        MarkerParam(value="^")
        MarkerParam(value="*")
        MarkerParam(value="+")
        MarkerParam(value="x")
        MarkerParam(value="D")
        MarkerParam(value="d")
        MarkerParam(value="|")
        MarkerParam(value="_")
        MarkerParam(value="h")
        MarkerParam(value="H")
        MarkerParam(value="v")
        MarkerParam(value="<")
        MarkerParam(value=">")
        MarkerParam(value="1")
        MarkerParam(value="2")
        MarkerParam(value="3")
        MarkerParam(value="4")
        MarkerParam(value="p")
        MarkerParam(value="P")
        MarkerParam(value=",")
        MarkerParam(value="")
        MarkerParam(value=" ")
        MarkerParam(value="none")
        MarkerParam(value="None")

    def test_init_with_valid_numeric_markers(self):
        for i in range(12):
            MarkerParam(value=i)

    def test_init_with_valid_dict_markers(self):
        MarkerParam(value={"marker": "o"})
        MarkerParam(value={"marker": "o", "fillstyle": "full"})
        MarkerParam(value={"marker": "o", "fillstyle": "none"})
        MarkerParam(value={"marker": "o", "fillstyle": "left"})
        MarkerParam(value={"marker": "o", "fillstyle": "right"})
        MarkerParam(value={"marker": "o", "fillstyle": "bottom"})
        MarkerParam(value={"marker": "o", "fillstyle": "top"})
        MarkerParam(value={"marker": "o", "capstyle": "butt"})
        MarkerParam(value={"marker": "o", "capstyle": "round"})
        MarkerParam(value={"marker": "o", "capstyle": "projecting"})
        MarkerParam(value={"marker": "o", "joinstyle": "miter"})
        MarkerParam(value={"marker": "o", "joinstyle": "round"})
        MarkerParam(value={"marker": "o", "joinstyle": "bevel"})

    def test_init_with_valid_markerstyle(self):
        from matplotlib.markers import MarkerStyle

        MarkerParam(value=MarkerStyle("o"))
        MarkerParam(value=MarkerStyle("s"))
        MarkerParam(value=MarkerStyle("^"))
        MarkerParam(value=MarkerStyle("*"))

    def test_init_with_none(self):
        MarkerParam(value=None)

    def test_init_with_invalid_string_markers(self):
        with self.assertRaises(ValidationError):
            MarkerParam(value="invalid")
        with self.assertRaises(ValidationError):
            MarkerParam(value="invalid_marker")

    def test_init_with_invalid_numeric_markers(self):
        with self.assertRaises(ValidationError):
            MarkerParam(value=-1)
        with self.assertRaises(ValidationError):
            MarkerParam(value=12)
        with self.assertRaises(ValidationError):
            MarkerParam(value=100)
        with self.assertRaises(ValidationError):
            MarkerParam(value=1.5)

    def test_init_with_invalid_dict_markers(self):
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": {"marker": "invalid"}})
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate(
                {"value": {"marker": "o", "fillstyle": "invalid"}}
            )
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate(
                {"value": {"marker": "o", "capstyle": "invalid"}}
            )
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate(
                {"value": {"marker": "o", "joinstyle": "invalid"}}
            )
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate(
                {"value": {"marker": "o", "transform": "invalid"}}
            )
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": {"marker": "o", "transform": 123}})

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            MarkerParam(value=[])
        with self.assertRaises(ValidationError):
            MarkerParam(value=())
        with self.assertRaises(ValidationError):
            MarkerParam(value=1.5)
        with self.assertRaises(ValidationError):
            MarkerParam(value=object())

    def test_init_with_dict_initialization(self):
        MarkerParam.model_validate({"value": "o"})
        MarkerParam.model_validate({"value": 0})
        MarkerParam.model_validate({"value": {"marker": "o"}})
        MarkerParam.model_validate({"value": None})

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": "invalid"})
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": -1})
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": {"marker": "invalid"}})
        with self.assertRaises(ValidationError):
            MarkerParam.model_validate({"value": []})


class TestMarkerSequenceParam(TestCase):
    def test_init_with_valid_sequences(self):
        MarkerSequenceParam(value=["o", "s", "D"])
        MarkerSequenceParam(value=[0, 1, 2])
        MarkerSequenceParam(value=[{"marker": "o"}, {"marker": "s"}])
        MarkerSequenceParam(value=[None, "o", None])

    def test_init_with_invalid_sequences(self):
        with self.assertRaises(ValidationError):
            MarkerSequenceParam(value=["invalid", "o"])
        with self.assertRaises(ValidationError):
            MarkerSequenceParam(value=[-1, 0])
        with self.assertRaises(ValidationError):
            MarkerSequenceParam(value=[{"marker": "invalid"}, "o"])

    def test_init_with_empty_sequence(self):
        MarkerSequenceParam(value=[])
        MarkerSequenceParam(value=())

    def test_init_with_single_marker(self):
        MarkerSequenceParam(value=["o"])
        MarkerSequenceParam(value=[0])
        MarkerSequenceParam(value=[{"marker": "o"}])

    def test_init_with_dict_initialization(self):
        MarkerSequenceParam.model_validate({"value": ["o", "s"]})
        MarkerSequenceParam.model_validate({"value": [0, 1]})
        MarkerSequenceParam.model_validate({"value": [{"marker": "o"}]})

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            MarkerSequenceParam.model_validate({"value": ["invalid"]})
        with self.assertRaises(ValidationError):
            MarkerSequenceParam.model_validate({"value": [{"marker": "invalid"}]})


class TestMarkerSequencesParam(TestCase):
    def test_init_with_valid_sequences(self):
        MarkerSequencesParam(value=[["o", "s"], ["D", "p"]])
        MarkerSequencesParam(value=[[0, 1], [2, 3]])
        MarkerSequencesParam(value=[[{"marker": "o"}], [{"marker": "s"}]])
        MarkerSequencesParam(value=[[None, "o"], ["s", None]])

    def test_init_with_invalid_sequences(self):
        with self.assertRaises(ValidationError):
            MarkerSequencesParam(value=[["invalid", "o"], ["s", "D"]])
        with self.assertRaises(ValidationError):
            MarkerSequencesParam(value=[[-1, 0], [1, 2]])
        with self.assertRaises(ValidationError):
            MarkerSequencesParam(value=[[{"marker": "invalid"}], [{"marker": "o"}]])

    def test_init_with_empty_sequences(self):
        MarkerSequencesParam(value=[])
        MarkerSequencesParam(value=[[]])
        MarkerSequencesParam(value=[[], []])

    def test_init_with_single_sequence(self):
        MarkerSequencesParam(value=[["o"]])
        MarkerSequencesParam(value=[[0]])
        MarkerSequencesParam(value=[[{"marker": "o"}]])

    def test_init_with_dict_initialization(self):
        MarkerSequencesParam.model_validate({"value": [["o", "s"], ["D", "p"]]})
        MarkerSequencesParam.model_validate({"value": [[0, 1], [2, 3]]})
        MarkerSequencesParam.model_validate({"value": [[{"marker": "o"}]]})

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            MarkerSequencesParam.model_validate({"value": [["invalid"]]})
        with self.assertRaises(ValidationError):
            MarkerSequencesParam.model_validate({"value": [[-1]]})
        with self.assertRaises(ValidationError):
            MarkerSequencesParam.model_validate({"value": [[{"marker": "invalid"}]]})


class TestLiteralParam(TestCase):
    def test_init_with_valid_literals(self):
        LiteralParam.model_validate(
            {"value": "option1", "options": ["option1", "option2"]}
        )
        LiteralParam.model_validate(
            {"value": "option2", "options": ["option1", "option2"]}
        )

    def test_init_with_invalid_literals(self):
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate(
                {"value": "invalid", "options": ["option1", "option2"]}
            )
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate({"value": "option1", "options": []})

    def test_init_with_missing_fields(self):
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate({"value": "option1"})
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate({"options": ["option1"]})

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate({"value": 1, "options": ["option1"]})
        with self.assertRaises(ValidationError):
            LiteralParam.model_validate({"value": "option1", "options": "option1"})


class TestLiteralSequenceParam(TestCase):
    def test_init_with_valid_sequences(self):
        LiteralSequenceParam(
            value=["option1", "option2"], options=["option1", "option2"]
        )

    def test_init_with_invalid_sequences(self):
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate(
                {"value": ["invalid"], "options": ["option1", "option2"]}
            )
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate({"value": ["option1"], "options": []})

    def test_init_with_missing_fields(self):
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate({"value": ["option1"]})
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate({"options": ["option1"]})

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate({"value": [1], "options": ["option1"]})
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate(
                {"value": ["option1"], "options": "option1"}
            )


class TestLiteralSequencesParam(TestCase):
    def test_init_with_valid_sequences(self):
        LiteralSequencesParam.model_validate(
            {"value": [["option1"], ["option2"]], "options": ["option1", "option2"]}
        )
        LiteralSequencesParam.model_validate(
            {"value": [["option1", "option2"]], "options": ["option1", "option2"]}
        )

    def test_init_with_invalid_sequences(self):
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [["invalid"]], "options": ["option1", "option2"]}
            )
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [["option1"]], "options": []}
            )

    def test_init_with_missing_fields(self):
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate({"value": [["option1"]]})
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate({"options": ["option1"]})

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [[1]], "options": ["option1"]}
            )
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [["option1"]], "options": "option1"}
            )


class TestNormalizationParam(TestCase):
    def test_init_with_valid_literals(self):
        NormalizationParam(value="linear")
        NormalizationParam(value="log")
        NormalizationParam(value="symlog")

    def test_init_with_valid_normalize_instance(self):
        from matplotlib.colors import Normalize

        NormalizationParam(value=Normalize())

    def test_init_with_invalid_literals(self):
        with self.assertRaises(ValidationError):
            NormalizationParam(value="invalid")
        with self.assertRaises(ValidationError):
            NormalizationParam(value=123)


if __name__ == "__main__":
    unittest.main()
