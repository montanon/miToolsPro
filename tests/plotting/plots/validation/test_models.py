import unittest
from typing import Optional
from unittest import TestCase

import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.validation.models import (
    BinsParam,
    BinsSequenceParam,
    BinsSequencesParam,
    BoolParam,
    BoolSequenceParam,
    BoolSequencesParam,
    ColormapParam,
    ColormapSequenceParam,
    ColormapSequencesParam,
    ColorParam,
    ColorSequenceParam,
    ColorSequencesParam,
    DataSequenceParam,
    DataSequencesParam,
    DictParam,
    DictSequenceParam,
    DictSequencesParam,
    EdgeColorParam,
    EdgeColorSequenceParam,
    EdgeColorSequencesParam,
    FloatParam,
    IntParam,
    LiteralParam,
    LiteralSequenceParam,
    LiteralSequencesParam,
    MarkerParam,
    MarkerSequenceParam,
    MarkerSequencesParam,
    NormalizationParam,
    NormalizationSequenceParam,
    NormalizationSequencesParam,
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
    NumericTupleParam,
    NumericTupleSequenceParam,
    NumericTupleSequencesParam,
    NumStrParam,
    Param,
    RangeParam,
    RangeSequenceParam,
    RangeSequencesParam,
    SequenceParam,
    SequencesParam,
    SpineParam,
    SpinesParam,
    StrParam,
    StrSequenceParam,
    StrSequencesParam,
    TransformParam,
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

    def test_transform_param_valid_values(self):
        param = TransformParam(value=Transform())
        self.assertEqual(param.value._shorthand_name, Transform()._shorthand_name)
        param = TransformParam(value=Transform(shorthand_name="log"))
        self.assertEqual(
            param.value._shorthand_name, Transform(shorthand_name="log")._shorthand_name
        )

    def test_transform_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            TransformParam(value="not_a_transform")
        with self.assertRaises(ValidationError):
            TransformParam(value=None)
        with self.assertRaises(ValidationError):
            TransformParam(value=[])

    def test_float_param_valid_values(self):
        param = FloatParam(value=3.14)
        self.assertEqual(param.value, 3.14)
        param = FloatParam(value=0.0)
        self.assertEqual(param.value, 0.0)
        param = FloatParam(value=-1.0)
        self.assertEqual(param.value, -1.0)

    def test_float_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            FloatParam(value="3.14")
        with self.assertRaises(ValidationError):
            FloatParam(value=None)
        with self.assertRaises(ValidationError):
            FloatParam(value=[])

    def test_int_param_valid_values(self):
        param = IntParam(value=42)
        self.assertEqual(param.value, 42)
        param = IntParam(value=0)
        self.assertEqual(param.value, 0)
        param = IntParam(value=-1)
        self.assertEqual(param.value, -1)

    def test_int_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            IntParam(value="42")
        with self.assertRaises(ValidationError):
            IntParam(value=None)
        with self.assertRaises(ValidationError):
            IntParam(value=[])

    def test_num_str_param_valid_values(self):
        param = NumStrParam(value=42)
        self.assertEqual(param.value, 42)
        param = NumStrParam(value="42")
        self.assertEqual(param.value, "42")

    def test_num_str_param_invalid_values(self):
        with self.assertRaises(ValidationError):
            NumStrParam(value=3.14)

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

    def test_transform_param_with_dict_initialization(self):
        param = TransformParam.model_validate({"value": Transform()})
        self.assertEqual(param.value._shorthand_name, Transform()._shorthand_name)

    def test_float_param_with_dict_initialization(self):
        param = FloatParam.model_validate({"value": 3.14})
        self.assertEqual(param.value, 3.14)

    def test_int_param_with_dict_initialization(self):
        param = IntParam.model_validate({"value": 42})
        self.assertEqual(param.value, 42)

    def test_num_str_param_with_dict_initialization(self):
        param = NumStrParam.model_validate({"value": "42"})
        self.assertEqual(param.value, "42")


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

    def test_init_with_valid_sizes_single_int(self):
        param = SequenceParam[int](value=[1, 2, 3], sizes=3)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [3])

    def test_init_with_valid_sizes_sequence(self):
        param = SequenceParam[int](value=[1, 2, 3], sizes=[2, 3, 4])
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [2, 3, 4])

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int](value=[1, 2, 3], sizes=4)
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_invalid_size_sequence(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int](value=[1, 2, 3], sizes=[4, 5, 6])
        self.assertIn(
            "Expected Sequence of sizes: [4, 5, 6], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_none_sizes(self):
        param = SequenceParam[int](value=[1, 2, 3], sizes=None)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertIsNone(param.sizes)

    def test_init_with_empty_sequence_and_size(self):
        param = SequenceParam[int](value=[], sizes=0)
        self.assertEqual(param.value, [])
        self.assertEqual(param.sizes, [0])

    def test_init_with_dict_initialization_and_sizes(self):
        param = SequenceParam[int].model_validate({"value": [1, 2, 3], "sizes": 3})
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [3])

    def test_init_with_multiple_valid_sizes(self):
        param = SequenceParam[int](value=[1, 2], sizes=[2, 3])
        self.assertEqual(param.value, [1, 2])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_structured_true_and_valid_size(self):
        param = SequenceParam[int](value=[1, 2, 3], sizes=3, structured=True)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int](value=[1, 2], sizes=3, structured=True)
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int](value=[1, 2, 3], sizes=[2, 3, 4], structured=True)
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_sequence_sizes(self):
        param = SequenceParam[int](value=[1, 2, 3], sizes=[2, 3, 4], structured=False)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [2, 3, 4])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = SequenceParam[int](value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_structured_true_and_no_sizes(self):
        param = SequenceParam[int](value=[1, 2, 3], structured=True)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertIsNone(param.sizes)
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = SequenceParam[int].model_validate(
            {"value": [1, 2, 3], "sizes": 3, "structured": True}
        )
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int].model_validate(
                {"value": [1, 2], "sizes": 3, "structured": True}
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int].model_validate(
                {"value": [1, 2, 3], "sizes": [2, 3, 4], "structured": True}
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )


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

    def test_init_with_valid_sizes_single_int(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sizes=2)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])

    def test_init_with_valid_sizes_sequence(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sizes=[2, 3])
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2, 3])

    def test_init_with_valid_sub_sizes_single_int(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sub_sizes=2)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sub_sizes, [2])

    def test_init_with_valid_sub_sizes_sequence(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sub_sizes=[2, 3])
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sub_sizes, [2, 3])

    def test_init_with_valid_sizes_and_sub_sizes(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sizes=2, sub_sizes=2)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2])

    def test_init_with_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int](value=[[1, 2], [3, 4]], sizes=3)
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2",
            str(context.exception),
        )

    def test_init_with_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int](value=[[1, 2, 3], [4, 5, 6]], sub_sizes=2)
        self.assertIn(
            "Expected sub Sequences of sizes: [2] got size: 3 at index=0",
            str(context.exception),
        )

    def test_init_with_mixed_valid_sub_sizes(self):
        param = SequencesParam[int](value=[[1], [1, 2], [1, 2, 3]], sub_sizes=[1, 2, 3])
        self.assertEqual(param.value, [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(param.sub_sizes, [1, 2, 3])

    def test_init_with_none_sizes(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], sizes=None, sub_sizes=None)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)

    def test_init_with_empty_sequences_and_sizes(self):
        param = SequencesParam[int](value=[], sizes=0)
        self.assertEqual(param.value, [])
        self.assertEqual(param.sizes, [0])

    def test_init_with_empty_sub_sequences_and_sub_sizes(self):
        param = SequencesParam[int](value=[[], []], sub_sizes=0)
        self.assertEqual(param.value, [[], []])
        self.assertEqual(param.sub_sizes, [0])

    def test_init_with_dict_initialization_and_sizes(self):
        param = SequencesParam[int].model_validate(
            {"value": [[1, 2], [3, 4]], "sizes": 2, "sub_sizes": 2}
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = SequencesParam[int](
            value=[[1, 2], [3, 4]], sizes=2, sub_sizes=[2, 2], structured=True
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = SequencesParam[int](
            value=[[1], [1, 2], [1, 2, 3]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(param.value, [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int](
                value=[[1, 2], [3, 4]], sizes=3, sub_sizes=2, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int](
                value=[[1, 2, 3], [4, 5]], sizes=2, sub_sizes=[3, 3], structured=True
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int](
                value=[[1], [1, 2]], sizes=2, sub_sizes=[1, 2, 3], structured=True
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = SequencesParam[int](
            value=[[1, 2], [3, 4]], sizes=[2, 3], sub_sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]])
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_structured_true_and_no_sizes(self):
        param = SequencesParam[int](value=[[1, 2], [3, 4]], structured=True)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = SequencesParam[int].model_validate(
            {
                "value": [[1, 2], [3, 4]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            SequencesParam[int].model_validate(
                {
                    "value": [[1, 2], [3, 4]],
                    "sizes": 3,
                    "sub_sizes": 2,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            SequenceParam[int].model_validate(
                {"value": [1, 2, 3], "sizes": [2, 3, 4], "structured": True}
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )


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
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_valid_tuple_and_size(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=3)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [3])

    def test_init_with_valid_tuple_and_sizes(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [2, 3, 4])

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

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=None)

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=2)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, 4])

    def test_init_with_valid_size_range(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [2, 3, 4])

    def test_init_with_single_size(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=3)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [3])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[])

    def test_init_with_none_size(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=None)
        self.assertEqual(param.value, (1, 2, 3))
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes="not_a_size")
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes={"key": "value"})

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=-1)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[-1, -2])

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(), tuple_sizes=0)

    def test_init_with_very_large_size(self):
        param = NumericTupleParam(value=tuple(range(1000)), tuple_sizes=1000)
        self.assertEqual(len(param.value), 1000)
        self.assertEqual(param.tuple_sizes, [1000])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=3.0)
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=np.array([2, 3, 4]))

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=pd.Series([2, 3, 4]))

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, None, 4])

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, -1, 4])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, 3, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [2, 3, 3, 4])

    def test_init_with_single_size_matching(self):
        param = NumericTupleParam(value=(1, 2), tuple_sizes=2)
        self.assertEqual(param.value, (1, 2))
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleParam(value=(1, 2, 3), tuple_sizes=[2, 3, 4])
        self.assertEqual(param.value, (1, 2, 3))
        self.assertEqual(param.tuple_sizes, [2, 3, 4])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleParam(value=(1, 2, 3), tuple_sizes=[4, 5, 6])

    def test_init_with_single_element_and_size(self):
        param = NumericTupleParam(value=(1,), tuple_sizes=1)
        self.assertEqual(param.value, (1,))
        self.assertEqual(param.tuple_sizes, [1])


class TestNumericTupleSequenceParam(TestCase):
    def test_init_with_valid_sequence(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_valid_sequence_and_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_valid_sequence_and_sizes(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], tuple_sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.tuple_sizes, [2, 3])

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

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), None])

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=3)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[3, 4])

    def test_init_with_valid_size_range(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], tuple_sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.tuple_sizes, [2, 3])

    def test_init_with_single_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[])

    def test_init_with_none_size(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=None)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes="not_a_size")
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(
                value=[(1, 2), (3, 4)], tuple_sizes={"key": "value"}
            )

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=-1)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[-1, -2])

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(), ()], tuple_sizes=0)

    def test_init_with_very_large_size(self):
        large_sequence = [(i,) for i in range(1000)]
        param = NumericTupleSequenceParam(value=large_sequence, tuple_sizes=1)
        self.assertEqual(len(param.value), 1000)
        self.assertEqual(param.tuple_sizes, [1])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=2.0)
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(
                value=[(1, 2), (3, 4)], tuple_sizes=np.array([2, 3])
            )

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(
                value=[(1, 2), (3, 4)], tuple_sizes=pd.Series([2, 3])
            )

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[2, None, 3])

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[2, -1, 3])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[2, 2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.tuple_sizes, [2, 2, 3])

    def test_init_with_single_size_matching(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=2)
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4, 5)], tuple_sizes=[2, 3])
        self.assertEqual(param.value, [(1, 2), (3, 4, 5)])
        self.assertEqual(param.tuple_sizes, [2, 3])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam(value=[(1, 2), (3, 4)], tuple_sizes=[3, 4])

    def test_init_with_empty_tuple_sequence(self):
        param = NumericTupleSequenceParam(value=[(), ()])
        self.assertEqual(param.value, [(), ()])

    def test_init_with_single_element_sequence_and_size(self):
        param = NumericTupleSequenceParam(value=[(1,)], tuple_sizes=1)
        self.assertEqual(param.value, [(1,)])
        self.assertEqual(param.tuple_sizes, [1])

    def test_init_with_uneven_tuple_sizes(self):
        param = NumericTupleSequenceParam(
            value=[(1,), (2, 3), (4, 5, 6)], tuple_sizes=[1, 2, 3]
        )
        self.assertEqual(param.value, [(1,), (2, 3), (4, 5, 6)])
        self.assertEqual(param.tuple_sizes, [1, 2, 3])

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
            {"value": [(1, 2), (3, 4)], "tuple_sizes": 2}
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequenceParam.model_validate(
                {"value": [(1, 2), (3, 4)], "tuple_sizes": "invalid"}
            )

    def test_init_with_structured_true_and_valid_sizes(self):
        param = NumericTupleSequenceParam(
            value=[(1, 2), (3, 4)], sizes=2, tuple_sizes=2, structured=True
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.tuple_sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = NumericTupleSequenceParam(
            value=[(1,), (1, 2), (1, 2, 3)],
            sizes=3,
            tuple_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(param.value, [(1,), (1, 2), (1, 2, 3)])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.tuple_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequenceParam(
                value=[(1, 2), (3, 4)], sizes=3, tuple_sizes=2, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_tuple_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequenceParam(
                value=[(1, 2, 3), (4, 5)], sizes=2, tuple_sizes=[3, 3], structured=True
            )
        self.assertIn(
            "Invalid tuple length 2 at index 1. Allowed sizes: [3, 3]",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_tuple_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequenceParam(
                value=[(1,), (2, 3)], sizes=2, tuple_sizes=[1, 2, 3], structured=True
            )
        self.assertIn(
            "Validation of structured Sequence requires a single tuple size, got tuple_sizes=[1, 2, 3]",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = NumericTupleSequenceParam(
            value=[(1, 2), (3, 4)], sizes=[2, 3], tuple_sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.tuple_sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = NumericTupleSequenceParam(value=[(1, 2), (3, 4)])
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.tuple_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = NumericTupleSequenceParam.model_validate(
            {
                "value": [(1, 2), (3, 4)],
                "sizes": 2,
                "tuple_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [(1, 2), (3, 4)])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.tuple_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequenceParam.model_validate(
                {
                    "value": [(1, 2), (3, 4)],
                    "sizes": 3,
                    "tuple_sizes": 2,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_tuple_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequenceParam.model_validate(
                {
                    "value": [(1, 2, 3), (4, 5)],
                    "sizes": 2,
                    "tuple_sizes": [3, 3],
                    "structured": True,
                }
            )
        self.assertIn(
            "Invalid tuple length 2 at index 1. Allowed sizes: [3, 3]",
            str(context.exception),
        )


class TestNumericTupleSequencesParam(TestCase):
    def test_init_with_valid_sequences(self):
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_valid_sequences_and_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_valid_sequences_and_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], tuple_sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.tuple_sizes, [2, 3])

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

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[(1, 2), None], [(3, 4), (5, 6)]])

    def test_init_with_invalid_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=3
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[3, 4]
            )

    def test_init_with_valid_size_range(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], tuple_sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.tuple_sizes, [2, 3])

    def test_init_with_single_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_empty_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[]
            )

    def test_init_with_none_size(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=None
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertIsNone(param.tuple_sizes)

    def test_init_with_invalid_size_type(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes="not_a_size"
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes={"key": "value"}
            )

    def test_init_with_negative_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=-1
            )
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[-1, -2]
            )

    def test_init_with_zero_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[(), ()], [(), ()]], tuple_sizes=0)

    def test_init_with_very_large_size(self):
        large_sequence = [[(i,) for i in range(1000)], [(i,) for i in range(1000)]]
        param = NumericTupleSequencesParam(value=large_sequence, tuple_sizes=1)
        self.assertEqual(len(param.value), 2)
        self.assertEqual(param.tuple_sizes, [1])

    def test_init_with_float_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], tuple_sizes=2.0)
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], tuple_sizes=[2.0, 3.0])

    def test_init_with_numpy_array_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=np.array([2, 3])
            )

    def test_init_with_pandas_series_as_size(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[(1, 2), (3, 4)], tuple_sizes=pd.Series([2, 3])
            )

    def test_init_with_none_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[2, None, 3]
            )

    def test_init_with_negative_in_size_list(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[(1, 2), (3, 4)], tuple_sizes=[2, -1, 3])

    def test_init_with_duplicate_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[2, 2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.tuple_sizes, [2, 2, 3])

    def test_init_with_single_size_matching(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=2
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_multiple_sizes_matching(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]], tuple_sizes=[2, 3]
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6, 7), (8, 9, 10)]])
        self.assertEqual(param.tuple_sizes, [2, 3])

    def test_init_with_size_not_matching(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]], tuple_sizes=[3, 4]
            )

    def test_init_with_empty_tuple_sequences(self):
        param = NumericTupleSequencesParam(value=[[(), ()], [(), ()]])
        self.assertEqual(param.value, [[(), ()], [(), ()]])

    def test_init_with_single_element_sequences_and_size(self):
        param = NumericTupleSequencesParam(value=[[(1,)], [(2,)]], tuple_sizes=1)
        self.assertEqual(param.value, [[(1,)], [(2,)]])
        self.assertEqual(param.tuple_sizes, [1])

    def test_init_with_uneven_tuple_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1,), (2, 3), (4, 5, 6)]], tuple_sizes=[1, 2, 3]
        )
        self.assertEqual(param.value, [[(1,), (2, 3), (4, 5, 6)]])
        self.assertEqual(param.tuple_sizes, [1, 2, 3])

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
        param = NumericTupleSequencesParam(value=[[(1, 2), (3, 4, 5), (6,)]])
        self.assertEqual(param.value, [[(1, 2), (3, 4, 5), (6,)]])

    def test_init_with_nested_sequences(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam(value=[[[[[(1, 2)]]]]])

    def test_init_with_dict_initialization(self):
        param = NumericTupleSequencesParam.model_validate(
            {"value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]], "tuple_sizes": 2}
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.tuple_sizes, [2])

    def test_init_with_invalid_dict_initialization(self):
        with self.assertRaises(ValidationError):
            NumericTupleSequencesParam.model_validate(
                {
                    "value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
                    "tuple_sizes": "invalid",
                }
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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
            sizes=2,
            sub_sizes=[2, 2],
            tuple_sizes=2,
            structured=True,
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertEqual(param.tuple_sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1,)], [(1, 2), (1, 2)], [(1, 2, 3), (1, 2, 3), (1, 2, 3)]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            tuple_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(
            param.value, [[(1,)], [(1, 2), (1, 2)], [(1, 2, 3), (1, 2, 3), (1, 2, 3)]]
        )
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertEqual(param.tuple_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam(
                value=[[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
                sizes=3,
                sub_sizes=2,
                tuple_sizes=2,
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam(
                value=[[(1, 2, 3)], [(4, 5), (6, 7)]],
                sizes=2,
                sub_sizes=[1, 3],
                tuple_sizes=3,
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [1, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_tuple_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam(
                value=[[(1, 2, 3)], [(4, 5)]],
                sizes=2,
                sub_sizes=[1, 1],
                tuple_sizes=[3, 3],
                structured=True,
            )
        self.assertIn(
            "Invalid tuple length 2 at index [1, 0]. Allowed sizes: [3, 3]",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam(
                value=[[(1,)], [(2, 3)]],
                sizes=2,
                sub_sizes=[1, 2, 3],
                tuple_sizes=[1, 2],
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = NumericTupleSequencesParam(
            value=[[(1, 2)], [(3, 4), (5, 6)]],
            sizes=[2, 3],
            sub_sizes=[1, 2],
            tuple_sizes=[2, 2],
            structured=False,
        )
        self.assertEqual(param.value, [[(1, 2)], [(3, 4), (5, 6)]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [1, 2])
        self.assertEqual(param.tuple_sizes, [2, 2])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = NumericTupleSequencesParam(value=[[(1, 2)], [(3, 4), (5, 6)]])
        self.assertEqual(param.value, [[(1, 2)], [(3, 4), (5, 6)]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertIsNone(param.tuple_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = NumericTupleSequencesParam.model_validate(
            {
                "value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "tuple_sizes": 2,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertEqual(param.tuple_sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam.model_validate(
                {
                    "value": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
                    "sizes": 3,
                    "sub_sizes": 2,
                    "tuple_sizes": 2,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam.model_validate(
                {
                    "value": [[(1, 2, 3)], [(4, 5), (6, 7)]],
                    "sizes": 2,
                    "sub_sizes": [1, 3],
                    "tuple_sizes": 3,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [1, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_tuple_size(self):
        with self.assertRaises(ValidationError) as context:
            NumericTupleSequencesParam.model_validate(
                {
                    "value": [[(1, 2, 3)], [(4, 5)]],
                    "sizes": 2,
                    "sub_sizes": [1, 1],
                    "tuple_sizes": [3, 3],
                    "structured": True,
                }
            )
        self.assertIn(
            "Invalid tuple length 2 at index [1, 0]. Allowed sizes: [3, 3]",
            str(context.exception),
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
        # Color can be a number representing a color scale
        ColorParam(value=256)
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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = ColorSequenceParam(
            value=["red", "#00FF00", (0, 0, 1)], sizes=3, structured=True
        )
        self.assertEqual(param.value, ["red", "#00FF00", (0.0, 0.0, 1.0)])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequenceParam(
                value=["red", "#00FF00", (0, 0, 1)], sizes=4, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequenceParam(
                value=["red", "#00FF00", (0, 0, 1)], sizes=[2, 3, 4], structured=True
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = ColorSequenceParam(
            value=["red", "#00FF00"], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, ["red", "#00FF00"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = ColorSequenceParam(value=["red", "#00FF00"])
        self.assertEqual(param.value, ["red", "#00FF00"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = ColorSequenceParam.model_validate(
            {
                "value": ["red", "#00FF00", (0, 0, 1)],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, ["red", "#00FF00", (0.0, 0.0, 1.0)])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequenceParam.model_validate(
                {
                    "value": ["red", "#00FF00", (0, 0, 1)],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequenceParam.model_validate(
                {
                    "value": ["red", "#00FF00", (0, 0, 1)],
                    "sizes": [2, 3, 4],
                    "structured": True,
                }
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_color(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequenceParam(
                value=["red", "invalid_color", (0, 0, 1)], sizes=3, structured=True
            )
        self.assertIn(
            "Invalid color format: 'invalid_color' at index 1",
            str(context.exception),
        )


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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = ColorSequencesParam(
            value=[["red", "#00FF00"], ["blue", "#0000FF"]], sizes=2, structured=True
        )
        self.assertEqual(param.value, [["red", "#00FF00"], ["blue", "#0000FF"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = ColorSequencesParam(
            value=[["red"], ["blue", "green"], ["yellow", "#FF0000", "#00FF00"]],
            sizes=3,
            structured=True,
        )
        self.assertEqual(
            param.value, [["red"], ["blue", "green"], ["yellow", "#FF0000", "#00FF00"]]
        )
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequencesParam(
                value=[["red", "#00FF00"], ["blue", "#0000FF"]],
                sizes=3,
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes_list(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequencesParam(
                value=[["red", "#00FF00"], ["blue", "#0000FF"]],
                sizes=[2, 3],
                structured=True,
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_color(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequencesParam(
                value=[["red", "#00FF00"], ["blue", "invalid_color"]],
                sizes=2,
                structured=True,
            )
        self.assertIn(
            "Invalid color format: 'invalid_color' at index [1, 1]",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = ColorSequencesParam(
            value=[["red"], ["blue", "green"]], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, [["red"], ["blue", "green"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = ColorSequencesParam(value=[["red"], ["blue", "green"]])
        self.assertEqual(param.value, [["red"], ["blue", "green"]])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = ColorSequencesParam.model_validate(
            {
                "value": [["red", "#00FF00"], ["blue", "#0000FF"]],
                "sizes": 2,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["red", "#00FF00"], ["blue", "#0000FF"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequencesParam.model_validate(
                {
                    "value": [["red", "#00FF00"], ["blue", "#0000FF"]],
                    "sizes": 3,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_color(self):
        with self.assertRaises(ValidationError) as context:
            ColorSequencesParam.model_validate(
                {
                    "value": [["red", "#00FF00"], ["blue", "invalid_color"]],
                    "sizes": 2,
                    "structured": True,
                }
            )
        self.assertIn(
            "Invalid color format: 'invalid_color' at index [1, 1]",
            str(context.exception),
        )

    def test_init_with_mixed_color_formats_structured(self):
        param = ColorSequencesParam(
            value=[["red", "#00FF00", (0, 0, 1)], ["#FF0000", (0, 1, 0), "blue"]],
            sizes=2,
            structured=True,
        )
        self.assertEqual(
            param.value,
            [["red", "#00FF00", (0.0, 0.0, 1.0)], ["#FF0000", (0.0, 1.0, 0.0), "blue"]],
        )
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)


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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = EdgeColorSequenceParam(
            value=["face", "#00FF00", (0, 0, 1)], sizes=3, structured=True
        )
        self.assertEqual(param.value, ["face", "#00FF00", (0.0, 0.0, 1.0)])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequenceParam(
                value=["face", "#00FF00", (0, 0, 1)], sizes=4, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequenceParam(
                value=["face", "#00FF00", (0, 0, 1)], sizes=[2, 3, 4], structured=True
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = EdgeColorSequenceParam(
            value=["face", "#00FF00"], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, ["face", "#00FF00"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = EdgeColorSequenceParam(value=["face", "#00FF00"])
        self.assertEqual(param.value, ["face", "#00FF00"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = EdgeColorSequenceParam.model_validate(
            {
                "value": ["face", "#00FF00", (0, 0, 1)],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, ["face", "#00FF00", (0.0, 0.0, 1.0)])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequenceParam.model_validate(
                {
                    "value": ["face", "#00FF00", (0, 0, 1)],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_sequence_with_structured_true_and_invalid_color(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequenceParam(
                value=["face", "invalid_color", (0, 0, 1)], sizes=3, structured=True
            )
        self.assertIn(
            "Invalid color format: 'invalid_color' at index 1",
            str(context.exception),
        )

    def test_sequences_with_structured_true_and_valid_sizes(self):
        param = EdgeColorSequencesParam(
            value=[["face", "#00FF00"], ["blue", "#0000FF"]], sizes=2, structured=True
        )
        self.assertEqual(param.value, [["face", "#00FF00"], ["blue", "#0000FF"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_sequences_with_structured_true_and_valid_sequence_sizes(self):
        param = EdgeColorSequencesParam(
            value=[["face"], ["blue", "face"], ["yellow", "#FF0000", "#00FF00"]],
            sizes=3,
            structured=True,
        )
        self.assertEqual(
            param.value, [["face"], ["blue", "face"], ["yellow", "#FF0000", "#00FF00"]]
        )
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_sequences_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequencesParam(
                value=[["face", "#00FF00"], ["blue", "#0000FF"]],
                sizes=3,
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_sequences_with_structured_true_and_sequence_sizes_list(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequencesParam(
                value=[["face", "#00FF00"], ["blue", "#0000FF"]],
                sizes=[2, 3],
                structured=True,
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_sequences_with_structured_true_and_invalid_color(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequencesParam(
                value=[["face", "#00FF00"], ["blue", "invalid_color"]],
                sizes=2,
                structured=True,
            )
        self.assertIn(
            "Invalid color format: 'invalid_color' at index [1, 1]",
            str(context.exception),
        )

    def test_sequences_with_structured_false_and_valid_sizes(self):
        param = EdgeColorSequencesParam(
            value=[["face"], ["blue", "face"]], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, [["face"], ["blue", "face"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_sequences_with_structured_default_value(self):
        param = EdgeColorSequencesParam(value=[["face"], ["blue", "face"]])
        self.assertEqual(param.value, [["face"], ["blue", "face"]])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_sequences_with_dict_initialization_and_structured(self):
        param = EdgeColorSequencesParam.model_validate(
            {
                "value": [["face", "#00FF00"], ["blue", "#0000FF"]],
                "sizes": 2,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["face", "#00FF00"], ["blue", "#0000FF"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_sequences_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            EdgeColorSequencesParam.model_validate(
                {
                    "value": [["face", "#00FF00"], ["blue", "#0000FF"]],
                    "sizes": 3,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_sequences_with_mixed_color_formats_structured(self):
        param = EdgeColorSequencesParam(
            value=[["face", "#00FF00", (0, 0, 1)], ["#FF0000", (0, 1, 0), "face"]],
            sizes=2,
            structured=True,
        )
        self.assertEqual(
            param.value,
            [
                ["face", "#00FF00", (0.0, 0.0, 1.0)],
                ["#FF0000", (0.0, 1.0, 0.0), "face"],
            ],
        )
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)


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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = MarkerSequenceParam(value=["o", "s", "D"], sizes=3, structured=True)
        self.assertEqual(param.value, ["o", "s", "D"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequenceParam(value=["o", "s", "D"], sizes=4, structured=True)
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequenceParam(value=["o", "s", "D"], sizes=[2, 3, 4], structured=True)
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = MarkerSequenceParam(value=["o", "s"], sizes=[2, 3], structured=False)
        self.assertEqual(param.value, ["o", "s"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = MarkerSequenceParam(value=["o", "s"])
        self.assertEqual(param.value, ["o", "s"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = MarkerSequenceParam.model_validate(
            {"value": ["o", "s", "D"], "sizes": 3, "structured": True}
        )
        self.assertEqual(param.value, ["o", "s", "D"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequenceParam.model_validate(
                {"value": ["o", "s", "D"], "sizes": 4, "structured": True}
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )


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

    def test_init_with_structured_true_and_valid_sizes(self):
        param = MarkerSequencesParam(
            value=[["o", "s"], ["D", "p"]], sizes=2, sub_sizes=[2, 2], structured=True
        )
        self.assertEqual(param.value, [["o", "s"], ["D", "p"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = MarkerSequencesParam(
            value=[["o"], ["D", "p"], ["s", "x", "*"]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(param.value, [["o"], ["D", "p"], ["s", "x", "*"]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequencesParam(
                value=[["o", "s"], ["D", "p"]],
                sizes=3,
                sub_sizes=[2, 2],
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequencesParam(
                value=[["o", "s", "D"], ["p", "x"]],
                sizes=2,
                sub_sizes=[3, 3],
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequencesParam(
                value=[["o"], ["D", "p"]],
                sizes=2,
                sub_sizes=[1, 2, 3],
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = MarkerSequencesParam(
            value=[["o"], ["D", "p"]], sizes=[2, 3], sub_sizes=[1, 2], structured=False
        )
        self.assertEqual(param.value, [["o"], ["D", "p"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [1, 2])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = MarkerSequencesParam(value=[["o"], ["D", "p"]])
        self.assertEqual(param.value, [["o"], ["D", "p"]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = MarkerSequencesParam.model_validate(
            {
                "value": [["o", "s"], ["D", "p"]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["o", "s"], ["D", "p"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            MarkerSequencesParam.model_validate(
                {
                    "value": [["o", "s"], ["D", "p"]],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )


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

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate({"value": [1], "options": ["option1"]})
        with self.assertRaises(ValidationError):
            LiteralSequenceParam.model_validate(
                {"value": ["option1"], "options": "option1"}
            )

    def test_init_with_structured_true_and_valid_sizes(self):
        param = LiteralSequenceParam(
            value=["option1", "option2", "option1"],
            options=["option1", "option2"],
            sizes=3,
            structured=True,
        )
        self.assertEqual(param.value, ["option1", "option2", "option1"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequenceParam(
                value=["option1", "option2", "option1"],
                options=["option1", "option2"],
                sizes=4,
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequenceParam(
                value=["option1", "option2", "option1"],
                options=["option1", "option2"],
                sizes=[2, 3, 4],
                structured=True,
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = LiteralSequenceParam(
            value=["option1", "option2"],
            options=["option1", "option2"],
            sizes=[2, 3],
            structured=False,
        )
        self.assertEqual(param.value, ["option1", "option2"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = LiteralSequenceParam(
            value=["option1", "option2"], options=["option1", "option2"]
        )
        self.assertEqual(param.value, ["option1", "option2"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = LiteralSequenceParam.model_validate(
            {
                "value": ["option1", "option2", "option1"],
                "options": ["option1", "option2"],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, ["option1", "option2", "option1"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequenceParam.model_validate(
                {
                    "value": ["option1", "option2", "option1"],
                    "options": ["option1", "option2"],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
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

    def test_init_with_invalid_types(self):
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [[1]], "options": ["option1"]}
            )
        with self.assertRaises(ValidationError):
            LiteralSequencesParam.model_validate(
                {"value": [["option1"]], "options": "option1"}
            )

    def test_init_with_structured_true_and_valid_sizes(self):
        param = LiteralSequencesParam(
            value=[["option1", "option2"], ["option2", "option1"]],
            options=["option1", "option2"],
            sizes=2,
            sub_sizes=[2, 2],
            structured=True,
        )
        self.assertEqual(param.value, [["option1", "option2"], ["option2", "option1"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = LiteralSequencesParam(
            value=[
                ["option1"],
                ["option1", "option2"],
                ["option1", "option2", "option1"],
            ],
            options=["option1", "option2"],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(
            param.value,
            [["option1"], ["option1", "option2"], ["option1", "option2", "option1"]],
        )
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequencesParam(
                value=[["option1", "option2"], ["option2", "option1"]],
                options=["option1", "option2"],
                sizes=3,
                sub_sizes=[2, 2],
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequencesParam(
                value=[["option1", "option2", "option1"], ["option2", "option1"]],
                options=["option1", "option2"],
                sizes=2,
                sub_sizes=[3, 3],
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequencesParam(
                value=[["option1"], ["option1", "option2"]],
                options=["option1", "option2"],
                sizes=2,
                sub_sizes=[1, 2, 3],
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = LiteralSequencesParam(
            value=[["option1"], ["option1", "option2"]],
            options=["option1", "option2"],
            sizes=[2, 3],
            sub_sizes=[1, 2],
            structured=False,
        )
        self.assertEqual(param.value, [["option1"], ["option1", "option2"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [1, 2])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = LiteralSequencesParam(
            value=[["option1"], ["option1", "option2"]], options=["option1", "option2"]
        )
        self.assertEqual(param.value, [["option1"], ["option1", "option2"]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = LiteralSequencesParam.model_validate(
            {
                "value": [["option1", "option2"], ["option2", "option1"]],
                "options": ["option1", "option2"],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["option1", "option2"], ["option2", "option1"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            LiteralSequencesParam.model_validate(
                {
                    "value": [["option1", "option2"], ["option2", "option1"]],
                    "options": ["option1", "option2"],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )


class TestNormalizationParam(TestCase):
    def test_init_with_valid_literals(self):
        NormalizationParam(value="linear")
        NormalizationParam(value="log")
        NormalizationParam(value="symlog")

    def test_init_with_valid_normalize_instance(self):
        NormalizationParam(value=Normalize())

    def test_init_with_invalid_literals(self):
        with self.assertRaises(ValidationError):
            NormalizationParam(value="invalid")
        with self.assertRaises(ValidationError):
            NormalizationParam(value=123)


class TestNormalizationSequenceParam(TestCase):
    def test_init_with_valid_literals(self):
        NormalizationSequenceParam(value=["linear", "log", "symlog"])
        NormalizationSequenceParam(value=["linear"])

    def test_init_with_valid_normalize_instances(self):
        NormalizationSequenceParam(value=[Normalize(), Normalize()])

    def test_init_with_mixed_valid_values(self):
        NormalizationSequenceParam(value=["linear", Normalize(), "log"])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            NormalizationSequenceParam(value=["invalid"])
        with self.assertRaises(ValidationError):
            NormalizationSequenceParam(value=[123])
        with self.assertRaises(ValidationError):
            NormalizationSequenceParam(value=["linear", "invalid"])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = NormalizationSequenceParam(
            value=["linear", "log", "symlog"], sizes=3, structured=True
        )
        self.assertEqual(param.value, ["linear", "log", "symlog"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            NormalizationSequenceParam(
                value=["linear", "log", "symlog"], sizes=4, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            NormalizationSequenceParam(
                value=["linear", "log", "symlog"], sizes=[2, 3, 4], structured=True
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = NormalizationSequenceParam(
            value=["linear", "log"], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, ["linear", "log"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = NormalizationSequenceParam(value=["linear", "log"])
        self.assertEqual(param.value, ["linear", "log"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = NormalizationSequenceParam.model_validate(
            {
                "value": ["linear", "log", "symlog"],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, ["linear", "log", "symlog"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            NormalizationSequenceParam.model_validate(
                {
                    "value": ["linear", "log", "symlog"],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )


class TestNormalizationSequencesParam(TestCase):
    def test_init_with_valid_literals(self):
        NormalizationSequencesParam(value=[["linear", "log"], ["symlog"]])
        NormalizationSequencesParam(value=[["linear"]])

    def test_init_with_valid_normalize_instances(self):
        NormalizationSequencesParam(value=[[Normalize()], [Normalize()]])

    def test_init_with_mixed_valid_values(self):
        NormalizationSequencesParam(value=[["linear", Normalize()], ["log"]])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            NormalizationSequencesParam(value=[["invalid"]])
        with self.assertRaises(ValidationError):
            NormalizationSequencesParam(value=[[123]])
        with self.assertRaises(ValidationError):
            NormalizationSequencesParam(value=[["linear", "invalid"]])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = NormalizationSequencesParam(
            value=[["linear", "log"], ["symlog", "linear"]],
            sizes=2,
            structured=True,
        )
        self.assertEqual(param.value, [["linear", "log"], ["symlog", "linear"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = NormalizationSequencesParam(
            value=[["linear"], ["log", "symlog"], ["linear", "log", "symlog"]],
            sizes=3,
            structured=True,
        )
        self.assertEqual(
            param.value, [["linear"], ["log", "symlog"], ["linear", "log", "symlog"]]
        )
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            NormalizationSequencesParam(
                value=[["linear", "log"], ["symlog", "linear"]],
                sizes=3,
                sub_sizes=[2, 2],
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = NormalizationSequencesParam(
            value=[["linear"], ["log", "symlog"]],
            sizes=[2, 3],
            structured=False,
        )
        self.assertEqual(param.value, [["linear"], ["log", "symlog"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = NormalizationSequencesParam(value=[["linear"], ["log", "symlog"]])
        self.assertEqual(param.value, [["linear"], ["log", "symlog"]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = NormalizationSequencesParam.model_validate(
            {
                "value": [["linear", "log"], ["symlog", "linear"]],
                "sizes": 2,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["linear", "log"], ["symlog", "linear"]])
        self.assertEqual(param.sizes, [2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            NormalizationSequencesParam.model_validate(
                {
                    "value": [["linear", "log"], ["symlog", "linear"]],
                    "sizes": 3,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )


class TestColormapParam(TestCase):
    def test_init_with_valid_literals(self):
        ColormapParam(value="viridis")
        ColormapParam(value="plasma")
        ColormapParam(value="inferno")

    def test_init_with_valid_colormap_instance(self):
        ColormapParam(value=Colormap("viridis"))

    def test_init_with_invalid_literals(self):
        with self.assertRaises(ValidationError):
            ColormapParam(value="invalid")
        with self.assertRaises(ValidationError):
            ColormapParam(value=123)

    def test_init_with_none(self):
        with self.assertRaises(ValidationError):
            ColormapParam(value=None)


class TestColormapSequenceParam(TestCase):
    def test_init_with_valid_literals(self):
        ColormapSequenceParam(value=["viridis", "plasma", "inferno"])

    def test_init_with_valid_colormap_instances(self):
        ColormapSequenceParam(value=[Colormap("viridis"), Colormap("plasma")])

    def test_init_with_mixed_valid_values(self):
        ColormapSequenceParam(value=["viridis", Colormap("plasma"), "inferno"])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            ColormapSequenceParam(value=["invalid", "viridis"])
        with self.assertRaises(ValidationError):
            ColormapSequenceParam(value=[123, "viridis"])

    def test_init_with_empty_sequence(self):
        ColormapSequenceParam(value=[])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = ColormapSequenceParam(
            value=["viridis", "plasma", "inferno"], sizes=3, structured=True
        )
        self.assertEqual(param.value, ["viridis", "plasma", "inferno"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequenceParam(
                value=["viridis", "plasma", "inferno"], sizes=4, structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequenceParam(
                value=["viridis", "plasma", "inferno"], sizes=[2, 3, 4], structured=True
            )
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = ColormapSequenceParam(
            value=["viridis", "plasma"], sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, ["viridis", "plasma"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = ColormapSequenceParam(value=["viridis", "plasma"])
        self.assertEqual(param.value, ["viridis", "plasma"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = ColormapSequenceParam.model_validate(
            {
                "value": ["viridis", "plasma", "inferno"],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, ["viridis", "plasma", "inferno"])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequenceParam.model_validate(
                {
                    "value": ["viridis", "plasma", "inferno"],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )


class TestColormapSequencesParam(TestCase):
    def test_init_with_valid_literals(self):
        ColormapSequencesParam(value=[["viridis", "plasma"], ["inferno", "magma"]])

    def test_init_with_valid_colormap_instances(self):
        ColormapSequencesParam(value=[[Colormap("viridis")], [Colormap("plasma")]])

    def test_init_with_mixed_valid_values(self):
        ColormapSequencesParam(value=[["viridis", Colormap("plasma")], ["inferno"]])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            ColormapSequencesParam(value=[["invalid", "viridis"], ["plasma"]])
        with self.assertRaises(ValidationError):
            ColormapSequencesParam(value=[[123, "viridis"], ["plasma"]])

    def test_init_with_empty_sequences(self):
        ColormapSequencesParam(value=[[], []])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = ColormapSequencesParam(
            value=[["viridis", "plasma"], ["inferno", "magma"]],
            sizes=2,
            sub_sizes=[2, 2],
            structured=True,
        )
        self.assertEqual(param.value, [["viridis", "plasma"], ["inferno", "magma"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = ColormapSequencesParam(
            value=[["viridis"], ["plasma", "inferno"], ["magma", "viridis", "plasma"]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(
            param.value,
            [["viridis"], ["plasma", "inferno"], ["magma", "viridis", "plasma"]],
        )
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequencesParam(
                value=[["viridis", "plasma"], ["inferno", "magma"]],
                sizes=3,
                sub_sizes=[2, 2],
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequencesParam(
                value=[["viridis", "plasma", "inferno"], ["magma", "viridis"]],
                sizes=2,
                sub_sizes=[3, 3],
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequencesParam(
                value=[["viridis"], ["plasma", "inferno"]],
                sizes=2,
                sub_sizes=[1, 2, 3],
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = ColormapSequencesParam(
            value=[["viridis"], ["plasma", "inferno"]],
            sizes=[2, 3],
            sub_sizes=[1, 2],
            structured=False,
        )
        self.assertEqual(param.value, [["viridis"], ["plasma", "inferno"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [1, 2])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = ColormapSequencesParam(value=[["viridis"], ["plasma", "inferno"]])
        self.assertEqual(param.value, [["viridis"], ["plasma", "inferno"]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = ColormapSequencesParam.model_validate(
            {
                "value": [["viridis", "plasma"], ["inferno", "magma"]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [["viridis", "plasma"], ["inferno", "magma"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            ColormapSequencesParam.model_validate(
                {
                    "value": [["viridis", "plasma"], ["inferno", "magma"]],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )


class TestBinsParam(TestCase):
    def test_init_with_valid_integer(self):
        BinsParam(value=10)

    def test_init_with_valid_literal(self):
        BinsParam(value="auto")

    def test_init_with_invalid_value(self):
        with self.assertRaises(ValidationError):
            BinsParam(value=-1)
        with self.assertRaises(ValidationError):
            BinsParam(value="invalid")


class TestBinsSequenceParam(TestCase):
    def test_init_with_valid_integers(self):
        BinsSequenceParam(value=[10, 20, 30])

    def test_init_with_valid_literals(self):
        BinsSequenceParam(value=["auto", "sturges", "fd"])

    def test_init_with_mixed_valid_values(self):
        BinsSequenceParam(value=[10, "auto", 20])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            BinsSequenceParam(value=[-1, 10])
        with self.assertRaises(ValidationError):
            BinsSequenceParam(value=["invalid", "auto"])

    def test_init_with_empty_sequence(self):
        BinsSequenceParam(value=[])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = BinsSequenceParam(value=[10, "auto", 20], sizes=3, structured=True)
        self.assertEqual(param.value, [10, "auto", 20])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequenceParam(value=[10, "auto", 20], sizes=4, structured=True)
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_sequence_sizes(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequenceParam(value=[10, "auto", 20], sizes=[2, 3, 4], structured=True)
        self.assertIn(
            "Validation of structured Sequence requires a single size: int",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = BinsSequenceParam(value=[10, "auto"], sizes=[2, 3], structured=False)
        self.assertEqual(param.value, [10, "auto"])
        self.assertEqual(param.sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = BinsSequenceParam(value=[10, "auto"])
        self.assertEqual(param.value, [10, "auto"])
        self.assertIsNone(param.sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = BinsSequenceParam.model_validate(
            {
                "value": [10, "auto", 20],
                "sizes": 3,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [10, "auto", 20])
        self.assertEqual(param.sizes, [3])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequenceParam.model_validate(
                {
                    "value": [10, "auto", 20],
                    "sizes": 4,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [4], got size: 3 instead",
            str(context.exception),
        )


class TestBinsSequencesParam(TestCase):
    def test_init_with_valid_integers(self):
        BinsSequencesParam(value=[[10, 20], [30, 40]])

    def test_init_with_valid_literals(self):
        BinsSequencesParam(value=[["auto", "sturges"], ["fd", "sqrt"]])

    def test_init_with_mixed_valid_values(self):
        BinsSequencesParam(value=[[10, "auto"], ["fd", 20]])

    def test_init_with_invalid_values(self):
        with self.assertRaises(ValidationError):
            BinsSequencesParam(value=[[-1, 10], [20, 30]])
        with self.assertRaises(ValidationError):
            BinsSequencesParam(value=[["invalid", "auto"], ["fd"]])

    def test_init_with_empty_sequences(self):
        BinsSequencesParam(value=[[], []])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = BinsSequencesParam(
            value=[[10, "auto"], [20, "sturges"]],
            sizes=2,
            sub_sizes=[2, 2],
            structured=True,
        )
        self.assertEqual(param.value, [[10, "auto"], [20, "sturges"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = BinsSequencesParam(
            value=[[10], ["auto", "sturges"], [20, "fd", 30]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(param.value, [[10], ["auto", "sturges"], [20, "fd", 30]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequencesParam(
                value=[[10, "auto"], [20, "sturges"]],
                sizes=3,
                sub_sizes=[2, 2],
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequencesParam(
                value=[[10, "auto", "sturges"], [20, "fd"]],
                sizes=2,
                sub_sizes=[3, 3],
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequencesParam(
                value=[[10], ["auto", "sturges"]],
                sizes=2,
                sub_sizes=[1, 2, 3],
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = BinsSequencesParam(
            value=[[10], ["auto", "sturges"]],
            sizes=[2, 3],
            sub_sizes=[1, 2],
            structured=False,
        )
        self.assertEqual(param.value, [[10], ["auto", "sturges"]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [1, 2])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = BinsSequencesParam(value=[[10], ["auto", "sturges"]])
        self.assertEqual(param.value, [[10], ["auto", "sturges"]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = BinsSequencesParam.model_validate(
            {
                "value": [[10, "auto"], [20, "sturges"]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [[10, "auto"], [20, "sturges"]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            BinsSequencesParam.model_validate(
                {
                    "value": [[10, "auto"], [20, "sturges"]],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )


class TestRangeSequenceParam(TestCase):
    def test_init_with_valid_sequence(self):
        param = RangeSequenceParam(value=[5, 10])
        self.assertEqual(len(param.value), 2)
        self.assertEqual(param.value[0], 5)
        self.assertEqual(param.value[1], 10)

    def test_init_with_valid_sequence_and_ranges(self):
        param = RangeSequenceParam(value=[5, 15], min_value=0, max_value=20)
        self.assertEqual(param.value[0], 5)
        self.assertEqual(param.value[1], 15)
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 20)

    def test_init_with_empty_sequence(self):
        param = RangeSequenceParam(value=[])
        self.assertEqual(len(param.value), 0)

    def test_init_with_single_element(self):
        param = RangeSequenceParam(value=[5])
        self.assertEqual(len(param.value), 1)
        self.assertEqual(param.value[0], 5)

    def test_init_with_strict_ranges(self):
        param = RangeSequenceParam(
            value=[5, 15], min_value=0, max_value=20, strict=True
        )
        self.assertEqual(param.value[0], 5)
        self.assertEqual(param.value[1], 15)
        self.assertTrue(param.strict)

    def test_init_with_float_values(self):
        param = RangeSequenceParam(value=[5.5, 10.5])
        self.assertEqual(param.value[0], 5.5)
        self.assertEqual(param.value[1], 10.5)

    def test_init_with_mixed_int_and_float(self):
        param = RangeSequenceParam(value=[5, 10.5])
        self.assertEqual(param.value[0], 5)
        self.assertEqual(param.value[1], 10.5)

    def test_init_with_invalid_sequence_type(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value="not_a_sequence")

    def test_init_with_invalid_element_type(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=["5", "10"])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=[5, None])

    def test_init_with_value_below_min(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=[5, 15], min_value=10, max_value=20)

    def test_init_with_value_above_max(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=[25, 15], min_value=10, max_value=20)

    def test_init_with_strict_value_at_min(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=[10, 15], min_value=10, max_value=20, strict=True)

    def test_init_with_strict_value_at_max(self):
        with self.assertRaises(ValidationError):
            RangeSequenceParam(value=[15, 20], min_value=10, max_value=20, strict=True)

    def test_init_with_very_large_numbers(self):
        param = RangeSequenceParam(value=[1e100, 1e101])
        self.assertEqual(param.value[0], 1e100)
        self.assertEqual(param.value[1], 1e101)

    def test_init_with_very_small_numbers(self):
        param = RangeSequenceParam(value=[1e-100, 1e-101])
        self.assertEqual(param.value[0], 1e-100)
        self.assertEqual(param.value[1], 1e-101)


class TestRangeSequencesParam(TestCase):
    def test_init_with_valid_sequences(self):
        param = RangeSequencesParam(value=[[5, 10], [15, 20]])
        self.assertEqual(len(param.value), 2)
        self.assertEqual(len(param.value[0]), 2)
        self.assertEqual(len(param.value[1]), 2)
        self.assertEqual(param.value[0][0], 5)
        self.assertEqual(param.value[0][1], 10)
        self.assertEqual(param.value[1][0], 15)
        self.assertEqual(param.value[1][1], 20)

    def test_init_with_valid_sequences_and_ranges(self):
        param = RangeSequencesParam(
            value=[[5, 15], [25, 35]], min_value=0, max_value=40
        )
        self.assertEqual(param.value[0][0], 5)
        self.assertEqual(param.value[0][1], 15)
        self.assertEqual(param.value[1][0], 25)
        self.assertEqual(param.value[1][1], 35)
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 40)

    def test_init_with_empty_sequences(self):
        param = RangeSequencesParam(value=[[], []])
        self.assertEqual(len(param.value), 2)
        self.assertEqual(len(param.value[0]), 0)
        self.assertEqual(len(param.value[1]), 0)

    def test_init_with_single_sequence(self):
        param = RangeSequencesParam(value=[[5]])
        self.assertEqual(len(param.value), 1)
        self.assertEqual(len(param.value[0]), 1)
        self.assertEqual(param.value[0][0], 5)

    def test_init_with_strict_ranges(self):
        param = RangeSequencesParam(
            value=[[5, 15]], min_value=0, max_value=20, strict=True
        )
        self.assertEqual(param.value[0][0], 5)
        self.assertEqual(param.value[0][1], 15)
        self.assertTrue(param.strict)

    def test_init_with_float_values(self):
        param = RangeSequencesParam(value=[[5.5, 10.5], [15.5, 20.5]])
        self.assertEqual(param.value[0][0], 5.5)
        self.assertEqual(param.value[1][1], 20.5)

    def test_init_with_mixed_int_and_float(self):
        param = RangeSequencesParam(value=[[5, 10.5], [15, 20.5]])
        self.assertEqual(param.value[0][0], 5)
        self.assertEqual(param.value[0][1], 10.5)
        self.assertEqual(param.value[1][0], 15)
        self.assertEqual(param.value[1][1], 20.5)

    def test_init_with_invalid_outer_sequence_type(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value="not_a_sequence")

    def test_init_with_invalid_inner_sequence_type(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=["not_a_sequence"])

    def test_init_with_invalid_element_type(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=[["5", "10"], ["15", "20"]])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=None)

    def test_init_with_none_in_sequence(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=[[5, None], [15, 20]])

    def test_init_with_value_below_min(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=[[5, 15], [25, 35]], min_value=10, max_value=40)

    def test_init_with_value_above_max(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(value=[[15, 25], [35, 45]], min_value=10, max_value=40)

    def test_init_with_strict_value_at_min(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(
                value=[[10, 15], [25, 35]], min_value=10, max_value=40, strict=True
            )

    def test_init_with_strict_value_at_max(self):
        with self.assertRaises(ValidationError):
            RangeSequencesParam(
                value=[[15, 25], [35, 40]], min_value=10, max_value=40, strict=True
            )

    def test_init_with_uneven_sequence_lengths(self):
        param = RangeSequencesParam(value=[[5], [10, 15]])
        self.assertEqual(len(param.value[0]), 1)
        self.assertEqual(len(param.value[1]), 2)

    def test_init_with_very_large_numbers(self):
        param = RangeSequencesParam(value=[[1e100, 1e101], [1e102, 1e103]])
        self.assertEqual(param.value[0][0], 1e100)
        self.assertEqual(param.value[1][1], 1e103)

    def test_init_with_very_small_numbers(self):
        param = RangeSequencesParam(value=[[1e-100, 1e-101], [1e-102, 1e-103]])
        self.assertEqual(param.value[0][0], 1e-100)
        self.assertEqual(param.value[1][1], 1e-103)

    def test_init_with_structured_true_and_valid_sizes(self):
        param = RangeSequencesParam(
            value=[[1, 2], [3, 4]],
            sizes=2,
            sub_sizes=[2, 2],
            min_value=0,
            max_value=5,
            structured=True,
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 5)
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = RangeSequencesParam(
            value=[[1], [1, 2], [1, 2, 3]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            min_value=0,
            max_value=5,
            structured=True,
        )
        self.assertEqual(param.value, [[1], [1, 2], [1, 2, 3]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 5)
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam(
                value=[[1, 2], [3, 4]],
                sizes=3,
                sub_sizes=[2, 2],
                min_value=0,
                max_value=5,
                structured=True,
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam(
                value=[[1, 2, 3], [4, 5]],
                sizes=2,
                sub_sizes=[3, 3],
                min_value=0,
                max_value=5,
                structured=True,
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam(
                value=[[1], [2, 3]],
                sizes=2,
                sub_sizes=[1, 2, 3],
                min_value=0,
                max_value=5,
                structured=True,
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_true_and_out_of_range(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam(
                value=[[1, 2], [3, 6]],
                sizes=2,
                sub_sizes=[2, 2],
                min_value=0,
                max_value=5,
                structured=True,
            )
        self.assertIn(
            "Value 6 at index [1, 1] is not in range [0, 5]", str(context.exception)
        )

    def test_init_with_structured_true_and_strict_range(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam(
                value=[[1, 2], [3, 5]],
                sizes=2,
                sub_sizes=[2, 2],
                min_value=0,
                max_value=5,
                strict=True,
                structured=True,
            )
        self.assertIn(
            "Value 5 at index [1, 1] is not in range (0, 5)", str(context.exception)
        )

    def test_init_with_structured_false_and_valid_sizes(self):
        param = RangeSequencesParam(
            value=[[1, 2], [3, 4]],
            sizes=[2, 3],
            sub_sizes=[2, 3],
            min_value=0,
            max_value=5,
            structured=False,
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [2, 3])
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 5)
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = RangeSequencesParam(value=[[1, 2], [3, 4]], min_value=0, max_value=5)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 5)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = RangeSequencesParam.model_validate(
            {
                "value": [[1, 2], [3, 4]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "min_value": 0,
                "max_value": 5,
                "structured": True,
            }
        )
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertEqual(param.min_value, 0)
        self.assertEqual(param.max_value, 5)
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam.model_validate(
                {
                    "value": [[1, 2], [3, 4]],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "min_value": 0,
                    "max_value": 5,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam.model_validate(
                {
                    "value": [[1, 2, 3], [4, 5]],
                    "sizes": 2,
                    "sub_sizes": [3, 3],
                    "min_value": 0,
                    "max_value": 5,
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_out_of_range(self):
        with self.assertRaises(ValidationError) as context:
            RangeSequencesParam.model_validate(
                {
                    "value": [[1, 2], [3, 6]],
                    "sizes": 2,
                    "sub_sizes": [2, 2],
                    "min_value": 0,
                    "max_value": 5,
                    "structured": True,
                }
            )
        self.assertIn(
            "Value 6 at index [1, 1] is not in range [0, 5]", str(context.exception)
        )


class TestSpineParam(TestCase):
    def test_valid_minimal(self):
        param = SpineParam()
        self.assertTrue(param.visible)

    def test_valid_full(self):
        param = SpineParam(
            visible=False,
            position=(0.0, 1.0),
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.7,
            bounds=(0.1, 0.9),
            capstyle="round",
        )
        self.assertEqual(param.capstyle, "round")
        self.assertEqual(param.linewidth, 2.5)

    def test_invalid_extra_field(self):
        with self.assertRaises(ValidationError):
            SpineParam(invalid_field=123)

    def test_invalid_capstyle_literal(self):
        with self.assertRaises(ValidationError):
            SpineParam(capstyle="square")  # Not in allowed literals

    def test_invalid_position_type(self):
        with self.assertRaises(ValidationError):
            SpineParam(position=42)  # Not a tuple or str

    def test_invalid_bounds_length(self):
        with self.assertRaises(ValidationError):
            SpineParam(bounds=(0.1,))  # Too short

    def test_color_can_be_rgb_tuple(self):
        param = SpineParam(color=(0.1, 0.2, 0.3))
        self.assertEqual(param.color, (0.1, 0.2, 0.3))


class TestSpinesParam(TestCase):
    def test_valid_empty(self):
        spines = SpinesParam()
        self.assertIsNone(spines.left)
        self.assertIsNone(spines.top)

    def test_partial_side_config(self):
        spines = SpinesParam(
            left=SpineParam(visible=False), top=SpineParam(color="red")
        )
        self.assertFalse(spines.left.visible)
        self.assertEqual(spines.top.color, "red")
        self.assertIsNone(spines.right)

    def test_invalid_side_type(self):
        with self.assertRaises(ValidationError):
            SpinesParam(left={"invalid": True})  # not a SpineParam instance

    def test_all_sides_configured(self):
        spines = SpinesParam(
            left=SpineParam(linewidth=1.0),
            right=SpineParam(linewidth=2.0),
            top=SpineParam(linewidth=3.0),
            bottom=SpineParam(linewidth=4.0),
        )
        self.assertEqual(spines.left.linewidth, 1.0)
        self.assertEqual(spines.right.linewidth, 2.0)
        self.assertEqual(spines.top.linewidth, 3.0)
        self.assertEqual(spines.bottom.linewidth, 4.0)


class TestDataSequenceParam(TestCase):
    def test_init_with_valid_numeric_sequence(self):
        param = DataSequenceParam(value=[1, 2, 3])
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_valid_float_sequence(self):
        param = DataSequenceParam(value=[1.0, 2.5, 3.7])
        self.assertEqual(param.value, [1.0, 2.5, 3.7])

    def test_init_with_mixed_numeric_sequence(self):
        param = DataSequenceParam(value=[1, 2.5, 3])
        self.assertEqual(param.value, [1, 2.5, 3])

    def test_init_with_none_values(self):
        param = DataSequenceParam(value=[1, None, 3])
        self.assertEqual(param.value, [1, None, 3])

    def test_init_with_all_none_values(self):
        param = DataSequenceParam(value=[None, None, None])
        self.assertEqual(param.value, [None, None, None])

    def test_init_with_empty_sequence(self):
        param = DataSequenceParam(value=[])
        self.assertEqual(param.value, [])

    def test_init_with_numpy_array(self):
        param = DataSequenceParam(value=np.array([1, 2, 3]))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_pandas_series(self):
        param = DataSequenceParam(value=pd.Series([1, 2, 3]))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_tuple(self):
        param = DataSequenceParam(value=(1, 2, 3))
        self.assertEqual(param.value, [1, 2, 3])

    def test_init_with_valid_sizes(self):
        param = DataSequenceParam(value=[1, 2, 3], sizes=3)
        self.assertEqual(param.value, [1, 2, 3])
        self.assertEqual(param.sizes, [3])

    def test_init_with_invalid_sizes(self):
        with self.assertRaises(ValidationError):
            DataSequenceParam(value=[1, 2, 3], sizes=4)

    def test_init_with_invalid_sequence_type(self):
        with self.assertRaises(ValidationError):
            DataSequenceParam(value="not_a_sequence")

    def test_init_with_invalid_numeric_type(self):
        with self.assertRaises(ValidationError):
            DataSequenceParam(value=[1, "2", 3])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            DataSequenceParam(value=None)

    def test_init_with_very_large_numbers(self):
        param = DataSequenceParam(value=[1e100, 1e101, 1e102])
        self.assertEqual(param.value, [1e100, 1e101, 1e102])

    def test_init_with_very_small_numbers(self):
        param = DataSequenceParam(value=[1e-100, 1e-101, 1e-102])
        self.assertEqual(param.value, [1e-100, 1e-101, 1e-102])

    def test_init_with_mixed_none_and_numbers(self):
        param = DataSequenceParam(value=[1e100, None, 1e-100])
        self.assertEqual(param.value, [1e100, None, 1e-100])

    def test_init_with_dict_initialization(self):
        param = DataSequenceParam.model_validate({"value": [1, None, 3], "sizes": 3})
        self.assertEqual(param.value, [1, None, 3])
        self.assertEqual(param.sizes, [3])


class TestDataSequencesParam(TestCase):
    def test_init_with_valid_numeric_sequences(self):
        param = DataSequencesParam(value=[[1, 2, 3], [4, 5, 6]])
        self.assertEqual(param.value, [[1, 2, 3], [4, 5, 6]])

    def test_init_with_valid_float_sequences(self):
        param = DataSequencesParam(value=[[1.0, 2.5], [3.7, 4.2]])
        self.assertEqual(param.value, [[1.0, 2.5], [3.7, 4.2]])

    def test_init_with_mixed_numeric_sequences(self):
        param = DataSequencesParam(value=[[1, 2.5], [3, 4.2]])
        self.assertEqual(param.value, [[1, 2.5], [3, 4.2]])

    def test_init_with_none_values(self):
        param = DataSequencesParam(value=[[1, None], [None, 4]])
        self.assertEqual(param.value, [[1, None], [None, 4]])

    def test_init_with_all_none_values(self):
        param = DataSequencesParam(value=[[None, None], [None, None]])
        self.assertEqual(param.value, [[None, None], [None, None]])

    def test_init_with_empty_sequences(self):
        param = DataSequencesParam(value=[[], []])
        self.assertEqual(param.value, [[], []])

    def test_init_with_single_value_sequences(self):
        param = DataSequencesParam(value=[[1], [2]])
        self.assertEqual(param.value, [[1], [2]])

    def test_init_with_single_none_sequences(self):
        param = DataSequencesParam(value=[[None], [None]])
        self.assertEqual(param.value, [[None], [None]])

    def test_init_with_numpy_arrays(self):
        param = DataSequencesParam(value=[np.array([1, 2]), np.array([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_pandas_series(self):
        param = DataSequencesParam(value=[pd.Series([1, 2]), pd.Series([3, 4])])
        self.assertEqual(param.value, [[1, 2], [3, 4]])

    def test_init_with_mixed_sequences_types(self):
        param = DataSequencesParam(
            value=[np.array([1, 2]), pd.Series([3, 4]), [5, 6], (7, 8)]
        )
        self.assertEqual(param.value, [[1, 2], [3, 4], [5, 6], [7, 8]])

    def test_init_with_valid_sizes(self):
        param = DataSequencesParam(value=[[1, 2], [3, 4]], sizes=2, sub_sizes=2)
        self.assertEqual(param.value, [[1, 2], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2])

    def test_init_with_invalid_sizes(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[[1, 2], [3, 4]], sizes=3)

    def test_init_with_invalid_sub_sizes(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[[1, 2], [3, 4]], sub_sizes=3)

    def test_init_with_single_point_sequence(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[1, 2, 3])

    def test_init_with_single_point_none_sequence(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[None, 2, None])

    def test_init_with_invalid_sequence_type(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value="not_a_sequence")

    def test_init_with_invalid_numeric_type(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[[1, "2"], [3, 4]])

    def test_init_with_none_value(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=None)

    def test_init_with_very_large_numbers(self):
        param = DataSequencesParam(value=[[1e100, 1e101], [1e102, 1e103]])
        self.assertEqual(param.value, [[1e100, 1e101], [1e102, 1e103]])

    def test_init_with_very_small_numbers(self):
        param = DataSequencesParam(value=[[1e-100, 1e-101], [1e-102, 1e-103]])
        self.assertEqual(param.value, [[1e-100, 1e-101], [1e-102, 1e-103]])

    def test_init_with_mixed_none_and_numbers(self):
        param = DataSequencesParam(value=[[1e100, None], [None, 1e-100]])
        self.assertEqual(param.value, [[1e100, None], [None, 1e-100]])

    def test_init_with_uneven_sequence_lengths(self):
        param = DataSequencesParam(value=[[1, 2, 3], [4, 5], [6]])
        self.assertEqual(param.value, [[1, 2, 3], [4, 5], [6]])

    def test_init_with_dict_initialization(self):
        param = DataSequencesParam.model_validate(
            {"value": [[1, None], [3, 4]], "sizes": 2, "sub_sizes": 2}
        )
        self.assertEqual(param.value, [[1, None], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2])

    def test_init_with_structured_true_and_valid_sizes(self):
        param = DataSequencesParam(
            value=[[1, None], [3, 4]], sizes=2, sub_sizes=[2, 2], structured=True
        )
        self.assertEqual(param.value, [[1, None], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_valid_sequence_sizes(self):
        param = DataSequencesParam(
            value=[[1], [1, None], [1, 2, 3]],
            sizes=3,
            sub_sizes=[1, 2, 3],
            structured=True,
        )
        self.assertEqual(param.value, [[1], [1, None], [1, 2, 3]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 2, 3])
        self.assertTrue(param.structured)

    def test_init_with_structured_true_and_invalid_outer_size(self):
        with self.assertRaises(ValidationError) as context:
            DataSequencesParam(
                value=[[1, None], [3, 4]], sizes=3, sub_sizes=[2, 2], structured=True
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_structured_true_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            DataSequencesParam(
                value=[[1, 2, None], [4, 5]], sizes=2, sub_sizes=[3, 3], structured=True
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )

    def test_init_with_structured_true_and_mismatched_sub_sizes_length(self):
        with self.assertRaises(ValidationError) as context:
            DataSequencesParam(
                value=[[1], [None, 2]], sizes=2, sub_sizes=[1, 2, 3], structured=True
            )
        self.assertIn(
            "Mismatch in structured Sequence of length",
            str(context.exception),
        )

    def test_init_with_structured_true_and_single_point_sequence(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[1, None, 3], structured=True)

    def test_init_with_structured_true_and_single_point_sequence_with_sizes(self):
        with self.assertRaises(ValidationError):
            DataSequencesParam(value=[1, None, 3], sizes=3, structured=True)

    def test_init_with_structured_true_and_mixed_sequences(self):
        param = DataSequencesParam(
            value=[[1], [None], [3]], sizes=3, sub_sizes=[1, 1, 1], structured=True
        )
        self.assertEqual(param.value, [[1], [None], [3]])
        self.assertEqual(param.sizes, [3])
        self.assertEqual(param.sub_sizes, [1, 1, 1])
        self.assertTrue(param.structured)

    def test_init_with_structured_false_and_valid_sizes(self):
        param = DataSequencesParam(
            value=[[1, None], [3, 4]], sizes=[2, 3], sub_sizes=[2, 3], structured=False
        )
        self.assertEqual(param.value, [[1, None], [3, 4]])
        self.assertEqual(param.sizes, [2, 3])
        self.assertEqual(param.sub_sizes, [2, 3])
        self.assertFalse(param.structured)

    def test_init_with_structured_default_value(self):
        param = DataSequencesParam(value=[[1, None], [3, 4]])
        self.assertEqual(param.value, [[1, None], [3, 4]])
        self.assertIsNone(param.sizes)
        self.assertIsNone(param.sub_sizes)
        self.assertFalse(param.structured)

    def test_init_with_dict_initialization_and_structured(self):
        param = DataSequencesParam.model_validate(
            {
                "value": [[1, None], [3, 4]],
                "sizes": 2,
                "sub_sizes": [2, 2],
                "structured": True,
            }
        )
        self.assertEqual(param.value, [[1, None], [3, 4]])
        self.assertEqual(param.sizes, [2])
        self.assertEqual(param.sub_sizes, [2, 2])
        self.assertTrue(param.structured)

    def test_init_with_dict_initialization_structured_and_invalid_size(self):
        with self.assertRaises(ValidationError) as context:
            DataSequencesParam.model_validate(
                {
                    "value": [[1, None], [3, 4]],
                    "sizes": 3,
                    "sub_sizes": [2, 2],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected Sequence of sizes: [3], got size: 2 instead",
            str(context.exception),
        )

    def test_init_with_dict_initialization_structured_and_invalid_sub_size(self):
        with self.assertRaises(ValidationError) as context:
            DataSequencesParam.model_validate(
                {
                    "value": [[1, 2, None], [4, 5]],
                    "sizes": 2,
                    "sub_sizes": [3, 3],
                    "structured": True,
                }
            )
        self.assertIn(
            "Expected sub Sequences of sizes: [3, 3] got size: 2 at index=1",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
