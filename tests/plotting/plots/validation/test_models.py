import unittest
from unittest import TestCase

from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.validation.models import Param, RangeParam


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


if __name__ == "__main__":
    unittest.main()
