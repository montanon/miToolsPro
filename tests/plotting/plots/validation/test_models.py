import unittest
from unittest import TestCase

from pydantic import ValidationError

from mitoolspro.plotting.plots.validation.models import Param


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


if __name__ == "__main__":
    unittest.main()
