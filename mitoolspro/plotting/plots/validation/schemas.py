from typing import Any

from numpy import ndarray
from pandas import Series
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


def numpy_array_schema(handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    def validate_numpy_array(value: Any) -> ndarray:
        if not isinstance(value, ndarray):
            raise ValueError(f"Expected numpy array, got {type(value)}")
        return value

    return core_schema.no_info_after_validator_function(
        validate_numpy_array,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist()
        ),
    )


class NumpyArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return numpy_array_schema(handler)


def pandas_series_schema(handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
    def validate_pandas_series(value: Any) -> Series:
        if not isinstance(value, Series):
            raise ValueError(f"Expected pandas Series, got {type(value)}")
        return value

    return core_schema.no_info_after_validator_function(
        validate_pandas_series,
        core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda x: x.tolist()
        ),
    )


class PandasSeries:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return pandas_series_schema(handler)
