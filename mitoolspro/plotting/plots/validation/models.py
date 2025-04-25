import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
from pandas import Series
from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    GetCoreSchemaHandler,
    field_validator,
)
from pydantic_core import core_schema
from typing_extensions import TypeGuard

from mitoolspro.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValueError,
)
from mitoolspro.plotting.plots.matplotlib_typing import (
    BINS,
    CMAPS,
    COLORS,
    MARKERS,
    MARKERS_FILLSTYLES,
    NORMALIZATIONS,
    NumericSequences,
    NumericType,
)


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


class NumpyArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return numpy_array_schema(handler)


class PandasSeries:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return pandas_series_schema(handler)


def is_numpy_array(value: Any) -> TypeGuard[ndarray]:
    return isinstance(value, ndarray)


def is_pandas_series(value: Any) -> TypeGuard[Series]:
    return isinstance(value, Series)


def is_sequence_type(value: Any) -> TypeGuard[Union[list, tuple, ndarray, Series]]:
    return isinstance(value, (list, tuple, ndarray, Series))


def convert_to_list(value: Union[list, tuple, ndarray, Series]) -> list:
    if isinstance(value, ndarray):
        return value.tolist()
    if isinstance(value, Series):
        return value.tolist()
    return list(value)


# Type aliases with validators
Numeric: TypeAlias = Union[int, float]
String: TypeAlias = str
Boolean: TypeAlias = bool
Dictionary: TypeAlias = dict

# Sequence type aliases with validators
SequenceType: TypeAlias = Annotated[
    Union[list, tuple, NumpyArray, PandasSeries], BeforeValidator(convert_to_list)
]
NumericSequence: TypeAlias = Annotated[
    Sequence[Numeric], BeforeValidator(convert_to_list)
]
StringSequence: TypeAlias = Annotated[
    Sequence[String], BeforeValidator(convert_to_list)
]
BooleanSequence: TypeAlias = Annotated[
    Sequence[Boolean], BeforeValidator(convert_to_list)
]
DictionarySequence: TypeAlias = Annotated[
    Sequence[Dictionary], BeforeValidator(convert_to_list)
]

# Nested sequence type aliases
NumericSequences: TypeAlias = Annotated[
    Sequence[NumericSequence], BeforeValidator(convert_to_list)
]
StringSequences: TypeAlias = Annotated[
    Sequence[StringSequence], BeforeValidator(convert_to_list)
]
BooleanSequences: TypeAlias = Annotated[
    Sequence[BooleanSequence], BeforeValidator(convert_to_list)
]
DictionarySequences: TypeAlias = Annotated[
    Sequence[DictionarySequence], BeforeValidator(convert_to_list)
]

# Color type aliases
ColorTuple: TypeAlias = (
    tuple[Numeric, Numeric, Numeric] | tuple[Numeric, Numeric, Numeric, Numeric]
)
ColorHex: TypeAlias = str  # Format: #RRGGBB or #RRGGBBAA
ColorName: TypeAlias = Literal[*COLORS]
Color: TypeAlias = Union[ColorTuple, ColorHex, ColorName, Numeric, None]

# Marker type aliases
MarkerName: TypeAlias = Literal[*MARKERS]
MarkerStyleName: TypeAlias = Literal[*MARKERS_FILLSTYLES]
MarkerTransform: TypeAlias = Union[str, Normalize]
MarkerCapStyle: TypeAlias = Literal["butt", "round", "projecting"]
MarkerJoinStyle: TypeAlias = Literal["miter", "round", "bevel"]


class BaseParam(BaseModel):
    def validate(self) -> None:
        raise NotImplementedError


class NumericParam(BaseParam):
    value: Numeric = Field(..., description="Numeric value")

    @field_validator("value")
    @classmethod
    def validate_numeric(cls, v: Any) -> Numeric:
        if not isinstance(v, (int, float)):
            raise ArgumentTypeError(f"Value must be numeric, got {type(v)}")
        return v


class StringParam(BaseParam):
    value: String = Field(..., description="String value")

    @field_validator("value")
    @classmethod
    def validate_string(cls, v: Any) -> String:
        if not isinstance(v, str):
            raise ArgumentTypeError(f"Value must be string, got {type(v)}")
        return v


class BooleanParam(BaseParam):
    value: Boolean = Field(..., description="Boolean value")

    @field_validator("value")
    @classmethod
    def validate_boolean(cls, v: Any) -> Boolean:
        if not isinstance(v, bool):
            raise ArgumentTypeError(f"Value must be boolean, got {type(v)}")
        return v


class DictionaryParam(BaseParam):
    value: Dictionary = Field(..., description="Dictionary value")

    @field_validator("value")
    @classmethod
    def validate_dictionary(cls, v: Any) -> Dictionary:
        if not isinstance(v, dict):
            raise ArgumentTypeError(f"Value must be dictionary, got {type(v)}")
        return v


class SequenceParam(BaseParam):
    value: SequenceType = Field(..., description="Sequence of values")

    @field_validator("value")
    @classmethod
    def validate_sequence(cls, v: Any) -> list:
        if not is_sequence_type(v):
            raise ArgumentTypeError(f"Value must be a sequence, got {type(v)}")
        return convert_to_list(v)


class NumericSequenceParam(SequenceParam):
    value: NumericSequence = Field(..., description="Sequence of numeric values")

    @field_validator("value")
    @classmethod
    def validate_numeric_sequence(cls, v: Any) -> list[Numeric]:
        v = super().validate_sequence(v)
        for item in v:
            if not isinstance(item, (int, float)) and item is not None:
                raise ArgumentTypeError(f"All items must be numeric, got {type(item)}")
        return v


class StringSequenceParam(SequenceParam):
    value: StringSequence = Field(..., description="Sequence of string values")

    @field_validator("value")
    @classmethod
    def validate_string_sequence(cls, v: Any) -> list[String]:
        v = super().validate_sequence(v)
        for item in v:
            if not isinstance(item, str) and item is not None:
                raise ArgumentTypeError(f"All items must be strings, got {type(item)}")
        return v


class BooleanSequenceParam(SequenceParam):
    value: BooleanSequence = Field(..., description="Sequence of boolean values")

    @field_validator("value")
    @classmethod
    def validate_boolean_sequence(cls, v: Any) -> list[Boolean]:
        v = super().validate_sequence(v)
        for item in v:
            if not isinstance(item, bool) and item is not None:
                raise ArgumentTypeError(f"All items must be booleans, got {type(item)}")
        return v


class DictionarySequenceParam(SequenceParam):
    value: DictionarySequence = Field(..., description="Sequence of dictionary values")

    @field_validator("value")
    @classmethod
    def validate_dictionary_sequence(cls, v: Any) -> list[Dictionary]:
        v = super().validate_sequence(v)
        for item in v:
            if not isinstance(item, dict) and item is not None:
                raise ArgumentTypeError(
                    f"All items must be dictionaries, got {type(item)}"
                )
        return v


class SequencesParam(BaseParam):
    value: Sequence[SequenceType] = Field(..., description="Sequence of sequences")

    @field_validator("value")
    @classmethod
    def validate_sequences(cls, v: Any) -> list[list]:
        if not is_sequence_type(v):
            raise ArgumentTypeError(f"Value must be a sequence, got {type(v)}")
        for seq in v:
            if not is_sequence_type(seq):
                raise ArgumentTypeError(f"All items must be sequences, got {type(seq)}")
        return [convert_to_list(seq) for seq in v]


class NumericSequencesParam(SequencesParam):
    value: NumericSequences = Field(..., description="Sequence of numeric sequences")

    @field_validator("value")
    @classmethod
    def validate_numeric_sequences(cls, v: Any) -> list[list[Numeric]]:
        v = super().validate_sequences(v)
        for seq in v:
            for item in seq:
                if not isinstance(item, (int, float)) and item is not None:
                    raise ArgumentTypeError(
                        f"All items must be numeric, got {type(item)}"
                    )
        return v


class StringSequencesParam(SequencesParam):
    value: StringSequences = Field(..., description="Sequence of string sequences")

    @field_validator("value")
    @classmethod
    def validate_string_sequences(cls, v: Any) -> list[list[String]]:
        v = super().validate_sequences(v)
        for seq in v:
            for item in seq:
                if not isinstance(item, str) and item is not None:
                    raise ArgumentTypeError(
                        f"All items must be strings, got {type(item)}"
                    )
        return v


if __name__ == "__main__":
    # Test basic params
    NumericParam(value=3.14)
    StringParam(value="abc")
    BooleanParam(value=True)
    DictionaryParam(value={"key": "value"})

    # Test sequence params
    NumericSequenceParam(value=[1, 2, 3])
    StringSequenceParam(value=["a", "b", "c"])
    BooleanSequenceParam(value=[True, False, True])
    DictionarySequenceParam(value=[{"a": 1}, {"b": 2}])

    # Test sequences of sequences
    NumericSequencesParam(value=[[1.0, 2.0], [3.0]])
    StringSequencesParam(value=[["a", "b"], ["c", "d"]])
