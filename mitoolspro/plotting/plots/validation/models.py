from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
from pandas import Series
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import TypeGuard

from mitoolspro.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValidationError,
)
from mitoolspro.plotting.plots.matplotlib_typing import (
    BINS,
    CMAPS,
    COLORS,
    MARKERS,
    MARKERS_FILLSTYLES,
    NORMALIZATIONS,
    NumericSequences,
)
from mitoolspro.plotting.plots.validation.functions import (
    coerce_to_list,
    is_color,
    is_marker,
    normalize_rgb_tuple,
)

T = TypeVar("T")
NumericType: TypeAlias = float | int
NumericTuple: TypeAlias = tuple[NumericType, ...]
ColorType = Union[
    str,
    tuple[NumericType, NumericType, NumericType],  # RGB
    tuple[NumericType, NumericType, NumericType, NumericType],  # RGBA
    list[NumericType],
    int,
    float,
    None,
]
EdgeColorType = Union[Literal["face"], ColorType]


class Param[T](BaseModel):
    value: T

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True,
    )


class RangeParam(Param[NumericType]):
    min_value: NumericType = -np.inf
    max_value: NumericType = np.inf
    strict: bool = False

    @model_validator(mode="after")
    def validate_range(self) -> "RangeParam":
        if not isinstance(self.value, (int, float)):
            raise ArgumentValidationError(
                f"Expected numeric {self.value=}, got {type(self.value)}"
            )
        if not self.strict:
            if not (self.min_value <= self.value <= self.max_value):
                raise ArgumentValidationError(
                    f"Value {self.value} is not in range [{self.min_value}, {self.max_value}]"
                )
        else:
            if not (self.min_value < self.value < self.max_value):
                raise ArgumentValidationError(
                    f"Value {self.value} is not in range ({self.min_value}, {self.max_value})"
                )
        return self


class StrParam(Param[str]):
    pass


class BoolParam(Param[bool]):
    pass


class NumericParam(Param[NumericType]):
    pass


class DictParam(Param[dict]):
    pass


class SequenceParam[T](Param[Sequence[T]]):
    value: Sequence[T]

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]
        values = coerce_to_list(values)
        if not isinstance(values, Sequence):
            raise ArgumentValidationError(f"Expected Sequence, got {type(values)}")
        return {"value": values}


class NumericSequenceParam(SequenceParam[NumericType]):
    pass


class StrSequenceParam(SequenceParam[str]):
    pass


class BoolSequenceParam(SequenceParam[bool]):
    pass


class DictSequenceParam(SequenceParam[dict]):
    pass


class SequencesParam[T](Param[SequenceParam[SequenceParam[T]]]):
    value: Sequence[Sequence[T]]

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            input_value = values["value"]
        else:
            input_value = values

        input_value = coerce_to_list(input_value)

        if not isinstance(input_value, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence, got {type(input_value)}"
            )

        normalized = []
        for value in input_value:
            value = coerce_to_list(value)
            if not isinstance(value, Sequence):
                raise ArgumentValidationError(
                    f"Expected a Sequence inside outer list, got {type(value)}"
                )
            normalized.append(value)

        return {"value": normalized}


class NumericSequencesParam(SequencesParam[NumericType]):
    pass


class StrSequencesParam(SequencesParam[str]):
    pass


class BoolSequencesParam(SequencesParam[bool]):
    pass


class DictSequencesParam(SequencesParam[dict]):
    pass


class NumericTupleParam(Param[NumericTuple]):
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="after")
    def validate_numeric_tuple(self) -> "NumericTupleParam":
        if self.sizes is not None:
            if not isinstance(self.sizes, Sequence):
                self.sizes = [self.sizes]
            if not all(size > 0 for size in self.sizes):
                raise ArgumentValidationError(
                    f"All sizes must be positive, got {self.sizes}."
                )
            if len(self.value) not in self.sizes:
                raise ArgumentValidationError(
                    f"Invalid tuple length {len(self.value)}. Allowed sizes: {self.sizes}."
                )
        return self


class NumericTupleSequenceParam(SequenceParam[NumericTuple]):
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            values = values["value"]

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}")

        for idx, v in enumerate(values):
            if not isinstance(v, tuple):
                raise ArgumentValidationError(
                    f"Expected each element to be a tuple, got {type(v)} at index {idx}"
                )

        return {"value": values, "sizes": sizes}

    @model_validator(mode="after")
    def validate_numeric_tuple_sequence(self) -> "NumericTupleSequenceParam":
        if self.sizes is not None:
            if not isinstance(self.sizes, Sequence):
                self.sizes = [self.sizes]
            if not all(size > 0 for size in self.sizes):
                raise ArgumentValidationError(
                    f"All sizes must be positive, got {self.sizes}."
                )
            for idx, tup in enumerate(self.value):
                if len(tup) not in self.sizes:
                    raise ArgumentValidationError(
                        f"Invalid tuple length {len(tup)} at index {idx}. Allowed sizes: {self.sizes}."
                    )
        return self


class NumericTupleSequencesParam(SequencesParam[NumericTuple]):
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            input_value = values["value"]
            sizes = values.get("sizes", None)
        else:
            input_value = values
            sizes = None

        input_value = coerce_to_list(input_value)
        if not isinstance(input_value, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence, got {type(input_value)}"
            )

        normalized = []
        for value in input_value:
            value = coerce_to_list(value)
            if not isinstance(value, Sequence):
                raise ArgumentValidationError(f"Expected a Sequence, got {type(value)}")
            normalized.append(value)

        return {"value": normalized, "sizes": sizes}

    @model_validator(mode="after")
    def validate_numeric_tuple_sequences(self) -> "NumericTupleSequencesParam":
        if self.sizes is not None:
            if not isinstance(self.sizes, Sequence):
                self.sizes = [self.sizes]
            if not all(size > 0 for size in self.sizes):
                raise ArgumentValidationError(
                    f"All sizes must be positive, got {self.sizes}."
                )
            for outer_idx, inner_sequence_param in enumerate(self.value):
                for inner_idx, tup in enumerate(inner_sequence_param):
                    if len(tup) not in self.sizes:
                        raise ArgumentValidationError(
                            f"Invalid tuple length {len(tup)} at outer {outer_idx}, inner {inner_idx}. Allowed sizes: {self.sizes}."
                        )
        return self


def validate_single_color(
    value: Any, allow_face_literal: bool = False
) -> ColorType | Literal["face"]:
    if isinstance(value, (np.ndarray, Series)):
        value = value.tolist()

    if allow_face_literal and value == "face":
        return value

    value = normalize_rgb_tuple(value)

    if not is_color(value):
        raise ArgumentValidationError(f"Invalid color format: {value!r}")

    return value


class ColorParam(Param[ColorType]):
    value: ColorType

    @model_validator(mode="before")
    @classmethod
    def validate_color(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]
        values = validate_single_color(values)

        return {"value": values}


class ColorSequenceParam(SequenceParam[ColorType]):
    value: Sequence[ColorType]

    @model_validator(mode="before")
    @classmethod
    def validate_color_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence of colors, got {type(values)}"
            )

        normalized = []
        for idx, v in enumerate(values):
            try:
                v = validate_single_color(v)
            except ArgumentValidationError:
                raise ArgumentValidationError(
                    f"Invalid color format: {v!r} at index {idx}"
                )
            normalized.append(v)

        return {"value": normalized}


class ColorSequencesParam(SequencesParam[ColorType]):
    value: Sequence[Sequence[ColorType]]

    @model_validator(mode="before")
    @classmethod
    def validate_color_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence of Sequences, got {type(values)}"
            )

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)

            if not isinstance(outer, Sequence):
                raise ArgumentValidationError(
                    f"Expected a Sequence inside outer list at index {outer_idx}, got {type(outer)}"
                )

            normalized_inner = []
            for inner_idx, v in enumerate(outer):
                try:
                    v = validate_single_color(v)
                except ArgumentValidationError:
                    raise ArgumentValidationError(
                        f"Invalid color format: {v!r} at index [{outer_idx}, {inner_idx}]"
                    )

                normalized_inner.append(v)

            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}


class EdgeColorParam(Param[EdgeColorType]):
    value: EdgeColorType

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        value = validate_single_color(values, allow_face_literal=True)
        return {"value": value}


class EdgeColorSequenceParam(SequenceParam[EdgeColorType]):
    value: Sequence[EdgeColorType]

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence of colors, got {type(values)}"
            )

        normalized = []
        for idx, v in enumerate(values):
            try:
                v = validate_single_color(v, allow_face_literal=True)
            except ArgumentValidationError:
                raise ArgumentValidationError(
                    f"Invalid color format: {v!r} at index {idx}"
                )
            normalized.append(v)

        return {"value": normalized}


class EdgeColorSequencesParam(SequencesParam[EdgeColorType]):
    value: Sequence[Sequence[EdgeColorType]]

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence of Sequences, got {type(values)}"
            )

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)
            if not isinstance(outer, Sequence):
                raise ArgumentValidationError(
                    f"Expected a Sequence inside outer list at index {outer_idx}, got {type(outer)}"
                )

            normalized_inner = []
            for inner_idx, v in enumerate(outer):
                try:
                    v = validate_single_color(v, allow_face_literal=True)
                except ArgumentValidationError:
                    raise ArgumentValidationError(
                        f"Invalid color format: {v!r} at index [{outer_idx}, {inner_idx}]"
                    )
                normalized_inner.append(v)

            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}


class MarkerParam(Param[Any]):
    value: Any

    @model_validator(mode="before")
    @classmethod
    def validate_marker(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value", values)

        if not is_marker(values):
            raise ArgumentValidationError(f"Invalid marker format: {values!r}")

        return {"value": values}
