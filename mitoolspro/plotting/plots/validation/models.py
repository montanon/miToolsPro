from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
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
from mitoolspro.plotting.plots.validation.functions import coerce_to_list

T = TypeVar("T")
NumericType: TypeAlias = float | int
NumericTuple: TypeAlias = tuple[NumericType, ...]


class Param[T](BaseModel):
    value: T

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)


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
            if isinstance(value, (list, tuple)):
                sequence_param = SequenceParam[T](value=value)
            elif isinstance(value, SequenceParam):
                sequence_param = value
            else:
                raise ArgumentValidationError(
                    f"Expected a SequenceParam[T] or raw sequence, got {type(value)}"
                )
            normalized.append(sequence_param)

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
            for idx, tup in enumerate(self.value):
                if len(tup) not in self.sizes:
                    raise ArgumentValidationError(
                        f"Invalid tuple length {len(tup)} at index {idx}. Allowed sizes: {self.sizes}."
                    )
        return self


class NumericTupleSequencesParam(Param[SequenceParam[SequenceParam[NumericTuple]]]):
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

        normalized_outer = []
        for outer_value in input_value:
            outer_value = coerce_to_list(outer_value)

            if not isinstance(outer_value, Sequence):
                raise ArgumentValidationError(
                    f"Expected each element to be a Sequence of tuples, got {type(outer_value)}"
                )

            normalized_inner = []
            for inner_value in outer_value:
                if not isinstance(inner_value, tuple):
                    raise ArgumentValidationError(
                        f"Expected a tuple, got {type(inner_value)}"
                    )
                normalized_inner.append(inner_value)

            inner_sequence_param = SequenceParam[NumericTuple](value=normalized_inner)
            normalized_outer.append(inner_sequence_param)

        outer_sequence_param = SequenceParam[SequenceParam[NumericTuple]](
            value=normalized_outer
        )

        return {"value": outer_sequence_param, "sizes": sizes}

    @model_validator(mode="after")
    def validate_numeric_tuple_sequences(self) -> "NumericTupleSequencesParam":
        if self.sizes is not None:
            if not isinstance(self.sizes, Sequence):
                self.sizes = [self.sizes]
            for outer_idx, inner_sequence_param in enumerate(self.value.value):
                for inner_idx, tup in enumerate(inner_sequence_param.value):
                    if len(tup) not in self.sizes:
                        raise ArgumentValidationError(
                            f"Invalid tuple length {len(tup)} at outer {outer_idx}, inner {inner_idx}. Allowed sizes: {self.sizes}."
                        )
        return self


if __name__ == "__main__":
    seq = SequenceParam[tuple](value=[(1, 2, 3), (1, 2, 3)])
