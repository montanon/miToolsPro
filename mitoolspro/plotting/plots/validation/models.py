from collections.abc import Sequence
from typing import Any, Literal, Optional, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from pandas import Series
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from mitoolspro.exceptions import (
    ArgumentValidationError,
)
from mitoolspro.plotting.plots.matplotlib_typing import (
    BINS,
    CMAPS,
    NORMALIZATIONS,
)
from mitoolspro.plotting.plots.validation.functions import (
    coerce_to_list,
    is_bins,
    is_color,
    is_literal,
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
NormalizeType = Union[Normalize, str]
ColormapType = Union[Colormap, str]


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


class MarkerSequenceParam(SequenceParam[MarkerParam]):
    value: Sequence[MarkerParam]

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence of markers, got {type(values)}"
            )

        normalized = []
        for idx, v in enumerate(values):
            if not is_marker(v):
                raise ArgumentValidationError(
                    f"Invalid marker format at index {idx}: {v!r}"
                )
            normalized.append(v)

        return {"value": normalized}


class MarkerSequencesParam(SequencesParam[MarkerParam]):
    value: Sequence[Sequence[MarkerParam]]

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequences(cls, values: Any) -> dict:
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
                if not is_marker(v):
                    raise ArgumentValidationError(
                        f"Invalid marker format at outer {outer_idx}, inner {inner_idx}: {v!r}"
                    )
                normalized_inner.append(v)

            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}


class LiteralParam(StrParam):
    options: Sequence[str] = Field(..., description="Allowed options for the literal.")

    @model_validator(mode="before")
    @classmethod
    def validate_literal(cls, values: Any) -> dict:
        if not isinstance(values, dict):
            raise ArgumentValidationError(
                "Expected dict input with 'value' and 'options' keys."
            )

        options = values.get("options")
        value = values.get("value")

        if options is None or not isinstance(options, Sequence) or not options:
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence."
            )

        if not is_literal(value, options):
            raise ArgumentValidationError(
                f"Invalid literal: {value!r}. Allowed options: {options}."
            )

        return {"value": value, "options": options}


class LiteralSequenceParam(SequenceParam[str]):
    options: Optional[Sequence[str]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_literal_sequence(cls, values: Any) -> dict:
        if not isinstance(values, dict):
            raise ArgumentValidationError(
                "Expected dict input with 'value' and 'options' keys."
            )

        options = values.get("options")
        input_value = values.get("value")

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence of strings."
            )

        input_value = coerce_to_list(input_value)

        if not isinstance(input_value, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence, got {type(input_value)}."
            )

        for idx, v in enumerate(input_value):
            if not is_literal(v, options):
                raise ArgumentValidationError(
                    f"Invalid literal at index {idx}: {v!r}. Allowed options: {options}."
                )

        return {"value": input_value, "options": options}


class LiteralSequencesParam(SequencesParam[str]):
    options: Optional[Sequence[str]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_literal_sequences(cls, values: Any) -> dict:
        if not isinstance(values, dict):
            raise ArgumentValidationError(
                "Expected dict input with 'value' and 'options' keys."
            )

        options = values.get("options")
        input_value = values.get("value")

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence."
            )

        input_value = coerce_to_list(input_value)

        if not isinstance(input_value, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence, got {type(input_value)}."
            )

        normalized_outer = []
        for outer_idx, outer in enumerate(input_value):
            outer = coerce_to_list(outer)

            if not isinstance(outer, Sequence):
                raise ArgumentValidationError(
                    f"Expected a Sequence at outer index {outer_idx}, got {type(outer)}."
                )

            normalized_inner = []
            for inner_idx, v in enumerate(outer):
                if not is_literal(v, options):
                    raise ArgumentValidationError(
                        f"Invalid literal at [{outer_idx}, {inner_idx}]: {v!r}. Allowed options: {options}."
                    )
                normalized_inner.append(v)

            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer, "options": options}


class NormalizationParam(Param[NormalizeType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization(cls, values: Any) -> dict:
        if isinstance(values, dict):
            value = values.get("value")
        else:
            value = values

        if isinstance(value, Normalize):
            return {"value": value}

        if not is_literal(value, NORMALIZATIONS):
            raise ArgumentValidationError(
                f"Invalid literal: {value!r}. Allowed options: {NORMALIZATIONS}."
            )

        return {"value": value}


class NormalizationSequenceParam(SequenceParam[NormalizeType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        values = coerce_to_list(values)

        normalized = []
        for idx, value in enumerate(values):
            if isinstance(value, Normalize):
                normalized.append(value)
            elif is_literal(value, NORMALIZATIONS):
                normalized.append(value)
            else:
                raise ArgumentValidationError(
                    f"Invalid normalization at index {idx}: {value!r}. Allowed options: {NORMALIZATIONS}."
                )

        return {"value": normalized}


class NormalizationSequencesParam(SequencesParam[NormalizeType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")
        else:
            values = values

        values = coerce_to_list(values)

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)

            normalized_inner = []
            for inner_idx, value in enumerate(outer):
                if isinstance(value, Normalize):
                    normalized_inner.append(value)
                elif is_literal(value, NORMALIZATIONS):
                    normalized_inner.append(value)
                else:
                    raise ArgumentValidationError(
                        f"Invalid normalization at [{outer_idx}, {inner_idx}]: {value!r}. Allowed options: {NORMALIZATIONS}."
                    )
            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}


class ColormapParam(Param[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        if isinstance(values, Colormap):
            return {"value": values}

        if not is_literal(values, CMAPS):
            raise ArgumentValidationError(
                f"Invalid colormap: {values!r}. Allowed options: {CMAPS}."
            )

        return {"value": values}


class ColormapSequenceParam(SequenceParam[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        values = coerce_to_list(values)

        normalized = []
        for idx, value in enumerate(values):
            if isinstance(value, Colormap):
                normalized.append(value)
            elif is_literal(value, CMAPS):
                normalized.append(value)
            else:
                raise ArgumentValidationError(
                    f"Invalid colormap at index {idx}: {value!r}. Allowed options: {CMAPS}."
                )

        return {"value": normalized}


class ColormapSequencesParam(SequencesParam[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        values = coerce_to_list(values)

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)

            normalized_inner = []
            for inner_idx, value in enumerate(outer):
                if isinstance(value, Colormap):
                    normalized_inner.append(value)
                elif is_literal(value, CMAPS):
                    normalized_inner.append(value)
                else:
                    raise ArgumentValidationError(
                        f"Invalid colormap at [{outer_idx}, {inner_idx}]: {value!r}. Allowed options: {CMAPS}."
                    )
            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}


class BinsParam(Param[int | str]):
    @model_validator(mode="before")
    @classmethod
    def validate_bins(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        if not is_bins(values):
            raise ArgumentValidationError(
                f"Invalid bins value: {values!r}. Must be a positive integer or one of {BINS}."
            )

        return {"value": values}


class BinsSequenceParam(SequenceParam[int | str]):
    @model_validator(mode="before")
    @classmethod
    def validate_bins_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        values = coerce_to_list(values)

        normalized = []
        for idx, value in enumerate(values):
            if is_bins(value):
                normalized.append(value)
            else:
                raise ArgumentValidationError(
                    f"Invalid bins value at index {idx}: {value!r}. Must be a positive integer or one of {BINS}."
                )

        return {"value": normalized}


class BinsSequencesParam(SequencesParam[int | str]):
    @model_validator(mode="before")
    @classmethod
    def validate_bins_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        values = coerce_to_list(values)

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)

            normalized_inner = []
            for inner_idx, value in enumerate(outer):
                if is_bins(value):
                    normalized_inner.append(value)
                else:
                    raise ArgumentValidationError(
                        f"Invalid bins value at [{outer_idx}, {inner_idx}]: {value!r}. Must be a positive integer or one of {BINS}."
                    )
            normalized_outer.append(normalized_inner)

        return {"value": normalized_outer}
