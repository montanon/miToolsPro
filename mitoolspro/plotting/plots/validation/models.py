from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from pandas import Series
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
BoolSequence = Sequence[bool]
BoolSequences = Sequence[BoolSequence]
BinsType: TypeAlias = int | str
BinsSequence = Sequence[BinsType]
BinsSequences = Sequence[BinsSequence]
DictSequence = Sequence[dict]
DictSequences = Sequence[DictSequence]
MarkerType = MarkerStyle | Path | str | dict
MarkerSequence = Sequence[MarkerType]
MarkerSequences = Sequence[MarkerSequence]
LiteralType = Literal["options"]
LiteralSequence = Sequence[LiteralType]
LiteralSequences = Sequence[LiteralSequence]
NumericType: TypeAlias = float | int
NumericSequence = Sequence[NumericType]
NumericSequences = Sequence[NumericSequence]
StrSequence = Sequence[str]
StrSequences = Sequence[StrSequence]
ColorType = Union[
    str,
    tuple[NumericType, NumericType, NumericType],  # RGB
    tuple[NumericType, NumericType, NumericType, NumericType],  # RGBA
    list[NumericType],
    int,
    float,
    None,
]
ColorSequence = Sequence[ColorType]
ColorSequences = Sequence[ColorSequence]
ColormapType = Union[Colormap, str]
ColormapSequence = Sequence[ColormapType]
ColormapSequences = Sequence[ColormapSequence]
EdgeColorType = Union[Literal["face"], ColorType]
EdgeColorSequence = Sequence[EdgeColorType]
EdgeColorSequences = Sequence[EdgeColorSequence]
NumericTupleType: TypeAlias = tuple[NumericType, ...]
NumericTupleSequence = Sequence[NumericTupleType]
NumericTupleSequences = Sequence[NumericTupleSequence]
NormalizationType = Union[Normalize, str]
NormalizationSequence = Sequence[NormalizationType]
NormalizationSequences = Sequence[NormalizationSequence]


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
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            values = values["value"]
        else:
            sizes = None
        values = coerce_to_list(values)
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected Sequence, got {type(values)}")
        if sizes is not None:
            sizes = sizes if isinstance(sizes, Sequence) else [sizes]
            if len(values) not in sizes:
                raise ArgumentValidationError(
                    f"Expected Sequence of sizes: {sizes}, got size: {len(values)} instead"
                )
        return {"value": values, "sizes": sizes}


class NumericSequenceParam(SequenceParam[NumericType]):
    pass


class StrSequenceParam(SequenceParam[str]):
    pass


class BoolSequenceParam(SequenceParam[bool]):
    pass


class DictSequenceParam(SequenceParam[dict]):
    pass


class RangeSequenceParam(SequenceParam[NumericType]):
    min_value: Optional[NumericType] = -np.inf
    max_value: Optional[NumericType] = np.inf
    strict: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]

            return {
                "value": values,
                "min_value": min_value,
                "max_value": max_value,
                "strict": strict,
            }
        return values

    @model_validator(mode="after")
    def validate_range_sequence(self) -> "RangeSequenceParam":
        for idx, value in enumerate(self.value):
            if not self.strict:
                if not (self.min_value <= value <= self.max_value):
                    raise ArgumentValidationError(
                        f"Value {value} at index {idx} is not in range [{self.min_value}, {self.max_value}]"
                    )
            else:
                if not (self.min_value < value < self.max_value):
                    raise ArgumentValidationError(
                        f"Value {value} at index {idx} is not in range ({self.min_value}, {self.max_value})"
                    )
        return self


class SequencesParam[T](Param[SequenceParam[SequenceParam[T]]]):
    value: Sequence[Sequence[T]]
    sizes: Optional[Sequence[int] | int] = None
    sub_sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}")

        if sizes is not None:
            sizes = sizes if isinstance(sizes, Sequence) else [sizes]
            if len(values) not in sizes:
                raise ArgumentValidationError(
                    f"Expected outer Sequence must be of sizes: {sizes}, got size: {len(values)}"
                )

        if sub_sizes is not None:
            sub_sizes = sub_sizes if isinstance(sub_sizes, Sequence) else [sub_sizes]

        normalized = []
        for idx, value in enumerate(values):
            value = coerce_to_list(value)
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ArgumentValidationError(
                    f"Expected a Sequence inside outer Sequence, got {type(value)} at index={idx}"
                )
            if sub_sizes is not None and len(value) not in sub_sizes:
                raise ArgumentValidationError(
                    f"Expected sub Sequences of sizes: {sub_sizes} got size: {len(value)} at index={idx}"
                )
            normalized.append(value)

        return {"value": normalized, "sizes": sizes, "sub_sizes": sub_sizes}


class NumericSequencesParam(SequencesParam[NumericType]):
    pass


class StrSequencesParam(SequencesParam[str]):
    pass


class BoolSequencesParam(SequencesParam[bool]):
    pass


class DictSequencesParam(SequencesParam[dict]):
    pass


class RangeSequencesParam(SequencesParam[NumericType]):
    min_value: Optional[NumericType] = -np.inf
    max_value: Optional[NumericType] = np.inf
    strict: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]
        else:
            values = values
            min_value = -np.inf
            max_value = np.inf
            strict = False

        values = coerce_to_list(values)
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}")

        normalized = []
        for value in values:
            value = coerce_to_list(value)
            if not isinstance(value, Sequence) or isinstance(value, str):
                raise ArgumentValidationError(f"Expected a Sequence, got {type(value)}")
            normalized.append(value)

        return {
            "value": normalized,
            "min_value": min_value,
            "max_value": max_value,
            "strict": strict,
        }

    @model_validator(mode="after")
    def validate_range_sequences(self) -> "RangeSequencesParam":
        for outer_idx, inner_sequence_param in enumerate(self.value):
            for inner_idx, value in enumerate(inner_sequence_param):
                if not self.strict:
                    if not (self.min_value <= value <= self.max_value):
                        raise ArgumentValidationError(
                            f"Value {value} at index [{outer_idx}, {inner_idx}] is not "
                            + f"in range [{self.min_value}, {self.max_value}]"
                        )
                else:
                    if not (self.min_value < value < self.max_value):
                        raise ArgumentValidationError(
                            f"Value {value} at index [{outer_idx}, {inner_idx}] is not "
                            + f"in range ({self.min_value}, {self.max_value})"
                        )
        return self


class NumericTupleParam(Param[NumericTupleType]):
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


class NumericTupleSequenceParam(SequenceParam[NumericTupleType]):
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            values = values["value"]

        if not isinstance(values, Sequence) or isinstance(values, str):
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


class NumericTupleSequencesParam(SequencesParam[NumericTupleType]):
    sizes: Optional[Sequence[int] | int] = None

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            values = values["value"]
        else:
            sizes = None

        values = coerce_to_list(values)
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}")

        normalized = []
        for value in values:
            value = coerce_to_list(value)
            if not isinstance(value, Sequence) or isinstance(value, str):
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
    value: ColorSequence

    @model_validator(mode="before")
    @classmethod
    def validate_color_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
    value: ColorSequences

    @model_validator(mode="before")
    @classmethod
    def validate_color_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
    value: EdgeColorSequence

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
    value: EdgeColorSequences

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
    value: Sequence[MarkerType]

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
    value: Sequence[Sequence[MarkerType]]

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
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
        values = values.get("value")

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence of strings."
            )

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}.")

        for idx, v in enumerate(values):
            if not is_literal(v, options):
                raise ArgumentValidationError(
                    f"Invalid literal at index {idx}: {v!r}. Allowed options: {options}."
                )

        return {"value": values, "options": options}


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
        values = values.get("value")

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence."
            )

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}.")

        normalized_outer = []
        for outer_idx, outer in enumerate(values):
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


class NormalizationParam(Param[NormalizationType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        if isinstance(values, Normalize):
            return {"value": values}

        if not is_literal(values, NORMALIZATIONS):
            raise ArgumentValidationError(
                f"Invalid literal: {values!r}. Allowed options: {NORMALIZATIONS}."
            )

        return {"value": values}


class NormalizationSequenceParam(SequenceParam[NormalizationType]):
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


class NormalizationSequencesParam(SequencesParam[NormalizationType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

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


class BinsParam(Param[BinsType]):
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


class BinsSequenceParam(SequenceParam[BinsType]):
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


class BinsSequencesParam(SequencesParam[BinsType]):
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
