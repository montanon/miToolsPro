from collections.abc import Sequence
from typing import Any, Literal, Optional, TypeVar

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.transforms import Transform
from pandas import Series
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.matplotlib_typing import BINS, CMAPS, NORMALIZATIONS
from mitoolspro.plotting.plots.validation.functions import (
    coerce_to_list,
    is_bins,
    is_literal,
    is_marker,
    standardize_sequences,
    validate_numeric,
    validate_range,
    validate_sequence,
    validate_sequence_range,
    validate_sequence_sizes,
    validate_sequences_range,
    validate_sequences_sizes,
    validate_single_color,
    validate_tuple_sequence,
    validate_tuple_sequence_sizes,
    validate_tuple_sequences,
    validate_tuple_sequences_sizes,
    validate_tuple_sizes,
)
from mitoolspro.plotting.plots.validation.types import (
    BinsType,
    ColormapType,
    ColorSequence,
    ColorSequences,
    ColorType,
    EdgeColorSequence,
    EdgeColorSequences,
    EdgeColorType,
    MarkerSequence,
    MarkerSequences,
    NormalizationType,
    NumericTupleType,
    NumericType,
    SizesType,
    StrSequence,
    StrSequences,
)

T = TypeVar("T")


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
        validate_numeric(self.value)
        validate_range(self.value, self.max_value, self.min_value, self.strict)
        return self


class StrParam(Param[str]):
    pass


class BoolParam(Param[bool]):
    pass


class NumericParam(Param[NumericType]):
    pass


class DictParam(Param[dict]):
    pass


class TransformParam(Param[Transform]):
    pass


class SequenceParam[T](Param[Sequence[T]]):
    value: Sequence[T]
    sizes: Optional[SizesType] = None

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            values = values["value"]
        else:
            sizes = None
        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes)
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
            sizes = values.get("sizes", None)
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]

        return {
            "value": values,
            "sizes": sizes,
            "min_value": min_value,
            "max_value": max_value,
            "strict": strict,
        }

    @model_validator(mode="after")
    def validate_range_sequence(self) -> "RangeSequenceParam":
        validate_sequence_range(self.value, self.min_value, self.max_value, self.strict)
        return self


class SequencesParam[T](Param[SequenceParam[SequenceParam[T]]]):
    value: Sequence[Sequence[T]]
    sizes: Optional[SizesType] = None
    sub_sizes: Optional[SizesType] = None

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            values = values["value"]

        values = coerce_to_list(values)
        validate_sequence(values)

        sizes = validate_sequence_sizes(values, sizes)
        values = standardize_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes)

        return {"value": values, "sizes": sizes, "sub_sizes": sub_sizes}


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
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            min_value = -np.inf
            max_value = np.inf
            strict = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes)
        values = standardize_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes)

        return {
            "value": values,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "min_value": min_value,
            "max_value": max_value,
            "strict": strict,
        }

    @model_validator(mode="after")
    def validate_range_sequences(self) -> "RangeSequencesParam":
        validate_sequences_range(
            self.value, self.min_value, self.max_value, self.strict
        )
        return self


class NumericTupleParam(Param[NumericTupleType]):
    tuple_sizes: Optional[SizesType] = None

    @model_validator(mode="after")
    def validate_numeric_tuple(self) -> "NumericTupleParam":
        self.tuple_sizes = validate_tuple_sizes(self.value, self.tuple_sizes)
        return self


class NumericTupleSequenceParam(SequenceParam[NumericTupleType]):
    tuple_sizes: Optional[SizesType] = None

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            tuple_sizes = values.get("tuple_sizes", None)
            values = values["value"]
        else:
            sizes = None
            tuple_sizes = None

        values = coerce_to_list(values)
        validate_sequence(values)
        validate_tuple_sequence(values)
        sizes = validate_sequence_sizes(values, sizes)
        return {"value": values, "sizes": sizes, "tuple_sizes": tuple_sizes}

    @model_validator(mode="after")
    def validate_numeric_tuple_sequence(self) -> "NumericTupleSequenceParam":
        self.tuple_sizes = validate_tuple_sequence_sizes(self.value, self.tuple_sizes)
        return self


class NumericTupleSequencesParam(SequencesParam[NumericTupleType]):
    tuple_sizes: Optional[SizesType] = None

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            tuple_sizes = values.get("tuple_sizes", None)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            tuple_sizes = None

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes)
        values = standardize_sequences(values)
        validate_tuple_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes)

        return {
            "value": values,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "tuple_sizes": tuple_sizes,
        }

    @model_validator(mode="after")
    def validate_numeric_tuple_sequences(self) -> "NumericTupleSequencesParam":
        self.tuple_sizes = validate_tuple_sequences_sizes(self.value, self.tuple_sizes)
        return self


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
        validate_sequence(values)

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

            if not isinstance(outer, Sequence) or isinstance(outer, str):
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
        validate_sequence(values)

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
        validate_sequence(values)

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
    value: MarkerSequence

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)
        validate_sequence(values)

        normalized = []
        for idx, v in enumerate(values):
            if not is_marker(v):
                raise ArgumentValidationError(
                    f"Invalid marker format at index {idx}: {v!r}"
                )
            normalized.append(v)

        return {"value": normalized}


class MarkerSequencesParam(SequencesParam[MarkerParam]):
    value: MarkerSequences

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]

        values = coerce_to_list(values)
        validate_sequence(values)

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
    options: StrSequence = Field(..., description="Allowed options for the literal.")

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
    options: Optional[StrSequence] = None

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
    options: Optional[StrSequences] = None

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
        validate_sequence(values)

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
        validate_sequence(values)

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
        validate_sequence(values)

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
        validate_sequence(values)

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
        validate_sequence(values)
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
        validate_sequence(values)

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
