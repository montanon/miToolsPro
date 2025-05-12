from collections.abc import Sequence
from typing import Any, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.transforms import Transform
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mitoolspro.exceptions import ArgumentValidationError
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
    BINS,
    COLORMAPS,
    NORMALIZATIONS,
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
        if self.min_value is None:
            self.min_value = -np.inf
        if self.max_value is None:
            self.max_value = np.inf
        validate_range(self.value, self.max_value, self.min_value, self.strict)
        return self


class StrParam(Param[str]):
    pass


class BoolParam(Param[bool]):
    pass


class NumericParam(Param[NumericType | None]):
    pass


class DictParam(Param[dict | None]):
    pass


class TransformParam(Param[Transform]):
    pass


class FloatParam(Param[float]):
    pass


class IntParam(Param[int]):
    pass


class NumStrParam(Param[int | str]):
    pass


class SequenceParam[T](Param[Sequence[T]]):
    value: Sequence[T]
    sizes: Optional[SizesType] = None
    structured: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            structured = False
        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        return {"value": values, "sizes": sizes, "structured": structured}


class NumericSequenceParam(SequenceParam[NumericType | None]):
    pass


class StrSequenceParam(SequenceParam[str]):
    pass


class BoolSequenceParam(SequenceParam[bool]):
    pass


class DictSequenceParam(SequenceParam[dict | None]):
    pass


class DataSequenceParam(SequenceParam[NumericType | None]):
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
            structured = values.get("structured", False)
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]
        else:
            sizes = None
            structured = False
            min_value = -np.inf
            max_value = np.inf
            strict = False

        if min_value is None:
            min_value = -np.inf
        if max_value is None:
            max_value = np.inf

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        return {
            "value": values,
            "sizes": sizes,
            "structured": structured,
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
    structured: Optional[bool] = False

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)

        sizes = validate_sequence_sizes(values, sizes, structured)
        values = standardize_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes, structured)

        return {
            "value": values,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


class NumericSequencesParam(SequencesParam[NumericType | None]):
    pass


class StrSequencesParam(SequencesParam[str]):
    pass


class BoolSequencesParam(SequencesParam[bool]):
    pass


class DictSequencesParam(SequencesParam[dict | None]):
    pass


class DataSequencesParam(SequencesParam[NumericType | None]):
    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)

        if not isinstance(values, Sequence) or isinstance(values, str):
            raise ArgumentValidationError(f"Expected a Sequence, got {type(values)}")
        if all(isinstance(v, NumericType | None) for v in values):
            raise ArgumentValidationError(
                "Expected at least one Sequence inside outer Sequence."
            )

        standardized = []
        for outer_idx, outer in enumerate(values):
            # Handle single point Sequence
            if isinstance(outer, NumericType | None):
                outer = [outer]
            outer = coerce_to_list(outer)
            if not isinstance(outer, Sequence) or isinstance(outer, str):
                raise ArgumentValidationError(
                    f"Expected a Sequence at index {outer_idx}, got {type(outer)}"
                )

            for inner_idx, v in enumerate(outer):
                if v is not None and not isinstance(v, (int, float)):
                    raise ArgumentValidationError(
                        f"Invalid value at [{outer_idx}, {inner_idx}]: {v!r}. Must be numeric or None."
                    )
            standardized.append(outer)

        sizes = validate_sequence_sizes(standardized, sizes, structured)
        sub_sizes = validate_sequences_sizes(standardized, sub_sizes, structured)

        return {
            "value": standardized,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


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
            structured = values.get("structured", False)
            min_value = values.get("min_value", -np.inf)
            max_value = values.get("max_value", np.inf)
            strict = values.get("strict", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False
            min_value = -np.inf
            max_value = np.inf
            strict = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        values = standardize_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes, structured)

        return {
            "value": values,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "min_value": min_value,
            "max_value": max_value,
            "strict": strict,
            "structured": structured,
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
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            tuple_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        validate_tuple_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        return {
            "value": values,
            "sizes": sizes,
            "tuple_sizes": tuple_sizes,
            "structured": structured,
        }

    @model_validator(mode="after")
    def validate_numeric_tuple_sequence(self) -> "NumericTupleSequenceParam":
        self.tuple_sizes = validate_tuple_sequence_sizes(
            self.value, self.tuple_sizes, self.structured
        )
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
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            tuple_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        values = standardize_sequences(values)
        validate_tuple_sequences(values)
        sub_sizes = validate_sequences_sizes(values, sub_sizes, structured)

        return {
            "value": values,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "tuple_sizes": tuple_sizes,
            "structured": structured,
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
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)

        normalized = []
        for idx, v in enumerate(values):
            try:
                v = validate_single_color(v)
            except ArgumentValidationError:
                raise ArgumentValidationError(
                    f"Invalid color format: {v!r} at index {idx}"
                )
            normalized.append(v)

        return {"value": normalized, "sizes": sizes, "structured": structured}


class ColorSequencesParam(SequencesParam[ColorType]):
    value: ColorSequences

    @model_validator(mode="before")
    @classmethod
    def validate_color_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)

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

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


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
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        normalized = []
        for idx, v in enumerate(values):
            try:
                v = validate_single_color(v, allow_face_literal=True)
            except ArgumentValidationError:
                raise ArgumentValidationError(
                    f"Invalid color format: {v!r} at index {idx}"
                )
            normalized.append(v)

        return {"value": normalized, "sizes": sizes, "structured": structured}


class EdgeColorSequencesParam(SequencesParam[EdgeColorType]):
    value: EdgeColorSequences

    @model_validator(mode="before")
    @classmethod
    def validate_edgecolor_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)

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

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


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
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        normalized = []
        for idx, v in enumerate(values):
            if not is_marker(v):
                raise ArgumentValidationError(
                    f"Invalid marker format at index {idx}: {v!r}"
                )
            normalized.append(v)

        return {"value": normalized, "sizes": sizes, "structured": structured}


class MarkerSequencesParam(SequencesParam[MarkerParam]):
    value: MarkerSequences

    @model_validator(mode="before")
    @classmethod
    def validate_marker_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
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

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


class LiteralParam(Param[str | None]):
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


class LiteralSequenceParam(SequenceParam[Union[str, None]]):
    options: Optional[StrSequence] = None

    @model_validator(mode="before")
    @classmethod
    def validate_literal_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            options = values.get("options", None)
            values = values["value"]
        else:
            sizes = None
            structured = False
            options = None

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence of strings."
            )

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)

        for idx, v in enumerate(values):
            if not is_literal(v, options):
                raise ArgumentValidationError(
                    f"Invalid literal at index {idx}: {v!r}. Allowed options: {options}."
                )

        return {
            "value": values,
            "options": options,
            "sizes": sizes,
            "structured": structured,
        }


class LiteralSequencesParam(SequencesParam[Union[str, None]]):
    options: Optional[StrSequences] = None

    @model_validator(mode="before")
    @classmethod
    def validate_literal_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            options = values.get("options", None)
            values = values["value"]
        else:
            sizes = None
            sub_sizes = None
            structured = False
            options = None

        if (
            options is None
            or not isinstance(options, Sequence)
            or isinstance(options, str)
        ):
            raise ArgumentValidationError(
                "Literal options must be a non-empty sequence."
            )

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)

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

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "options": options,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


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
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
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

        return {
            "value": normalized,
            "sizes": sizes,
            "structured": structured,
        }


class NormalizationSequencesParam(SequencesParam[NormalizationType]):
    @model_validator(mode="before")
    @classmethod
    def validate_normalization_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
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

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "structured": structured,
        }


class ColormapParam(Param[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values.get("value")

        if isinstance(values, Colormap):
            return {"value": values}

        if not is_literal(values, COLORMAPS):
            raise ArgumentValidationError(
                f"Invalid colormap: {values!r}. Allowed options: {COLORMAPS}."
            )

        return {"value": values}


class ColormapSequenceParam(SequenceParam[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap_sequence(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        normalized = []
        for idx, value in enumerate(values):
            if isinstance(value, Colormap):
                normalized.append(value)
            elif is_literal(value, COLORMAPS):
                normalized.append(value)
            else:
                raise ArgumentValidationError(
                    f"Invalid colormap at index {idx}: {value!r}. Allowed options: {COLORMAPS}."
                )

        return {
            "value": normalized,
            "sizes": sizes,
            "structured": structured,
        }


class ColormapSequencesParam(SequencesParam[ColormapType]):
    @model_validator(mode="before")
    @classmethod
    def validate_colormap_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        normalized_outer = []
        for outer_idx, outer in enumerate(values):
            outer = coerce_to_list(outer)

            normalized_inner = []
            for inner_idx, value in enumerate(outer):
                if isinstance(value, Colormap):
                    normalized_inner.append(value)
                elif is_literal(value, COLORMAPS):
                    normalized_inner.append(value)
                else:
                    raise ArgumentValidationError(
                        f"Invalid colormap at [{outer_idx}, {inner_idx}]: {value!r}. Allowed options: {COLORMAPS}."
                    )
            normalized_outer.append(normalized_inner)

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


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
            sizes = values.get("sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
        normalized = []
        for idx, value in enumerate(values):
            if is_bins(value):
                normalized.append(value)
            else:
                raise ArgumentValidationError(
                    f"Invalid bins value at index {idx}: {value!r}. Must be a positive integer or one of {BINS}."
                )

        return {
            "value": normalized,
            "sizes": sizes,
            "structured": structured,
        }


class BinsSequencesParam(SequencesParam[BinsType]):
    @model_validator(mode="before")
    @classmethod
    def validate_bins_sequences(cls, values: Any) -> dict:
        if isinstance(values, dict):
            sizes = values.get("sizes", None)
            sub_sizes = values.get("sub_sizes", None)
            structured = values.get("structured", False)
            values = values.get("value")
        else:
            sizes = None
            sub_sizes = None
            structured = False

        values = coerce_to_list(values)
        validate_sequence(values)
        sizes = validate_sequence_sizes(values, sizes, structured)
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

        sub_sizes = validate_sequences_sizes(normalized_outer, sub_sizes, structured)

        return {
            "value": normalized_outer,
            "sizes": sizes,
            "sub_sizes": sub_sizes,
            "structured": structured,
        }


class SpineParam(BaseModel):
    visible: Optional[bool] = True
    position: Optional[Union[Tuple[float, float], str]] = None
    color: Optional[ColorType] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    alpha: Optional[float] = None
    bounds: Optional[Tuple[float, float]] = None
    capstyle: Optional[Literal["butt", "round", "projecting"]] = None

    model_config = {"extra": "forbid"}


class SpinesParam(BaseModel):
    left: Optional[SpineParam] = None
    right: Optional[SpineParam] = None
    top: Optional[SpineParam] = None
    bottom: Optional[SpineParam] = None
