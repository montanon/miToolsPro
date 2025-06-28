from pathlib import Path
from typing import Any, Sequence, Type

import numpy as np
from matplotlib.colors import Normalize, is_color_like
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from numpy import floating, integer, ndarray
from pandas import Series
from pydantic import BaseModel, ValidationError

from mitoolspro.exceptions import ArgumentValidationError
from mitoolspro.plotting.plots.validation.types import (
    ColorType,
    EdgeColorType,
    NumericType,
    SizesType,
)
from mitoolspro.plotting.plots.validation.constants import (
    available_bins,
    available_colors,
    available_markers,
)

COLORS = set(available_colors())
MARKERS = set(available_markers())
MARKERS_FILLSTYLES = set(MarkerStyle.fillstyles)
BINS = available_bins()


def is_valid_model(model_class: Type[BaseModel], **kwargs) -> bool:
    try:
        model_class(**kwargs)
        return True
    except ValidationError:
        return False


def is_indexable(value: Any, index: Any) -> bool:
    try:
        value[index]
        return True
    except (TypeError, IndexError, KeyError):
        return False


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, integer, floating))


def is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, (Sequence, ndarray, Series)) and all(
        is_numeric(v) for v in value
    )


def is_numeric_sequences(value: Any) -> bool:
    return isinstance(value, (Sequence, ndarray, Series)) and all(
        is_numeric_sequence(v) for v in value
    )


def is_value_in_range(value: Any, min_value: NumericType, max_value: NumericType):
    return isinstance(value, NumericType) and min_value <= value and value <= max_value


def is_literal(value: Any, options: Sequence[Any]) -> bool:
    return (isinstance(value, str) and value in options) or value is None


def is_bins(value: Any) -> bool:
    if isinstance(value, (int, str)):
        if isinstance(value, int):
            return is_value_in_range(value, 0, 1_000_000)
        if isinstance(value, str):
            return value in BINS
    return False


def coerce_to_list(value: Any) -> Any:
    if isinstance(value, (ndarray, Series)):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def normalize_rgb_tuple(value: Any) -> Any:
    if not isinstance(value, (tuple, list)):
        return value
    if not all(isinstance(v, (int, float)) for v in value):
        return value
    if len(value) not in {3, 4}:
        return value
    elif len(value) == 3:
        if all(isinstance(v, float) for v in value) and all(
            0.0 <= v <= 1.0 for v in value
        ):
            return tuple(v for v in value)
        if (
            all(isinstance(v, (int, float)) for v in value)
            and all(0 <= v <= 255 for v in value)
            and max(value) > 10  # Custom threshold for [0, 1] float tuples
        ):
            return tuple(round(v / 255.0, 4) for v in value)
    elif len(value) == 4:
        if all(isinstance(v, float) for v in value) and all(
            0.0 <= v <= 1.0 for v in value
        ):
            return tuple(v for v in value)
        if (
            all(isinstance(v, (int, float)) for v in value[:3])
            and all(0 <= v <= 255 for v in value[:3])
            and max(value[:3]) > 10  # Custom threshold for [0, 1] float tuples
            and 0.0 <= value[3] <= 1.0
        ):
            return tuple(
                round(v / 255.0, 4) if n < 3 else v for n, v in enumerate(value)
            )
    return value


def is_color_none(value: Any) -> bool:
    return value is None or value == "none"


def is_color_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def is_color(value: Any) -> bool:
    return (
        is_color_like(value)
        or is_color_none(value)
        or is_color_numeric_scalar(value)
        or is_numeric(value)
    )


def is_marker(value: Any) -> bool:
    if isinstance(value, (str, int, Path, MarkerStyle)):
        if isinstance(value, str):
            return value in MARKERS
        if isinstance(value, int):
            return is_value_in_range(value, 0, 11)
        return True
    elif isinstance(value, dict):
        allowed_keys = {"marker", "fillstyle", "transform", "capstyle", "joinstyle"}
        if not set(value.keys()).issubset(allowed_keys):
            return False

        if "marker" in value:
            if value["marker"] not in MARKERS:
                return False

        if "fillstyle" in value:
            if value["fillstyle"] not in MARKERS_FILLSTYLES:
                return False

        if "transform" in value:
            if not isinstance(value["transform"], (Transform, Normalize)):
                return False

        if "capstyle" in value:
            if value["capstyle"] not in {"butt", "round", "projecting"}:
                return False

        if "joinstyle" in value:
            if value["joinstyle"] not in {"miter", "round", "bevel"}:
                return False

        return True

    elif value is None:
        return True

    return False


def validate_range(
    value: NumericType,
    max_value: NumericType = np.inf,
    min_value: NumericType = -np.inf,
    strict: bool = False,
) -> None:
    if not strict:
        if not (min_value <= value <= max_value):
            raise ArgumentValidationError(
                f"Value {value} is not in range [{min_value}, {max_value}]"
            )
    else:
        if not (min_value < value < max_value):
            raise ArgumentValidationError(
                f"Value {value} is not in range ({min_value}, {max_value})"
            )


def validate_sequence_range(
    value: Sequence,
    min_value: NumericType = -np.inf,
    max_value: NumericType = np.inf,
    strict: bool = False,
) -> None:
    for idx, value in enumerate(value):
        if not strict:
            if not (min_value <= value <= max_value):
                raise ArgumentValidationError(
                    f"Value {value} at index {idx} is not in range [{min_value}, {max_value}]"
                )
        else:
            if not (min_value < value < max_value):
                raise ArgumentValidationError(
                    f"Value {value} at index {idx} is not in range ({min_value}, {max_value})"
                )


def validate_sequences_range(
    values: Sequence,
    min_value: NumericType = -np.inf,
    max_value: NumericType = np.inf,
    strict: bool = False,
) -> None:
    for outer_idx, inner_sequence_param in enumerate(values):
        for inner_idx, value in enumerate(inner_sequence_param):
            if not strict:
                if not (min_value <= value <= max_value):
                    raise ArgumentValidationError(
                        f"Value {value} at index [{outer_idx}, {inner_idx}] is not "
                        + f"in range [{min_value}, {max_value}]"
                    )
            else:
                if not (min_value < value < max_value):
                    raise ArgumentValidationError(
                        f"Value {value} at index [{outer_idx}, {inner_idx}] is not "
                        + f"in range ({min_value}, {max_value})"
                    )


def validate_sequence(value: Any) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ArgumentValidationError(f"Expected Sequence, got {type(value)}")


def validate_numeric(value: Any) -> None:
    if not isinstance(value, (int, float)):
        raise ArgumentValidationError(f"Expected numeric {value=}, got {type(value)}")


def validate_sequence_sizes(
    values: Sequence, sizes: SizesType, structured: bool
) -> SizesType | None:
    if sizes is not None:
        sizes = sizes if isinstance(sizes, Sequence) else [sizes]
        if len(values) not in sizes:
            raise ArgumentValidationError(
                f"Expected Sequence of sizes: {sizes}, got size: {len(values)} instead"
            )
        if structured:
            if len(sizes) != 1:
                raise ArgumentValidationError(
                    f"Validation of structured Sequence requires a single size: int, got {sizes=}"
                )
            if len(values) != sizes[0]:
                raise ArgumentValidationError(
                    f"Expected Sequence of size: {sizes[0]}, got size: {len(values)} instead"
                )
    return sizes


def validate_sequences_sizes(
    values: Sequence, sub_sizes: SizesType, structured: bool
) -> SizesType | None:
    if sub_sizes is not None:
        sub_sizes = sub_sizes if isinstance(sub_sizes, Sequence) else [sub_sizes]
        for idx, value in enumerate(values):
            if len(value) != 1 and len(value) not in sub_sizes:
                raise ArgumentValidationError(
                    f"Expected sub Sequences of sizes: {sub_sizes} got size: {len(value)} at index={idx}"
                )
        if structured:
            if len(values) != 1 and len(values) != len(sub_sizes):
                raise ArgumentValidationError(
                    f"Mismatch in structured Sequence of length: {len(values)}, got sizes: {sub_sizes} instead"
                )
            for idx, value in enumerate(values):
                if len(value) != 1 and len(value) != sub_sizes[idx]:
                    raise ArgumentValidationError(
                        f"Expected sub Sequences of size: {sub_sizes[idx]}, got size: {len(value)} at index={idx}"
                    )
    return sub_sizes


def standardize_sequences(values: Sequence[Any]) -> list[list]:
    standardized = []
    for idx, value in enumerate(values):
        value = coerce_to_list(value)
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ArgumentValidationError(
                f"Expected a Sequence inside outer Sequence, got {type(value)} at index={idx}"
            )
        standardized.append(value)
    return standardized


def validate_tuple_sizes(value: Any, tuple_sizes: SizesType) -> SizesType | None:
    if tuple_sizes is not None:
        if not isinstance(tuple_sizes, Sequence):
            tuple_sizes = [tuple_sizes]
        if not all(size > 0 for size in tuple_sizes):
            raise ArgumentValidationError(
                f"All tuple_sizes must be positive, got {tuple_sizes}."
            )
        if len(value) not in tuple_sizes:
            raise ArgumentValidationError(
                f"Invalid tuple length {len(value)}. Allowed sizes: {tuple_sizes}."
            )
    return tuple_sizes


def validate_tuple_sequence_sizes(
    values: Sequence, tuple_sizes: SizesType, structured: bool
) -> SizesType | None:
    if tuple_sizes is not None:
        if not isinstance(tuple_sizes, Sequence):
            tuple_sizes = [tuple_sizes]
        if not all(size > 0 for size in tuple_sizes):
            raise ArgumentValidationError(
                f"All sizes must be positive, got {tuple_sizes}."
            )
        for idx, value in enumerate(values):
            if len(value) not in tuple_sizes:
                raise ArgumentValidationError(
                    f"Invalid tuple length {len(value)} at index {idx}. Allowed sizes: {tuple_sizes}."
                )
        if len(tuple_sizes) != 1 and structured:
            if len(tuple_sizes) != len(values):
                raise ArgumentValidationError(
                    f"Validation of structured Sequence requires a single tuple size, got {tuple_sizes=}"
                )
            for idx, value in enumerate(values):
                if len(value) != tuple_sizes[idx]:
                    raise ArgumentValidationError(
                        f"Expected tuple size: {tuple_sizes[idx]}, got size: {len(value)} at index {idx}"
                    )
    return tuple_sizes


def validate_tuple_sequence(values: Sequence) -> None:
    for idx, v in enumerate(values):
        if not isinstance(v, tuple):
            raise ArgumentValidationError(
                f"Expected each element to be a tuple, got {type(v)} at index {idx}"
            )


def validate_tuple_sequences(values: Sequence) -> None:
    for outer_idx, inner_sequence in enumerate(values):
        for inner_idx, value in enumerate(inner_sequence):
            if not isinstance(value, tuple):
                raise ArgumentValidationError(
                    f"Expected each element to be a tuple, got {type(value)} at index [{outer_idx}, {inner_idx}]"
                )


def validate_tuple_sequences_sizes(
    values: Sequence, tuple_sizes: SizesType
) -> SizesType | None:
    if tuple_sizes is not None:
        if not isinstance(tuple_sizes, Sequence):
            tuple_sizes = [tuple_sizes]
        if not all(size > 0 for size in tuple_sizes):
            raise ArgumentValidationError(
                f"All sizes must be positive, got {tuple_sizes}."
            )
        for outer_idx, inner_sequence in enumerate(values):
            for inner_idx, value in enumerate(inner_sequence):
                if len(value) not in tuple_sizes:
                    raise ArgumentValidationError(
                        f"Invalid tuple length {len(value)} at index [{outer_idx}, {inner_idx}]. Allowed sizes: {tuple_sizes}."
                    )
    return tuple_sizes


def validate_single_color(
    value: Any, allow_face_literal: bool = False
) -> ColorType | EdgeColorType:
    if isinstance(value, (np.ndarray, Series)):
        value = value.tolist()

    if allow_face_literal and value == "face":
        return value

    value = normalize_rgb_tuple(value)

    if not is_color(value):
        raise ArgumentValidationError(f"Invalid color format: {value!r}")

    return value
