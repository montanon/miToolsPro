from pathlib import Path
from typing import Any

from matplotlib.colors import Normalize, get_named_colors_mapping, is_color_like
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from numpy import ndarray
from pandas import Series
from typing_extensions import TypeAlias

NumericType: TypeAlias = float | int
COLORS = set(get_named_colors_mapping().keys())
MARKERS = set(MarkerStyle.markers.keys()).union(set(MarkerStyle.filled_markers))
MARKERS_FILLSTYLES = set(MarkerStyle.fillstyles)


def is_indexable(value: Any, index: Any) -> bool:
    try:
        value[index]
        return True
    except (TypeError, IndexError, KeyError):
        return False


def is_value_in_range(value: Any, min_value: NumericType, max_value: NumericType):
    return isinstance(value, NumericType) and min_value <= value and value <= max_value


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
        is_color_like(value) or is_color_none(value) or is_color_numeric_scalar(value)
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
