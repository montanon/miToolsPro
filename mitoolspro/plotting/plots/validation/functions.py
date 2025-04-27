from pathlib import Path
from typing import Any

from matplotlib.colors import Normalize, get_named_colors_mapping, is_color_like
from matplotlib.markers import MarkerStyle
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
    if isinstance(value, (str, int, Path, MarkerStyle, dict)):
        if isinstance(value, str):
            return value in MARKERS
        if isinstance(value, int):
            return is_value_in_range(value, 0, 11)
        if isinstance(value, dict):
            valid_keys = all(
                key in ["marker", "fillstyle", "transform", "capstyle", "joinstyle"]
                for key in value
            )
            valid_marker = value["marker"] in MARKERS if "marker" in value else True
            valid_fillstyle = (
                value["fillstyle"] in MARKERS_FILLSTYLES
                if "fillstyle" in value
                else True
            )
            valid_transform = (
                isinstance(value["transform"], (str, Normalize))
                if "transform" in value
                else True
            )
            valid_capstyle = (
                value["capstyle"] in ["butt", "round", "projecting"]
                if "capstyle" in value
                else True
            )
            valid_joinstyle = (
                value["joinstyle"] in ["miter", "round", "bevel"]
                if "joinstyle" in value
                else True
            )
            return (
                valid_keys
                and valid_marker
                and valid_fillstyle
                and valid_transform
                and valid_capstyle
                and valid_joinstyle
            )
        return isinstance(value, (Path, MarkerStyle))
    elif value is None:
        return True
    return False
