from typing import Any

from matplotlib.colors import (
    get_named_colors_mapping,
    is_color_like,
)
from numpy import ndarray
from pandas import Series

COLORS = set(get_named_colors_mapping().keys())


def is_indexable(value: Any, index: Any) -> bool:
    try:
        value[index]
        return True
    except (TypeError, IndexError, KeyError):
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
            return tuple(v / 255.0 for v in value)
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
            return tuple(v / 255.0 if n < 3 else v for n, v in enumerate(value))
    return value


def is_color_none(value: Any) -> bool:
    return value is None or value == "none"


def is_color_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0


def is_color(value: Any) -> bool:
    return (
        is_color_like(value) or is_color_none(value) or is_color_numeric_scalar(value)
    )
