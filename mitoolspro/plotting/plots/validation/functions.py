import re
from typing import Any

from matplotlib.colors import get_named_colors_mapping
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


def is_color_tuple(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) not in {3, 4}:
        return False
    return all(isinstance(val, (int, float)) for val in value)


def is_color_hex(value: Any) -> bool:
    return isinstance(value, str) and re.match(
        r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", value
    )


def is_color_str(value: Any) -> bool:
    return isinstance(value, str) and value in COLORS


def is_color(value: Any) -> bool:
    return (
        is_color_tuple(value)
        or is_color_hex(value)
        or is_color_str(value)
        or isinstance(value, (int, float))
        or value is None
    )
