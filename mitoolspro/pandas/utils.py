from typing import Any, List, Union

from pandas import DataFrame, IndexSlice, MultiIndex

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.exceptions.custom_exceptions import ArgumentValueError


def idxslice(
    df: DataFrame, level: Union[int, str], values: Union[List[Any], Any], axis: int
) -> slice:
    if axis not in {0, 1}:
        raise ArgumentValueError(
            f"Invalid 'axis'={axis}, must be 0 for index or 1 for columns"
        )
    values = [values] if not isinstance(values, list) else values
    idx = df.index if axis == 0 else df.columns
    if isinstance(idx, MultiIndex):
        if isinstance(level, str):
            if level not in idx.names:
                raise ArgumentValueError(
                    f"'level'={level} is not in the MultiIndex names: {idx.names}"
                )
            level = idx.names.index(level)
        elif not isinstance(level, int) or level < 0 or level >= idx.nlevels:
            raise ArgumentValueError(
                f"Provided 'level'={level} is out of bounds for the MultiIndex with {idx.nlevels} levels."
            )
        slices = [slice(None)] * idx.nlevels
        slices[level] = values
        return IndexSlice[tuple(slices)]
    if not isinstance(idx, MultiIndex):
        if isinstance(level, int) and level != 0:
            raise ArgumentValueError(
                "For single-level Index or Columns, level must be 0."
            )
        if isinstance(level, str) and level != idx.name:
            raise ArgumentValueError(
                f"Level '{level}' does not match the Index or Columns name."
            )
        return IndexSlice[values]
