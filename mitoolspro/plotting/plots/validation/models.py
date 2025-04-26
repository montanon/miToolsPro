from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from numpy import integer
from pandas import Series
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from typing_extensions import TypeGuard

from mitoolspro.exceptions import (
    ArgumentStructureError,
    ArgumentTypeError,
    ArgumentValidationError,
)
from mitoolspro.plotting.plots.matplotlib_typing import (
    BINS,
    CMAPS,
    COLORS,
    MARKERS,
    MARKERS_FILLSTYLES,
    NORMALIZATIONS,
    NumericSequences,
)
from mitoolspro.plotting.plots.validation.schemas import (
    NumpyArray,
    PandasSeries,
)

T = TypeVar("T")
NumericType: TypeAlias = float | int


class Param[T](BaseModel):
    value: T

    model_config = ConfigDict(extra="forbid")


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


if __name__ == "__main__":
    print(NumericParam(value=0))
    print(NumericParam(value=0.0))
    print(NumericParam(value=-1))
    print(NumericParam(value=-1.0))
    print(NumericParam(value=1))
    print(NumericParam(value=1.0))
    try:
        print(NumericParam(value="abcd"))
    except ValidationError as e:
        print(e)
    print(StrParam(value="abcd"))
    try:
        print(StrParam(value=1))
    except ValidationError as e:
        print(e)
    print(BoolParam(value=True))
    try:
        print(BoolParam(value="abcd"))
    except ValidationError as e:
        print(e)
    print(DictParam(value={"a": 1, "b": 2}))
    try:
        print(DictParam(value=1))
    except ValidationError as e:
        print(e)
    print(SequenceParam(value=[1, 2, 3, "abcd", True, False]))
    print(SequenceParam(value=(1, 2, 3, "abcd", True, False)))
    print(SequenceParam(value=np.array([1, 2, 3, "abcd", True, False])))
    print(SequenceParam(value=Series([1, 2, 3, "abcd", True, False])))
    try:
        print(SequenceParam(value=1))
    except ValidationError as e:
        print(e)
