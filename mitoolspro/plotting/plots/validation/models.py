from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, Optional, TypeAlias, TypeVar, Union

import numpy as np
from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle
from numpy import integer, ndarray
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

T = TypeVar("T")
NumericType: TypeAlias = float | int


class Param[T](BaseModel):
    value: T

    model_config = ConfigDict(extra="forbid")


class RangeParam(Param[NumericType]):
    min_value: Optional[NumericType] = -np.inf
    max_value: Optional[NumericType] = np.inf

    @model_validator(mode="after")
    def validate_range(self) -> "RangeParam":
        if not isinstance(self.value, (int, float)):
            raise ArgumentValidationError(
                f"Expected numeric value, got {type(self.value)}"
            )
        if not (self.min_value <= self.value <= self.max_value):
            raise ArgumentValidationError(
                f"Value {self.value} is not in range [{self.min_value}, {self.max_value}]"
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

    @model_validator(mode="before")
    @classmethod
    def validate_type(cls, values: Any) -> dict:
        if isinstance(values, dict):
            values = values["value"]
        if isinstance(values, (ndarray, Series)):
            values = values.tolist()
        elif isinstance(values, tuple):
            values = list(values)
        if not isinstance(values, Sequence):
            raise ArgumentValidationError(f"Expected Sequence, got {type(values)}")
        return {"value": values}


class SequencesParam[T](Param[SequenceParam[SequenceParam[T]]]):
    value: SequenceParam[SequenceParam[T]]

    @model_validator(mode="before")
    @classmethod
    def standardize_input(cls, values: Any) -> dict:
        if isinstance(values, dict):
            input_value = values["value"]
        else:
            input_value = values

        if isinstance(input_value, (ndarray, Series)):
            input_value = input_value.tolist()
        if not isinstance(input_value, Sequence):
            raise ArgumentValidationError(
                f"Expected a Sequence, got {type(input_value)}"
            )

        normalized = []
        for value in input_value:
            if isinstance(value, (list, tuple, ndarray, Series)):
                if isinstance(value, (ndarray, Series)):
                    value = value.tolist()
                elif isinstance(value, tuple):
                    value = list(value)
                sequence_param = SequenceParam[T](value=value)
            elif isinstance(value, SequenceParam):
                sequence_param = value
            else:
                raise ArgumentValidationError(
                    f"Expected a SequenceParam[T] or raw sequence, got {type(value)}"
                )
            normalized.append(sequence_param)

        return {"value": normalized}


class NumericSequenceParam(SequenceParam[NumericType]):
    pass


class NumericSequencesParam(SequencesParam[NumericType]):
    pass


class StrSequenceParam(SequenceParam[str]):
    pass


class StrSequencesParam(SequencesParam[str]):
    pass


class BoolSequenceParam(SequenceParam[bool]):
    pass


class BoolSequencesParam(SequencesParam[bool]):
    pass


class DictSequenceParam(SequenceParam[dict]):
    pass


class DictSequencesParam(SequencesParam[dict]):
    pass


if __name__ == "__main__":
    print(RangeParam(value=1))
    print(RangeParam(value=1, min_value=0, max_value=10))
    try:
        print(RangeParam(value=1, min_value=2))
    except ValidationError as e:
        print(e)
