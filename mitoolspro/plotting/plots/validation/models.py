from collections.abc import Sequence
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar, Union

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

    # @model_validator(mode="after")
    # def validate_element_types(self) -> "SequencesParam[T]":
    #     expected_type = self.__pydantic_generic_metadata__["args"][0]

    #     for sequence_param in self.value:
    #         for item in sequence_param.value:
    #             if not isinstance(item, expected_type):
    #                 raise ArgumentValidationError(
    #                     f"Expected inner element of type {expected_type}, got {type(item)}"
    #                 )
    #     return self


class NumericSequenceParam(SequenceParam[NumericType]):
    pass


class NumericSequencesParam(SequencesParam[NumericSequenceParam]):
    pass


if __name__ == "__main__":
    print(SequenceParam(value=[1, 2, 3, "abcd", True, False]))
    print(SequenceParam(value=(1, 2, 3, "abcd", True, False)))
    print(SequenceParam(value=np.array([1, 2, 3, "abcd", True, False])))
    print(SequenceParam(value=Series([1, 2, 3, "abcd", True, False])))
    try:
        print(SequenceParam(value=1))
    except ValidationError as e:
        print(e)
    print(SequencesParam(value=[[1, 2, 3], [4, 5, 6]]))
    print(
        SequencesParam(
            value=[
                [1, 2, 3],
                np.array([4, 5, 6]),
                Series([7, 8, 9]),
                tuple([10, "abcd", 12]),
            ]
        )
    )
    try:
        print(SequencesParam(value=1))
    except ValidationError as e:
        print(e)
    print(NumericSequenceParam(value=[1, 2, 3]))
    print(NumericSequenceParam(value=Series([1, 2, 3])))
    print(NumericSequenceParam(value=(1, 2, 3)))
    print(NumericSequenceParam(value=np.array([1, 2, 3])))
    try:
        print(NumericSequenceParam(value=[1, 2, 3, "abcd", True, False]))
    except ValidationError as e:
        print(e)
    print(NumericSequencesParam(value=[[1, 2, 3], [4, 5, 6]]))
    # print(NumericSequencesParam(value=[[1, 2, 3], [4, 5, 6]]))
    # print(
    #     NumericSequencesParam(
    #         value=[
    #             [1, 2, 3],
    #             np.array([4, 5, 6]),
    #             Series([7, 8, 9]),
    #             tuple([10, -10, 12]),
    #         ]
    #     )
    # )
