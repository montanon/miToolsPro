from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, Union

from matplotlib.colors import Colormap, Normalize
from matplotlib.markers import MarkerStyle

T = TypeVar("T")
BoolSequence = Sequence[bool]
BoolSequences = Sequence[BoolSequence]
BinsType: TypeAlias = int | str
BinsSequence = Sequence[BinsType]
BinsSequences = Sequence[BinsSequence]
DictSequence = Sequence[dict]
DictSequences = Sequence[DictSequence]
MarkerType = MarkerStyle | Path | str | dict | int | None
MarkerSequence = Sequence[MarkerType]
MarkerSequences = Sequence[MarkerSequence]
LiteralType = Literal["options"]
LiteralSequence = Sequence[LiteralType]
LiteralSequences = Sequence[LiteralSequence]
NumericType: TypeAlias = float | int
NumericSequence = Sequence[NumericType]
NumericSequences = Sequence[NumericSequence]
StrSequence = Sequence[str]
StrSequences = Sequence[StrSequence]
ColorType = Union[
    str,
    tuple[NumericType, NumericType, NumericType],  # RGB
    tuple[NumericType, NumericType, NumericType, NumericType],  # RGBA
    list[NumericType],
    int,
    float,
    None,
]
ColorSequence = Sequence[ColorType]
ColorSequences = Sequence[ColorSequence]
ColormapType = Union[Colormap, str]
ColormapSequence = Sequence[ColormapType]
ColormapSequences = Sequence[ColormapSequence]
EdgeColorType = Union[Literal["face"], ColorType]
EdgeColorSequence = Sequence[EdgeColorType]
EdgeColorSequences = Sequence[EdgeColorSequence]
NumericTupleType: TypeAlias = tuple[NumericType, ...]
NumericTupleSequence = Sequence[NumericTupleType]
NumericTupleSequences = Sequence[NumericTupleSequence]
NormalizationType = Union[Normalize, str]
NormalizationSequence = Sequence[NormalizationType]
NormalizationSequences = Sequence[NormalizationSequence]
SizesType = Union[Sequence[int], int]
ScaleType = Literal["linear", "log", "symlog", "logit"]
