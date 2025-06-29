from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Literal, TypeAlias, TypeVar, Union

from matplotlib.colors import Colormap, Normalize, get_named_colors_mapping
from matplotlib.markers import MarkerStyle

T = TypeVar("T")
BoolSequence = Sequence[bool]
BoolSequences = Sequence[BoolSequence]
BinsType: TypeAlias = (
    int | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
)
BinsSequence = Sequence[BinsType]
BinsSequences = Sequence[BinsSequence]
DictSequence = Sequence[dict | None]
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
ColormapType = Union[
    Colormap,
    Literal[
        "magma",
        "inferno",
        "plasma",
        "viridis",
        "cividis",
        "twilight",
        "twilight_shifted",
        "turbo",
    ],
]
ColormapSequence = Sequence[ColormapType]
ColormapSequences = Sequence[ColormapSequence]
EdgeColorType = Union[Literal["face", "none", None], ColorType]
EdgeColorSequence = Sequence[EdgeColorType]
EdgeColorSequences = Sequence[EdgeColorSequence]
NumericTupleType: TypeAlias = tuple[Union[NumericType, None], ...]
NumericTupleSequence = Sequence[NumericTupleType]
NumericTupleSequences = Sequence[NumericTupleSequence]
NormalizationType = Union[Normalize, str]
NormalizationSequence = Sequence[NormalizationType]
NormalizationSequences = Sequence[NormalizationSequence]
SizesType = Union[Sequence[int], int, None]
ScaleType = Literal["linear", "log", "symlog", "logit"]
LineStyleType = Union[
    Literal[
        "-",
        "--",
        "-.",
        ":",
        "None",
        "none",
        " ",
        "",
        "dotted",
        "dashed",
        "dashdot",
        "solid",
    ],
    Sequence[
        Literal[
            "-",
            "--",
            "-.",
            ":",
            "None",
            "none",
            " ",
            "",
            "dotted",
            "dashed",
            "dashdot",
            "solid",
        ]
    ],
]
ScaleType = Literal["linear", "log", "symlog", "logit"]
TickParamsType = Dict[str, Any]
from mitoolspro.plotting.plots.validation.constants import (
    available_bins,
    available_colormaps,
    available_colors,
    available_hatches,
    available_linestyles,
    available_markers,
)

COLORS = set(available_colors())
COLORMAPS = available_colormaps()
MARKERS = set(available_markers())
MARKERS_FILLSTYLES = set(MarkerStyle.fillstyles)
KERNELS = ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
ORIENTATIONS = ["horizontal", "vertical"]
BANDWIDTH_METHODS = ["scott", "silverman"]
BARS_ALIGN = ["center", "edge"]
NORMALIZATIONS = [
    "linear",
    "log",
    "symlog",
    "asinh",
    "logit",
    "function",
    "functionlog",
]
BINS = available_bins()
LINESTYLES = available_linestyles()
SCALES = ["linear", "log", "symlog", "logit"]
HATCHES = available_hatches()
HIST_ALIGN = ["left", "mid", "right"]
HIST_HISTTYPE = ["bar", "barstacked", "step", "stepfilled"]
TICKPARAMS = [
    "size",
    "width",
    "color",
    "tickdir",
    "pad",
    "labelsize",
    "labelcolor",
    "zorder",
    "gridOn",
    "tick1On",
    "tick2On",
    "label1On",
    "label2On",
    "length",
    "direction",
    "left",
    "bottom",
    "right",
    "top",
    "labelleft",
    "labelbottom",
    "labelright",
    "labeltop",
    "labelrotation",
    "grid_agg_filter",
    "grid_alpha",
    "grid_animated",
    "grid_antialiased",
    "grid_clip_box",
    "grid_clip_on",
    "grid_clip_path",
    "grid_color",
    "grid_dash_capstyle",
    "grid_dash_joinstyle",
    "grid_dashes",
    "grid_data",
    "grid_drawstyle",
    "grid_figure",
    "grid_fillstyle",
    "grid_gapcolor",
    "grid_gid",
    "grid_in_layout",
    "grid_label",
    "grid_linestyle",
    "grid_linewidth",
    "grid_marker",
    "grid_markeredgecolor",
    "grid_markeredgewidth",
    "grid_markerfacecolor",
    "grid_markerfacecoloralt",
    "grid_markersize",
    "grid_markevery",
    "grid_mouseover",
    "grid_path_effects",
    "grid_picker",
    "grid_pickradius",
    "grid_rasterized",
    "grid_sketch_params",
    "grid_snap",
    "grid_solid_capstyle",
    "grid_solid_joinstyle",
    "grid_transform",
    "grid_url",
    "grid_visible",
    "grid_xdata",
    "grid_ydata",
    "grid_zorder",
    "grid_aa",
    "grid_c",
    "grid_ds",
    "grid_ls",
    "grid_lw",
    "grid_mec",
    "grid_mew",
    "grid_mfc",
    "grid_mfcalt",
    "grid_ms",
]
