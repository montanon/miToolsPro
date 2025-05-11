from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from numpy import ndarray
from pandas import Series

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.plotting.plots.validation.models import (
    BoolParam,
    ColorParam,
    DictParam,
    DictSequenceParam,
    EdgeColorParam,
    IntParam,
    LiteralParam,
    NumericParam,
    NumericSequenceParam,
    NumericTupleParam,
    NumStrParam,
    RangeParam,
    SequenceParam,
    SpineParam,
    SpinesParam,
    StrParam,
    StrSequenceParam,
    TransformParam,
)
from mitoolspro.plotting.plots.validation.types import (
    ColorType,
    DictSequence,
    NumericSequence,
    NumericTupleType,
    NumericType,
    ScaleType,
    StrSequence,
)


class ParamsMixIn:
    def __init__(self, ax: Optional[Axes] = None, **kwargs):
        self.ax: Optional[Axes] = ax
        self.figure: Optional[Figure] = None if self.ax is None else self.ax.figure
        self._params = {
            "alpha": None,
            "aspect": None,
            "title": "",
            "suptitle": "",
            "transform": None,
            "xlabel": "",
            "ylabel": "",
            "xscale": None,
            "yscale": None,
            "xlim": None,
            "ylim": None,
            "xticks": None,
            "yticks": None,
            "xticklabels": None,
            "yticklabels": None,
            "xtickparams": None,
            "ytickparams": None,
            "spines": {},
            "legend": None,
            "texts": None,
            "grid": None,
            "facecolor": None,
            "background": None,
            "figure_background": None,
            "figsize": (10, 8),
            "tight_layout": False,
            "style": None,
        }
        self._init_params = {**self._params}
        self._params_to_avoid = [
            "xtickparams",
            "ytickparams",
            "textprops",
            "wedgeprops",  # Awful
            "capprops",
            "whiskerprops",
            "boxprops",
            "flierprops",
            "medianprops",
            "meanprops",
        ]
        self._set_init_params(**kwargs)

    def _set_init_params(self, **kwargs):
        for param, default in self._init_params.items():
            setattr(self, param, default)
            if param in kwargs and kwargs[param] is not None:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
                    if (
                        isinstance(kwargs[param], dict)
                        and param not in self._params_to_avoid
                    ):
                        getattr(self, setter_name)(**kwargs[param])
                    else:
                        getattr(self, setter_name)(kwargs[param])
                else:
                    raise ArgumentValueError(
                        f"Parameter '{param}' is not a valid parameter."
                    )

    def reset_params(self):
        for param, default in self._init_params.items():
            setattr(self, param, default)
        return self

    def set_alpha(self, alpha: NumericType | None):
        if alpha is not None:
            RangeParam(value=alpha, min_value=0, max_value=1)
        self.alpha = alpha
        return self

    def set_aspect(self, aspect: Literal["auto", "equal"] | float | None):
        if aspect is not None:
            if isinstance(aspect, str):
                LiteralParam(value=aspect, options=["auto", "equal"])
            elif isinstance(aspect, float):
                RangeParam(value=aspect, min_value=0)
        self.aspect = aspect
        return self

    def set_title(self, label: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_title.html"""
        StrParam(value=label)
        self.title = dict(label=label, **kwargs)
        return self

    def set_suptitle(self, t: str, **kwargs):
        StrParam(value=t)
        self.suptitle = dict(t=t, **kwargs)
        return self

    def set_transform(self, transform: Transform):
        TransformParam(value=transform)
        self.transform = transform
        return self

    def set_xlabel(self, xlabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel"""
        StrParam(value=xlabel)
        self.xlabel = dict(xlabel=xlabel, **kwargs)
        return self

    def set_ylabel(self, ylabel: str, **kwargs):
        """https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel"""
        StrParam(value=ylabel)
        self.ylabel = dict(ylabel=ylabel, **kwargs)
        return self

    def set_axes_labels(self, xlabel: str, ylabel: str, **kwargs):
        self.set_xlabel(xlabel, **kwargs)
        self.set_ylabel(ylabel, **kwargs)
        return self

    def set_xscale(self, xscale: Optional[ScaleType] = None):
        if xscale is not None:
            LiteralParam(value=xscale, options=["linear", "log", "symlog", "logit"])
        self.xscale = xscale
        return self

    def set_yscale(self, yscale: Optional[ScaleType] = None):
        if yscale is not None:
            LiteralParam(value=yscale, options=["linear", "log", "symlog", "logit"])
        self.yscale = yscale
        return self

    def set_scales(
        self,
        xscale: Optional[ScaleType] = None,
        yscale: Optional[ScaleType] = None,
    ):
        self.set_xscale(xscale)
        self.set_yscale(yscale)
        return self

    def set_xlim(self, xlim: Optional[NumericTupleType] = None):
        if xlim is not None:
            NumericTupleParam(value=xlim, tuple_sizes=2)
        self.xlim = xlim
        return self

    def set_ylim(self, ylim: Optional[NumericTupleType] = None):
        if ylim is not None:
            NumericTupleParam(value=ylim, tuple_sizes=2)
        self.ylim = ylim
        return self

    def set_limits(
        self,
        xlim: Union[Tuple[float, float], None] = None,
        ylim: Union[Tuple[float, float], None] = None,
    ):
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        return self

    def set_xticks(self, xticks: Optional[NumericSequence] = None):
        if xticks is not None:
            NumericSequenceParam(value=xticks)
        self.xticks = xticks
        return self

    def set_yticks(self, yticks: Optional[NumericSequence] = None):
        if yticks is not None:
            NumericSequenceParam(value=yticks)
        self.yticks = yticks
        return self

    def set_ticks(
        self,
        xticks: Optional[NumericSequence] = None,
        yticks: Optional[NumericSequence] = None,
    ):
        self.set_xticks(xticks)
        self.set_yticks(yticks)
        return self

    def set_xticklabels(self, xticklabels: Optional[StrSequence] = None):
        if xticklabels is not None:
            SequenceParam[str | None | int | float](value=xticklabels)
        self.xticklabels = xticklabels
        return self

    def set_yticklabels(self, yticklabels: Optional[StrSequence] = None):
        if yticklabels is not None:
            SequenceParam[str | None | int | float](value=yticklabels)
        self.yticklabels = yticklabels
        return self

    def set_ticklabels(
        self,
        xticklabels: Optional[StrSequence] = None,
        yticklabels: Optional[StrSequence] = None,
    ):
        self.set_xticklabels(xticklabels)
        self.set_yticklabels(yticklabels)
        return self

    def set_xtickparams(self, xtickparams: Optional[Dict[str, Any]] = None):
        if xtickparams is not None:
            DictParam(value=xtickparams)
        self.xtickparams = xtickparams
        return self

    def set_ytickparams(self, ytickparams: Optional[Dict[str, Any]] = None):
        if ytickparams is not None:
            DictParam(value=ytickparams)
        self.ytickparams = ytickparams
        return self

    def set_tickparams(
        self,
        xtickparams: Optional[Dict[str, Any]] = None,
        ytickparams: Optional[Dict[str, Any]] = None,
    ):
        self.set_xtickparams(xtickparams)
        self.set_ytickparams(ytickparams)
        return self

    def set_spines(
        self,
        left: Dict[str, Any] = None,
        right: Dict[str, Any] = None,
        bottom: Dict[str, Any] = None,
        top: Dict[str, Any] = None,
    ):
        self.spines = SpinesParam(
            left=SpineParam.model_validate(left) if left is not None else None,
            right=SpineParam.model_validate(right) if right is not None else None,
            bottom=SpineParam.model_validate(bottom) if bottom is not None else None,
            top=SpineParam.model_validate(top) if top is not None else None,
        )
        return self

    def set_legend(
        self,
        show: bool = True,
        labels: Optional[StrSequence | str] = None,
        handles: Optional[Sequence[Any]] = None,
        loc: Optional[str | int] = "best",
        bbox_to_anchor: Optional[Tuple[float, float]] = None,
        ncol: Optional[int] = 1,
        fontsize: Optional[int | str] = None,
        title: Optional[str] = None,
        title_fontsize: Optional[int | str] = None,
        frameon: bool = True,
        fancybox: bool = True,
        framealpha: float = 0.8,
        edgecolor: Optional[str] = None,
        facecolor: Optional[str] = "inherit",
        **kwargs,
    ):
        BoolParam(value=show)
        BoolParam(value=frameon)
        BoolParam(value=fancybox)
        IntParam(value=ncol)
        NumericParam(value=framealpha)
        if labels is not None:
            if isinstance(labels, str):
                labels = [labels]
            else:
                StrSequenceParam(value=labels)
        if handles is not None:
            SequenceParam(value=handles)
        if bbox_to_anchor is not None:
            NumericTupleParam(value=bbox_to_anchor, tuple_sizes=[2, 4])
        if fontsize is not None:
            NumStrParam(value=fontsize)
        if title is not None:
            StrParam(value=title)
        if title_fontsize is not None:
            NumStrParam(value=title_fontsize)
        if edgecolor is not None:
            EdgeColorParam(value=edgecolor)
        if facecolor is not None and facecolor != "inherit":
            ColorParam(value=facecolor)
        if "kwargs" not in kwargs:
            legend_kwargs = {
                "loc": loc,
                "ncol": ncol,
                "frameon": frameon,
                "fancybox": fancybox,
                "framealpha": framealpha,
            }
            if labels is not None:
                legend_kwargs["labels"] = labels
            if handles is not None:
                legend_kwargs["handles"] = handles
            if bbox_to_anchor is not None:
                legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
            if fontsize is not None:
                legend_kwargs["fontsize"] = fontsize
            if title is not None:
                legend_kwargs["title"] = title
            if title_fontsize is not None:
                legend_kwargs["title_fontsize"] = title_fontsize
            if edgecolor is not None:
                legend_kwargs["edgecolor"] = edgecolor
            if facecolor is not None:
                legend_kwargs["facecolor"] = facecolor
            legend_kwargs.update(kwargs)
            legend = {"show": show, "kwargs": legend_kwargs}
        else:
            legend = {"show": show, "kwargs": kwargs["kwargs"]}
        self.legend = legend if show else None
        return self

    def set_texts(self, texts: Optional[DictSequence | Dict] = None):
        if texts is not None:
            if isinstance(texts, dict):
                texts = [texts]
            else:
                DictSequenceParam(value=texts)
        self.texts = texts
        return self

    def set_grid(
        self,
        visible: bool = None,
        which: Literal["major", "minor", "both"] = "major",
        axis: Literal["both", "x", "y"] = "both",
        **kwargs,
    ):
        BoolParam(value=visible)
        LiteralParam(value=which, options=["major", "minor", "both"])
        LiteralParam(value=axis, options=["both", "x", "y"])
        self.grid = dict(visible=visible, which=which, axis=axis, **kwargs)
        return self

    def set_facecolor(self, facecolor: Optional[ColorType] = None):
        if facecolor is not None:
            ColorParam(value=facecolor)
        self.facecolor = facecolor
        return self

    def set_background(self, background: Optional[ColorType] = None):
        if background is not None:
            ColorParam(value=background)
        self.background = background
        return self

    def set_figure_background(self, figure_background: Optional[ColorType] = None):
        if figure_background is not None:
            ColorParam(value=figure_background)
        self.figure_background = figure_background
        return self

    def set_figsize(self, figsize: Optional[NumericTupleType] = None):
        if figsize is not None:
            NumericTupleParam(value=figsize, tuple_sizes=2)
        self.figsize = figsize
        return self

    def set_tight_layout(self, tight_layout: bool = False):
        BoolParam(value=tight_layout)
        self.tight_layout = tight_layout
        return self

    def set_style(self, style: Optional[str] = None):
        if style is not None:
            LiteralParam(value=style, options=plt.style.available)
        self.style = style
        return self

    def _prepare_draw(self, clear: bool = False):
        if clear:
            self.clear()
        if self.style is not None:
            self._default_style = plt.rcParams.copy()
            plt.style.use(self.style)
        if not self.ax:
            self.figure, self.ax = plt.subplots(figsize=self.figsize)
        if self.grid is not None and self.grid["visible"]:
            self.ax.grid(**self.grid)

    def _finalize_draw(self, show: bool = False):
        if self.tight_layout:
            plt.tight_layout()
        if show:
            self.figure.show()
        if self.style is not None:
            plt.rcParams.update(self._default_style)
        return self.ax

    def clear(self):
        if self.figure or self.ax:
            plt.close(self.figure)
            self.figure = None
            self.ax = None
        return self

    def _apply_common_properties(self):
        if self.title:
            self.ax.set_title(**self.title)
        if self.xlabel:
            self.ax.set_xlabel(**self.xlabel)
            if "color" in self.xlabel:
                self.ax.tick_params(axis="x", colors=self.xlabel["color"])
        if self.ylabel:
            self.ax.set_ylabel(**self.ylabel)
            if "color" in self.ylabel:
                self.ax.tick_params(axis="y", colors=self.ylabel["color"])
        if self.xscale:
            self.ax.set_xscale(self.xscale)
        if self.yscale:
            self.ax.set_yscale(self.yscale)
        if self.texts is not None:
            for text in self.texts:
                self.ax.text(**text)
        if self.legend is not None and self.legend["show"]:
            self.ax.legend(**self.legend["kwargs"])
        if self.background:
            self.ax.set_facecolor(self.background)
        if self.figure_background:
            self.figure.set_facecolor(self.figure_background)
        if self.suptitle:
            self.figure.suptitle(**self.suptitle)
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        if self.ylim is not None:
            self.ax.set_ylim(self.ylim)
        if self.xticks is not None:
            self.ax.set_xticks(self.xticks)
        if self.yticks is not None:
            self.ax.set_yticks(self.yticks)
        if self.xticklabels is not None:
            self.ax.set_xticklabels(self.xticklabels)
        if self.yticklabels is not None:
            self.ax.set_yticklabels(self.yticklabels)
        if self.xtickparams is not None:
            self.ax.tick_params(axis="x", **self.xtickparams)
        if self.ytickparams is not None:
            self.ax.tick_params(axis="y", **self.ytickparams)
        if self.spines:
            for spine, spine_params in self.spines.model_dump().items():
                if spine_params is not None:
                    for param, values in spine_params.items():
                        if values is not None:
                            if param == "visible":
                                self.ax.spines[spine].set_visible(values)
                            elif param == "position":
                                if isinstance(values, str):
                                    self.ax.spines[spine].set_position(values)
                                else:
                                    self.ax.spines[spine].set_position(("data", values))
                            elif param == "color":
                                self.ax.spines[spine].set_color(values)
                            elif param == "linewidth":
                                self.ax.spines[spine].set_linewidth(values)
                            elif param == "linestyle":
                                self.ax.spines[spine].set_linestyle(values)
                            elif param == "alpha":
                                self.ax.spines[spine].set_alpha(values)
                            elif param == "bounds":
                                self.ax.spines[spine].set_bounds(*values)
                            elif param == "capstyle":
                                self.ax.spines[spine].set_capstyle(values)

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, ndarray):
            return value.tolist()
        elif isinstance(value, Series):
            return value.to_list()
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Colormap):
            return value.name
        elif isinstance(value, Normalize):
            return value.__class__.__name__.lower()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, MarkerStyle):
            marker = dict(
                marker=value.get_marker(),
                fillstyle=value.get_fillstyle(),
                capstyle=value.get_capstyle(),
                joinstyle=value.get_joinstyle(),
            )
            return marker
        elif isinstance(value, SpinesParam):
            return value.model_dump()

        return value


class FigureParamsMixIn:
    def __init__(self, figure: Figure = None, **kwargs):
        self.figure: Figure = figure
        self._figure_params = {
            "figsize": tuple(self.figure.get_size_inches()) if self.figure else (10, 8),
            "style": None,
            "tight_layout": False,
            "figure_background": None,
            "suptitle": None,
        }
        self._init_params = {**self._figure_params}
        self._set_init_params(**kwargs)

    def _set_init_params(self, **kwargs):
        for param, default in self._init_params.items():
            setattr(self, param, default)
            if param in kwargs and kwargs[param] is not None:
                setter_name = f"set_{param}"
                if hasattr(self, setter_name):
                    getattr(self, setter_name)(kwargs[param])
                else:
                    raise ArgumentValueError(f"Parameter '{param}' is not valid.")

    def reset_params(self):
        for param, default in self._init_params.items():
            setattr(self, param, default)
        return self

    def set_figsize(self, figsize: NumericTupleType):
        NumericTupleParam(value=figsize, tuple_sizes=2)
        self.figsize = figsize
        return self

    def set_style(self, style: Optional[str] = None):
        if style is not None:
            LiteralParam(value=style, options=plt.style.available)
        self.style = style
        return self

    def set_tight_layout(self, tight_layout: bool = False):
        BoolParam(value=tight_layout)
        self.tight_layout = tight_layout
        return self

    def set_figure_background(self, figure_background: Optional[ColorType] = None):
        if figure_background is not None:
            ColorParam(value=figure_background)
        self.figure_background = figure_background
        return self

    def set_suptitle(self, t: str, **kwargs):
        StrParam(value=t)
        self.suptitle = dict(t=t, **kwargs)
        return self

    def _prepare_draw(self, clear: bool = False):
        if clear:
            self.clear()
        if self.style is not None:
            self._default_style = plt.rcParams.copy()
            plt.style.use(self.style)
        if not self.figure:
            self.figure = plt.figure(figsize=self.figsize)

    def _finalize_draw(self, show: bool = False):
        if self.tight_layout:
            plt.tight_layout()
        if show:
            self.figure.show()
        if self.style is not None:
            plt.rcParams.update(self._default_style)
        return self.figure

    def clear(self):
        if self.figure:
            plt.close(self.figure)
            self.figure = None
        return self

    def _to_serializable(self, value: Any) -> Any:
        if value is None:
            return None
        elif isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, ndarray):
            return value.tolist()
        elif isinstance(value, Series):
            return value.to_list()
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, Colormap):
            return value.name
        elif isinstance(value, Normalize):
            return value.__class__.__name__.lower()
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, MarkerStyle):
            marker = dict(
                marker=value.get_marker(),
                fillstyle=value.get_fillstyle(),
                capstyle=value.get_capstyle(),
                joinstyle=value.get_joinstyle(),
            )
            return marker

        return value
