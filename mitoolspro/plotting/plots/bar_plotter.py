from typing import Dict, Literal, Sequence, Union

import numpy as np
from matplotlib.axes import Axes

from mitoolspro.plotting.plots.plotter import Plotter
from mitoolspro.plotting.plots.validation.models import BoolParam, LiteralParam
from mitoolspro.plotting.plots.validation.types import (
    BARS_ALIGN,
    HATCHES,
    LINESTYLES,
    ORIENTATIONS,
    ColorSequence,
    ColorSequences,
    ColorType,
    DictSequence,
    EdgeColorSequence,
    EdgeColorSequences,
    EdgeColorType,
    LiteralSequence,
    LiteralSequences,
    NumericSequence,
    NumericSequences,
    NumericType,
)


class BarPlotterException(Exception):
    pass


class BarPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        kind: Literal["bar", "stacked"] = "bar",
        ax: Axes = None,
        **kwargs,
    ):
        self._bar_params = {
            # General Axes Parameters that are independent of the number of data sequences
            "log": False,
            "orientation": "vertical",
            # Specific Parameters that are based on the number of data sequences
            "width": 0.8,
            "bottom": None,
            "align": "center",
            "edgecolor": None,
            "linewidth": None,
            "xerr": None,
            "yerr": None,
            "ecolor": None,
            "capsize": None,
            "error_kw": None,
            "facecolor": None,
            "fill": True,
            "hatch": None,
            "linestyle": "-",
        }
        super().__init__(x_data, y_data, ax=ax, **kwargs)
        self._init_params.update(self._bar_params)
        self._set_init_params(**kwargs)
        self._kind = kind

    @property
    def kind(self):
        return self._kind

    def set_log(self, log: bool):
        BoolParam(value=log)
        self.log = log
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        LiteralParam(value=orientation, options=ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_width(self, widths: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(widths, "width")

    def set_bottom(
        self, bottoms: Union[NumericSequences, NumericSequence, NumericType]
    ):
        return self.set_numeric_sequences(bottoms, "bottom")

    def set_align(self, align: Union[LiteralSequence, Literal["center", "edge"]]):
        return self.set_literal_sequences(align, BARS_ALIGN, "align")

    def set_edgecolor(
        self, edgecolors: Union[EdgeColorSequences, EdgeColorSequence, EdgeColorType]
    ):
        return self.set_edgecolor_sequences(edgecolors, "edgecolor")

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(linewidths, "linewidth")

    def set_xerr(self, xerrs: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(xerrs, "xerr")

    def set_yerr(self, yerrs: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(yerrs, "yerr")

    def set_ecolor(self, ecolors: Union[ColorSequences, ColorSequence, ColorType]):
        return self.set_color_sequences(ecolors, "ecolor")

    def set_capsize(self, capsize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(capsize, "capsize")

    def set_error_kw(self, error_kw: Union[DictSequence, Dict]):
        return self.set_dict_sequence(error_kw, "error_kw")

    def set_facecolor(self, facecolors: Union[ColorSequence, ColorType]):
        return self.set_color_sequences(facecolors, "facecolor")

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        return self.set_bool_sequence(fill, "fill")

    def set_hatch(
        self, hatches: Union[LiteralSequences, LiteralSequence, Literal["hatches"]]
    ):
        return self.set_literal_sequences(hatches, HATCHES, "hatch")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequences(linestyles, LINESTYLES, "linestyle")

    def _create_bar_kwargs(self, n_sequence: int):
        bar_kwargs = {
            "width": self.get_sequences_param("width", n_sequence),
            "bottom": self.get_sequences_param("bottom", n_sequence),
            "align": self.get_sequences_param("align", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "edgecolor": self.get_sequences_param("edgecolor", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "xerr": self.get_sequences_param("xerr", n_sequence),
            "yerr": self.get_sequences_param("yerr", n_sequence),
            "ecolor": self.get_sequences_param("ecolor", n_sequence),
            "capsize": self.get_sequences_param("capsize", n_sequence),
            "error_kw": self.get_sequences_param("error_kw", n_sequence),
            "log": self.log,
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "fill": self.get_sequences_param("fill", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(bar_kwargs.get("alpha", []), NumericType)
            and len(bar_kwargs.get("alpha", [])) == 1
        ):
            bar_kwargs["alpha"] = bar_kwargs["alpha"][0]
        return bar_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            bar_kwargs = self._create_bar_kwargs(n_sequence)
            bar_kwargs = {k: v for k, v in bar_kwargs.items() if v is not None}
            if self.kind == "stacked":
                if n_sequence == 0:
                    bottom_reference = bar_kwargs.get(
                        "bottom", np.zeros_like(self.y_data[n_sequence])
                    )
                bar_kwargs["bottom"] = bottom_reference
            try:
                if self.orientation == "vertical":
                    bar_kwargs["x"] = self.x_data[n_sequence]
                    bar_kwargs["height"] = self.y_data[n_sequence]
                    self.ax.bar(
                        **bar_kwargs,
                    )
                else:
                    bar_kwargs["y"] = self.x_data[n_sequence]
                    bar_kwargs["width"], bar_kwargs["height"] = (
                        self.y_data[n_sequence],
                        bar_kwargs["width"],
                    )
                    if "bottom" in bar_kwargs:
                        bar_kwargs["left"] = bar_kwargs.pop("bottom")
                    self.ax.barh(
                        **bar_kwargs,
                    )
                if self.kind == "stacked":
                    bottom_reference += self.y_data[n_sequence]
            except Exception as e:
                raise BarPlotterException(f"Error while creating bar plot: {e}")
        return self.ax
