from typing import Literal, Sequence, Union

from matplotlib.axes import Axes

from mitoolspro.plotting.plots.plotter import Plotter
from mitoolspro.plotting.plots.validation.models import BoolParam, LiteralParam
from mitoolspro.plotting.plots.validation.types import (
    HATCHES,
    HIST_ALIGN,
    HIST_HISTTYPE,
    LINESTYLES,
    ORIENTATIONS,
    BinsSequence,
    BinsType,
    ColorSequence,
    ColorType,
    EdgeColorSequence,
    EdgeColorType,
    LiteralSequence,
    NumericSequence,
    NumericSequences,
    NumericTupleSequence,
    NumericTupleType,
    NumericType,
)


class HistogramPlotterException(Exception):
    pass


class HistogramPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: None = None,
        ax: Axes = None,
        **kwargs,
    ):
        super().__init__(x_data=x_data, y_data=None, ax=ax, **kwargs)
        self._hist_params = {
            # General Axes.scatter Parameters that are independent of the number of data sequences
            "orientation": "vertical",
            "stacked": False,
            "log": False,
            # Specific Parameters that are based on the number of data sequences
            "bins": "auto",
            "range": None,
            "weights": None,
            "cumulative": False,
            "bottom": None,
            "histtype": "bar",
            "align": "mid",
            "rwidth": None,
            "edgecolor": None,
            "facecolor": None,
            "fill": True,
            "linestyle": "-",
            "linewidth": None,
            "hatch": None,
        }
        self._init_params.update(self._hist_params)
        self._set_init_params(**kwargs)

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        LiteralParam(value=orientation, options=ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_stacked(self, stacked: bool):
        BoolParam(value=stacked)
        self.stacked = stacked
        return self

    def set_log(self, log: bool):
        BoolParam(value=log)
        self.log = log
        return self

    def set_bins(self, bins: Union[BinsSequence, BinsType]):
        return self.set_bins_sequence(bins, "bins")

    def set_range(self, range: Union[NumericTupleSequence, NumericTupleType, None]):
        return self.set_numeric_tuple_sequences(range, 2, "range", multi_param=False)

    def set_weights(self, weights: Union[NumericSequences, NumericSequence, None]):
        return self.set_numeric_sequences(weights, "weights")

    def set_cumulative(self, cumulative: Union[Sequence[bool], bool]):
        return self.set_bool_sequence(cumulative, "cumulative")

    def set_bottom(
        self, bottom: Union[NumericSequences, NumericSequence, NumericType, None]
    ):
        return self.set_numeric_sequences(bottom, "bottom")

    def set_histtype(
        self,
        histtype: Union[
            LiteralSequence,
            Literal["bar", "barstacked", "step", "stepfilled"],
        ],
    ):
        return self.set_literal_sequences(
            histtype, HIST_HISTTYPE, "histtype", multi_param=False
        )

    def set_align(self, align: Union[LiteralSequence, Literal["left", "mid", "right"]]):
        return self.set_literal_sequences(align, HIST_ALIGN, "align", multi_param=False)

    def set_rwidth(self, rwidth: Union[NumericSequence, NumericType, None]):
        return self.set_numeric_sequences(rwidth, "rwidth", multi_param=False)

    def set_edgecolor(self, edgecolors: Union[EdgeColorSequence, EdgeColorType]):
        return self.set_edgecolor_sequences(edgecolors, "edgecolor", multi_param=False)

    def set_facecolor(self, facecolors: Union[ColorSequence, ColorType]):
        return self.set_color_sequences(facecolors, "facecolor", multi_param=False)

    def set_fill(self, fill: Union[Sequence[bool], bool]):
        return self.set_bool_sequence(fill, "fill")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequences(
            linestyles, LINESTYLES, "linestyle", multi_param=False
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(linewidths, "linewidth", multi_param=False)

    def set_hatch(self, hatches: Union[LiteralSequence, Literal["hatches"]]):
        return self.set_literal_sequences(hatches, HATCHES, "hatch", multi_param=False)

    def _create_hist_kwargs(self, n_sequence: int):
        hist_kwargs = {
            "orientation": self.orientation,
            "stacked": self.stacked,
            "log": self.log,
            "bins": self.get_sequences_param("bins", n_sequence),
            "range": self.get_sequences_param("range", n_sequence),
            "weights": self.get_sequences_param("weights", n_sequence),
            "cumulative": self.get_sequences_param("cumulative", n_sequence),
            "bottom": self.get_sequences_param("bottom", n_sequence),
            "histtype": self.get_sequences_param("histtype", n_sequence),
            "align": self.get_sequences_param("align", n_sequence),
            "rwidth": self.get_sequences_param("rwidth", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "edgecolor": self.get_sequences_param("edgecolor", n_sequence),
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "fill": self.get_sequences_param("fill", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(hist_kwargs.get("alpha", []), NumericType)
            and len(hist_kwargs.get("alpha", [])) == 1
        ):
            hist_kwargs["alpha"] = hist_kwargs["alpha"][0]
        return hist_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            hist_kwargs = self._create_hist_kwargs(n_sequence)
            hist_kwargs = {k: v for k, v in hist_kwargs.items() if v is not None}
            try:
                self.ax.hist(self.x_data[n_sequence], **hist_kwargs)
            except Exception as e:
                raise HistogramPlotterException(f"Error while creating histogram: {e}")
