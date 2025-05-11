from typing import Literal, Union

from matplotlib.axes import Axes

from mitoolspro.plotting.plots.plotter import Plotter
from mitoolspro.plotting.plots.validation.types import (
    LINESTYLES,
    ColorSequence,
    ColorType,
    EdgeColorSequence,
    EdgeColorType,
    LiteralSequence,
    MarkerSequence,
    MarkerType,
    NumericSequence,
    NumericSequences,
    NumericType,
)


class LinePlotterException(Exception):
    pass


class LinePlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        ax: Axes = None,
        **kwargs,
    ):
        self._line_params = {
            # Specific Parameters that are based on the number of data sequences
            "marker": "",
            "markersize": None,
            "markeredgewidth": None,
            "markeredgecolor": None,
            "markerfacecolor": None,
            "linestyle": "-",
            "linewidth": None,
        }
        super().__init__(x_data, y_data, ax=ax, **kwargs)
        self._init_params.update(self._line_params)
        self._set_init_params(**kwargs)

    def set_marker(self, markers: Union[MarkerSequence, MarkerType]):
        return self.set_marker_sequences(
            markers, param_name="marker", multi_param=False
        )

    def set_markersize(self, markersize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(
            markersize, param_name="markersize", multi_param=False
        )

    def set_markeredgewidth(self, markeredgewidth: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(
            markeredgewidth, param_name="markeredgewidth", multi_param=False
        )

    def set_markeredgecolor(
        self, markeredgecolor: Union[EdgeColorSequence, EdgeColorType]
    ):
        return self.set_edgecolor_sequences(
            markeredgecolor, param_name="markeredgecolor", multi_param=False
        )

    def set_markerfacecolor(self, markerfacecolor: Union[ColorSequence, ColorType]):
        return self.set_color_sequences(
            markerfacecolor, param_name="markerfacecolor", multi_param=False
        )

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequences(
            linestyles, options=LINESTYLES, param_name="linestyle", multi_param=False
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(
            linewidths, param_name="linewidth", multi_param=False
        )

    def _create_line_kwargs(self, n_sequence: int):
        line_kwargs = {
            "color": self.get_sequences_param("color", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "markersize": self.get_sequences_param("markersize", n_sequence),
            "markerfacecolor": self.get_sequences_param("markerfacecolor", n_sequence),
            "markeredgecolor": self.get_sequences_param("markeredgecolor", n_sequence),
            "markeredgewidth": self.get_sequences_param("markeredgewidth", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(line_kwargs.get("alpha", []), NumericType)
            and len(line_kwargs.get("alpha", [])) == 1
        ):
            line_kwargs["alpha"] = line_kwargs["alpha"][0]
        return line_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            plot_kwargs = self._create_line_kwargs(n_sequence)
            plot_kwargs = {k: v for k, v in plot_kwargs.items() if v is not None}
            try:
                self.ax.plot(
                    self.x_data[n_sequence], self.y_data[n_sequence], **plot_kwargs
                )
            except Exception as e:
                raise LinePlotterException(f"Error while creating line plot: {e}")
        return self.ax
