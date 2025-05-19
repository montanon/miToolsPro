from typing import Literal, Union

import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray
from pandas import Series
from scipy import stats

from mitoolspro.exceptions import ArgumentTypeError
from mitoolspro.plotting.plots.plotter import Plotter
from mitoolspro.plotting.plots.validation.models import LiteralParam, RangeParam
from mitoolspro.plotting.plots.validation.types import (
    BANDWIDTH_METHODS,
    HATCHES,
    KERNELS,
    LINESTYLES,
    ORIENTATIONS,
    BoolSequence,
    ColorSequence,
    ColorType,
    LiteralSequence,
    NumericSequence,
    NumericSequences,
    NumericType,
)


class DistributionPlotterException(Exception):
    pass


class DistributionPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: None = None,
        ax: Axes = None,
        **kwargs,
    ):
        super().__init__(x_data=x_data, y_data=None, ax=ax, **kwargs)
        self._dist_params = {
            # General Axes.scatter Parameters that are independent of the number of data sequences
            "kernel": "gaussian",
            "bandwidth": "scott",
            "gridsize": 1_000,
            "cut": 3,
            "orientation": "vertical",
            # Specific Parameters that are based on the number of data sequences
            "fill": True,
            "linestyle": "-",
            "linewidth": None,
            "facecolor": None,
            "hatch": None,
        }
        self._init_params.update(self._dist_params)
        self._set_init_params(**kwargs)

    def set_kernel(self, kernel: str):
        LiteralParam(value=kernel, options=KERNELS)
        self.kernel = kernel
        return self

    def set_bandwidth(self, bandwidth: Union[Literal["bandwidth_methods"], float]):
        if isinstance(bandwidth, str):
            LiteralParam(value=bandwidth, options=BANDWIDTH_METHODS)
            self.bandwidth = bandwidth
        elif isinstance(bandwidth, NumericType):
            RangeParam(value=bandwidth, min_value=1e-9, max_value=np.inf)
            self.bandwidth = float(bandwidth)
        return self

    def set_gridsize(self, gridsize: NumericType):
        RangeParam(value=gridsize, min_value=1, max_value=np.inf)
        self.gridsize = int(gridsize)
        return self

    def set_cut(self, cut: NumericType):
        RangeParam(value=cut, min_value=0, max_value=np.inf)
        self.cut = float(cut)
        return self

    def set_orientation(self, orientation: Literal["horizontal", "vertical"]):
        LiteralParam(value=orientation, options=ORIENTATIONS)
        self.orientation = orientation
        return self

    def set_fill(self, fill: Union[BoolSequence, bool]):
        return self.set_bool_sequence(fill, "fill")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequences(linestyles, LINESTYLES, "linestyles")

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(linewidths, "linewidth")

    def set_facecolor(self, facecolors: Union[ColorSequence, ColorType]):
        return self.set_color_sequences(facecolors, "facecolor")

    def set_hatch(self, hatches: Union[LiteralSequence, Literal["hatches"]]):
        return self.set_literal_sequences(hatches, HATCHES, "hatch")

    def _compute_kde(self, data):
        kde = stats.gaussian_kde(
            data,
            bw_method=self.bandwidth,
        )
        if self.orientation == "vertical":
            grid_min = min(data) - self.cut * kde.covariance_factor()
            grid_max = max(data) + self.cut * kde.covariance_factor()
        else:
            grid_min = min(data) - self.cut * kde.covariance_factor()
            grid_max = max(data) + self.cut * kde.covariance_factor()
        grid = np.linspace(grid_min, grid_max, self.gridsize)
        kde_values = kde(grid)
        return grid, kde_values

    def _create_dist_kwargs(self, n_sequence: int):
        dist_kwargs = {
            "color": self.get_sequences_param("color", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        return dist_kwargs

    def _create_fill_kwargs(self, n_sequence: int):
        fill_kwargs = {
            "facecolor": self.get_sequences_param("facecolor", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "hatch": self.get_sequences_param("hatch", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
            "edgecolor": self.get_sequences_param("color", n_sequence),
        }
        return fill_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            try:
                if isinstance(self.x_data[n_sequence], (list, tuple, ndarray, Series)):
                    grid, density = self._compute_kde(self.x_data[n_sequence])
                else:
                    raise ArgumentTypeError("Data must be array-like")
                plot_kwargs = self._create_dist_kwargs(n_sequence)
                fill = self.get_sequences_param("fill", n_sequence)
                orientation = self.get_sequences_param("orientation", n_sequence)
                if fill:
                    fill_kwargs = self._create_fill_kwargs(n_sequence)
                    if orientation == "vertical":
                        self.ax.fill_between(grid, density, **fill_kwargs)
                    else:
                        self.ax.fill_betweenx(grid, density, **fill_kwargs)
                if orientation == "vertical":
                    self.ax.plot(grid, density, **plot_kwargs)
                else:
                    self.ax.plot(density, grid, **plot_kwargs)
            except Exception as e:
                raise DistributionPlotterException(
                    f"Error while creating distribution plot: {e}"
                )
