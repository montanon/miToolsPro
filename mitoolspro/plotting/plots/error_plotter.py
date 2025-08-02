from typing import Literal, Union

import numpy as np
from matplotlib.axes import Axes
import logging

logger = logging.getLogger(__name__)

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.plotting.plots.plotter import Plotter
from mitoolspro.plotting.plots.validation.functions import (
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
    is_valid_model,
)
from mitoolspro.plotting.plots.validation.models import (
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
    NumericTupleParam,
    NumericTupleSequenceParam,
    NumericTupleSequencesParam,
    StrParam,
)
from mitoolspro.plotting.plots.validation.types import (
    LINESTYLES,
    BoolSequence,
    ColorSequence,
    ColorSequences,
    ColorType,
    EdgeColorSequence,
    EdgeColorType,
    LiteralSequence,
    MarkerSequence,
    MarkerType,
    NumericSequence,
    NumericSequences,
    NumericTupleSequence,
    NumericTupleType,
    NumericType,
)


class ErrorPlotterException(Exception):
    pass


class ErrorPlotter(Plotter):
    def __init__(
        self,
        x_data: Union[NumericSequences, NumericSequence],
        y_data: Union[NumericSequences, NumericSequence],
        ax: Axes = None,
        **kwargs,
    ):
        self._error_params = {
            # General Axes Parameters that are independent of the number of data sequences
            "fmt": None,
            # Specific Parameters that are based on the number of data sequences
            "xerr": None,
            "yerr": None,
            "ecolor": None,
            "elinewidth": None,
            "capsize": 1.0,
            "capthick": 1.0,
            "barsabove": None,
            "lolims": None,
            "uplims": None,
            "xuplims": None,
            "xlolims": None,
            "errorevery": None,
            "marker": "o",
            "markersize": None,
            "markeredgewidth": None,
            "markeredgecolor": None,
            "markerfacecolor": None,
            "linestyle": "",
            "linewidth": None,
        }
        super().__init__(x_data, y_data, ax=ax, **kwargs)
        self._init_params.update(self._error_params)
        self._set_init_params(**kwargs)

    def set_fmt(self, fmt: str):
        StrParam(value=fmt)
        self.fmt = fmt
        return self

    def set_xerr(self, xerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if (
            is_valid_model(NumericSequencesParam, value=xerrs)
            or is_valid_model(NumericSequenceParam, value=xerrs)
            or is_valid_model(NumericParam, value=xerrs)
        ):
            logger.debug("Numeric Sequences Xerr")
            return self.set_numeric_sequences(xerrs, "xerr")
        elif (
            isinstance(xerrs, np.ndarray)
            and xerrs.ndim == 2
            and xerrs.shape[0] == 2
            and xerrs.shape[1] == self.n_sequences
        ):
            self.xerr = xerrs
            logger.debug("Numeric Sequence Xerr 1")
            return self
        elif (
            isinstance(xerrs, np.ndarray)
            and xerrs.ndim == 3
            and xerrs.shape[0] == self.n_sequences
            and xerrs.shape[1] == 2
            and xerrs.shape[2] == self.sizes
        ):
            logger.debug("Numeric Sequence Xerr 2")
            self.xerr = xerrs
            return self
        elif (
            is_valid_model(NumericTupleSequencesParam, value=xerrs)
            or is_valid_model(NumericTupleSequenceParam, value=xerrs)
            or is_valid_model(NumericTupleParam, value=xerrs)
        ):
            return self.set_numeric_tuple_sequences(xerrs, 2, "xerr")
        raise ArgumentValueError(
            "xerrs must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_yerr(self, yerrs: Union[NumericSequences, NumericSequence, NumericType]):
        if (
            is_valid_model(NumericSequencesParam, value=yerrs)
            or is_valid_model(NumericSequenceParam, value=yerrs)
            or is_valid_model(NumericParam, value=yerrs)
        ):
            return self.set_numeric_sequences(yerrs, "yerr")
        elif (
            isinstance(yerrs, np.ndarray)
            and yerrs.ndim == 2
            and yerrs.shape[0] == 2
            and yerrs.shape[1] == self.sizes
        ):
            self.yerr = yerrs
            return self
        elif (
            isinstance(yerrs, np.ndarray)
            and yerrs.ndim == 3
            and yerrs.shape[0] == self.n_sequences
            and yerrs.shape[1] == 2
            and yerrs.shape[2] == self.sizes
        ):
            self.yerr = yerrs
            return self
        elif (
            is_valid_model(NumericTupleSequencesParam, value=yerrs)
            or is_valid_model(NumericTupleSequenceParam, value=yerrs)
            or is_valid_model(NumericTupleParam, value=yerrs)
        ):
            return self.set_numeric_tuple_sequences(yerrs, 2, "yerr")
        raise ArgumentValueError(
            "yerrs must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_ecolor(self, ecolors: Union[ColorSequences, ColorSequence, ColorType]):
        return self.set_color_sequences(ecolors, "ecolor")

    def set_elinewidth(self, elinewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(elinewidths, "elinewidth")

    def set_capsize(self, capsize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(capsize, "capsize")

    def set_capthick(self, capthick: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(capthick, "capthick")

    def set_barsabove(self, barsabove: Union[BoolSequence, bool]):
        return self.set_bool_sequence(barsabove, "barsabove")

    def set_lolims(self, lolims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(lolims, "lolims")

    def set_uplims(self, uplims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(uplims, "uplims")

    def set_xuplims(self, xuplims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(xuplims, "xuplims")

    def set_xlolims(self, xlolims: Union[BoolSequence, bool]):
        return self.set_bool_sequence(xlolims, "xlolims")

    def set_errorevery(
        self,
        errorevery: Union[
            NumericTupleSequence, NumericSequence, NumericTupleType, NumericType
        ],
    ):
        if (
            is_numeric_sequences(errorevery)
            or is_numeric_sequence(errorevery)
            or is_numeric(errorevery)
        ):
            return self.set_numeric_sequences(errorevery, "errorevery")
        elif (
            is_valid_model(NumericTupleSequencesParam, value=errorevery)
            or is_valid_model(NumericTupleSequenceParam, value=errorevery)
            or is_valid_model(NumericTupleParam, value=errorevery)
        ):
            return self.set_numeric_tuple_sequences(errorevery, 2, "errorevery")
        raise ArgumentValueError(
            "errorevery must be numeric or numeric tuple or sequences or sequence of them."
        )

    def set_marker(self, markers: Union[MarkerSequence, MarkerType]):
        return self.set_marker_sequences(markers, param_name="marker")

    def set_markersize(self, markersize: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(markersize, param_name="markersize")

    def set_markeredgewidth(self, markeredgewidth: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(markeredgewidth, param_name="markeredgewidth")

    def set_markeredgecolor(
        self, markeredgecolor: Union[EdgeColorSequence, EdgeColorType]
    ):
        return self.set_edgecolor_sequences(
            markeredgecolor, param_name="markeredgecolor"
        )

    def set_markerfacecolor(self, markerfacecolor: Union[ColorSequence, ColorType]):
        return self.set_color_sequences(markerfacecolor, param_name="markerfacecolor")

    def set_linestyle(
        self,
        linestyles: Union[LiteralSequence, Literal["linestyles"]],
    ):
        return self.set_literal_sequences(
            linestyles, options=LINESTYLES, param_name="linestyle"
        )

    def set_linewidth(self, linewidths: Union[NumericSequence, NumericType]):
        return self.set_numeric_sequences(linewidths, param_name="linewidth")

    def _create_error_kwargs(self, n_sequence: int):
        error_kwargs = {
            "xerr": self.get_sequences_param("xerr", n_sequence),
            "yerr": self.get_sequences_param("yerr", n_sequence),
            "fmt": self.get_sequences_param("fmt", n_sequence),
            "ecolor": self.get_sequences_param("ecolor", n_sequence),
            "elinewidth": self.get_sequences_param("elinewidth", n_sequence),
            "capsize": self.get_sequences_param("capsize", n_sequence),
            "capthick": self.get_sequences_param("capthick", n_sequence),
            "barsabove": self.get_sequences_param("barsabove", n_sequence),
            "lolims": self.get_sequences_param("lolims", n_sequence),
            "uplims": self.get_sequences_param("uplims", n_sequence),
            "xlolims": self.get_sequences_param("xlolims", n_sequence),
            "xuplims": self.get_sequences_param("xuplims", n_sequence),
            "errorevery": self.get_sequences_param("errorevery", n_sequence),
            "alpha": self.get_sequences_param("alpha", n_sequence),
            "color": self.get_sequences_param("color", n_sequence),
            "label": self.get_sequences_param("label", n_sequence),
            "marker": self.get_sequences_param("marker", n_sequence),
            "markersize": self.get_sequences_param("markersize", n_sequence),
            "markeredgewidth": self.get_sequences_param("markeredgewidth", n_sequence),
            "markeredgecolor": self.get_sequences_param("markeredgecolor", n_sequence),
            "markerfacecolor": self.get_sequences_param("markerfacecolor", n_sequence),
            "linestyle": self.get_sequences_param("linestyle", n_sequence),
            "linewidth": self.get_sequences_param("linewidth", n_sequence),
            "zorder": self.get_sequences_param("zorder", n_sequence),
        }
        if (
            not isinstance(error_kwargs.get("alpha", []), NumericType)
            and len(error_kwargs.get("alpha", [])) == 1
        ):
            error_kwargs["alpha"] = error_kwargs["alpha"][0]
        return error_kwargs

    def _create_plot(self):
        for n_sequence in range(self.n_sequences):
            error_kwargs = self._create_error_kwargs(n_sequence)
            error_kwargs = {k: v for k, v in error_kwargs.items() if v is not None}
            try:
                self.ax.errorbar(
                    self.x_data[n_sequence], self.y_data[n_sequence], **error_kwargs
                )
            except Exception as e:
                raise ErrorPlotterException(f"Error while creating error plot: {e}")
        return self.ax
