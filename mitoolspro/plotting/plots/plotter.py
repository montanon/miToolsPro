import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

from matplotlib.axes import Axes
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentStructureError
from mitoolspro.plotting.plots.matplotlib_typing import (
    Color,
    ColorSequence,
    ColorSequences,
    NumericSequence,
    NumericSequences,
    NumericType,
    StrSequence,
)
from mitoolspro.plotting.plots.plot_params import ParamsMixIn
from mitoolspro.plotting.plots.setter import SetterMixIn
from mitoolspro.plotting.plots.validation.functions import (
    is_numeric,
    is_numeric_sequence,
    is_numeric_sequences,
)
from mitoolspro.plotting.plots.validation.models import (
    DataSequenceParam,
    DataSequencesParam,
)


class PlotterException(Exception):
    pass


class Plotter(ParamsMixIn, SetterMixIn, ABC):
    def __init__(
        self,
        x_data: Union[NumericSequence, NumericSequences],
        y_data: Union[NumericSequence, NumericSequences, None],
        ax: Axes = None,
        **kwargs,
    ):
        self.x_data, self.y_data = self._validate_data(x_data, y_data)
        self._n_sequences = len(self.x_data)
        self._multi_data = self._n_sequences > 1
        self._data_size = max(len(x) for x in self.x_data)
        # Specific Parameters that are based on the number of data sequences
        self._multi_data_params = {
            "color": None,
            "alpha": 1.0,
            "label": None,
            "zorder": None,
        }
        self._multi_params_structure = {}
        super().__init__(ax=ax, **kwargs)
        self._init_params.update(self._multi_data_params)
        self._set_init_params(**kwargs)

    @property
    def data_size(self) -> int:
        return self._data_size

    @property
    def n_sequences(self) -> int:
        return self._n_sequences

    @property
    def multi_data(self) -> bool:
        return self._multi_data

    @property
    def multi_params_structure(self) -> dict:
        return self._multi_params_structure

    def _validate_data(
        self,
        x_data: Union[NumericSequence, NumericSequences],
        y_data: Union[NumericSequence, NumericSequences, None],
    ) -> tuple[NumericSequences, NumericSequences | None]:
        try:
            x_data = DataSequencesParam(value=x_data).value
        except ValidationError:
            try:
                x_data = DataSequenceParam(value=x_data).value
                x_data = [x_data]
            except ValidationError:
                raise ArgumentStructureError(
                    "Invalid x_data, must be a sequence of sequences or a sequence of numeric values"
                )

        if y_data is None:
            return x_data, None
        try:
            y_data = DataSequencesParam(value=y_data).value
        except ValidationError:
            try:
                y_data = DataSequenceParam(value=y_data).value
                y_data = [y_data]
            except ValidationError:
                raise ArgumentStructureError(
                    "Invalid y_data, must be a sequence of sequences or a sequence of numeric values"
                )

        return x_data, y_data

    def set_color(self, color: Union[ColorSequences, ColorSequence, Color]):
        return self.set_color_sequences(color, param_name="color")

    def set_alpha(self, alpha: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(
            alpha, param_name="alpha", min_value=0, max_value=1
        )

    def set_label(self, labels: Union[StrSequence, str]):
        return self.set_str_sequences(labels, param_name="label")

    def set_zorder(self, zorder: Union[NumericSequences, NumericSequence, NumericType]):
        return self.set_numeric_sequences(zorder, param_name="zorder")

    @abstractmethod
    def _create_plot(self):
        raise NotImplementedError

    def draw(self, show: bool = False, clear: bool = True):
        self._prepare_draw(clear=clear)
        try:
            self._create_plot()
        except Exception as e:
            raise PlotterException(f"Error while creating plot: {e}")
        self._apply_common_properties()
        return self._finalize_draw(show)

    def save_plot(
        self,
        file_path: Path,
        dpi: int = 300,
        bbox_inches: str = "tight",
        draw: bool = False,
    ):
        if self.figure or draw:
            if self.figure is None and draw:
                self.draw()
            try:
                self.figure.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            except Exception as e:
                raise PlotterException(f"Error while saving scatter plot: {e}")
        else:
            raise PlotterException("Plot not drawn yet. Call draw() before saving.")
        return self

    def save_plotter(
        self, file_path: Union[str, Path], data: bool = True, return_json: bool = False
    ) -> None:
        init_params = {}
        for param in self._init_params:
            value = getattr(self, param)
            init_params[param] = self._to_serializable(value)
        if data:
            init_params["x_data"] = self._to_serializable(self.x_data)
            init_params["y_data"] = self._to_serializable(self.y_data)
        if return_json:
            return init_params
        with open(file_path, "w") as f:
            json.dump(init_params, f, indent=4)

    def __repr__(self):
        return f"<{self.__class__.__name__}(n_sequences={self.n_sequences}, data_size={self.data_size}, multi_data={self.multi_data})>"

    @classmethod
    def _convert_list_to_tuple(
        cls,
        value: Union[NumericSequences, NumericSequence, None],
        expected_size: Union[Tuple[NumericType], NumericType] = None,
    ) -> Any:
        if value is None:
            return None
        if expected_size is not None and is_numeric(expected_size):
            expected_size = (expected_size,)
        if is_numeric_sequences(value):
            if expected_size is not None:
                if all(len(item) in expected_size for item in value):
                    return [tuple(val) for val in value]
        elif is_numeric_sequence(value):
            if expected_size is not None:
                if len(value) in expected_size:
                    return tuple(value)
        return value

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "Plotter":
        with open(file_path, "r") as f:
            params = json.load(f)
        x_data = params.pop("x_data") if "x_data" in params else None
        y_data = params.pop("y_data") if "y_data" in params else None
        # Convert lists to tuples where needed
        _TUPLE_CONVERSION_KEYS = {
            "xlim": 2,
            "ylim": 2,
            "figsize": 2,
            "center": 2,
            "range": 2,
            "color": (3, 4),
            "whis": 2,
        }
        for key, size in _TUPLE_CONVERSION_KEYS.items():
            if key in params:
                params[key] = cls._convert_list_to_tuple(params[key], size)
        return cls(x_data=x_data, y_data=y_data, **params)
