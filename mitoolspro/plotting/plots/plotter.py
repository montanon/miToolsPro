import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple, Union

from matplotlib.axes import Axes
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentStructureError
from mitoolspro.plotting.plots.plot_params import ParamsMixIn
from mitoolspro.plotting.plots.setter import SetterMixIn
from mitoolspro.plotting.plots.validation.functions import is_valid_model
from mitoolspro.plotting.plots.validation.models import (
    DataSequenceParam,
    DataSequencesParam,
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
    Param,
    SequenceParam,
    SequencesParam,
)
from mitoolspro.plotting.plots.validation.types import (
    ColorSequence,
    ColorSequences,
    ColorType,
    NumericSequence,
    NumericSequences,
    NumericType,
    StrSequence,
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
        self._multi_data = len(self.x_data) > 1
        self._sizes, self._sub_sizes = self._calculate_sizes(
            self.x_data, self.multi_data
        )
        self._n_sequences = len(self.x_data) if self.multi_data else 1
        # Specific Parameters that are based on the number of data sequences
        self._multi_data_params = {
            "color": None,
            "alpha": 1.0,
            "label": None,
            "zorder": None,
        }
        super().__init__(ax=ax, **kwargs)
        self._init_params.update(self._multi_data_params)
        self._set_init_params(**kwargs)

    @property
    def sizes(self):
        return self._sizes

    @property
    def sub_sizes(self):
        return self._sub_sizes

    @property
    def multi_data(self) -> bool:
        return self._multi_data

    @property
    def n_sequences(self) -> int:
        return self._n_sequences

    def _validate_data(
        self,
        x_data: Union[NumericSequence, NumericSequences],
        y_data: Union[NumericSequence, NumericSequences, None],
    ) -> tuple[NumericSequences, NumericSequences | None]:
        if isinstance(x_data, list) and not x_data:
            raise ArgumentStructureError("x_data cannot be empty")
        if isinstance(y_data, list) and not y_data:
            raise ArgumentStructureError("y_data cannot be empty")
        try:
            x_data = DataSequencesParam(value=x_data).value
            x_data_size = len(x_data)
            x_data_sub_sizes = [len(x) for x in x_data]
        except ValidationError:
            try:
                x_data = DataSequenceParam(value=x_data).value
                x_data_size = len(x_data)
                x_data_sub_sizes = None
                x_data = [x_data]
            except ValidationError:
                raise ArgumentStructureError(
                    "Invalid x_data, must be a sequence of sequences or a sequence of numeric values"
                )

        if y_data is None:
            return x_data, None
        try:
            y_data = DataSequencesParam(
                value=y_data,
                sizes=x_data_size,
                sub_sizes=x_data_sub_sizes,
            ).value
        except ValidationError:
            try:
                y_data = DataSequenceParam(
                    value=y_data, sizes=x_data_size, sub_sizes=x_data_sub_sizes
                ).value
                y_data = [y_data]
            except ValidationError:
                raise ArgumentStructureError(
                    "Invalid y_data, must be a sequence of sequences or a sequence of numeric values"
                )
        if len(x_data) != len(y_data):
            raise ArgumentStructureError("x_data and y_data must have the same length")
        if not all(len(x) == len(y) for x, y in zip(x_data, y_data)):
            raise ArgumentStructureError(
                "x_data and y_data must have the same sub structure"
            )
        return x_data, y_data

    def set_color(self, color: Union[ColorSequences, ColorSequence, ColorType]):
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
        return f"<{self.__class__.__name__}(size={self.sizes}, sub_sizes={self.sub_sizes}, multi_data={self.multi_data})>"

    @classmethod
    def _convert_list_to_tuple(
        cls,
        value: Union[NumericSequences, NumericSequence, None],
        expected_size: Union[Tuple[NumericType], NumericType] = None,
    ) -> Any:
        if value is None:
            return None
        if expected_size is not None and is_valid_model(
            NumericParam, value=expected_size
        ):
            expected_size = (expected_size,)
        if is_valid_model(NumericSequencesParam, value=value):
            if expected_size is not None:
                if all(len(item) in expected_size for item in value):
                    return [tuple(val) for val in value]
        elif is_valid_model(NumericSequenceParam, value=value):
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

    def get_sequences_param(self, param_name: str, n_sequence: int):
        param_value = getattr(self, param_name)
        if self._multi_data:
            if not isinstance(param_value, tuple) and (
                is_valid_model(SequencesParam[Any], value=param_value)
                or is_valid_model(SequenceParam, value=param_value)
            ):
                return param_value[n_sequence]
            elif is_valid_model(Param[Any], value=param_value):
                return param_value
        return param_value
