from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np
from pydantic import ValidationError

from .validator_helper import apply_validators

from mitoolspro.exceptions import ArgumentStructureError
from mitoolspro.plotting.plots.validation.models import (
    BinsParam,
    BinsSequenceParam,
    BoolParam,
    BoolSequenceParam,
    ColormapParam,
    ColormapSequenceParam,
    ColorParam,
    ColorSequenceParam,
    ColorSequencesParam,
    DictParam,
    DictSequenceParam,
    EdgeColorParam,
    EdgeColorSequenceParam,
    EdgeColorSequencesParam,
    LiteralParam,
    LiteralSequenceParam,
    LiteralSequencesParam,
    MarkerParam,
    MarkerSequenceParam,
    MarkerSequencesParam,
    NormalizationParam,
    NormalizationSequenceParam,
    NumericParam,
    NumericSequenceParam,
    NumericSequencesParam,
    NumericTupleParam,
    NumericTupleSequenceParam,
    NumericTupleSequencesParam,
    RangeParam,
    RangeSequenceParam,
    RangeSequencesParam,
    StrParam,
    StrSequenceParam,
    StrSequencesParam,
)
from mitoolspro.plotting.plots.validation.types import (
    BinsSequence,
    BinsType,
    BoolSequence,
    ColormapSequence,
    ColormapType,
    ColorSequence,
    ColorSequences,
    ColorType,
    DictSequence,
    EdgeColorSequence,
    EdgeColorSequences,
    EdgeColorType,
    LiteralSequence,
    LiteralSequences,
    LiteralType,
    MarkerSequence,
    MarkerSequences,
    MarkerType,
    NormalizationSequence,
    NormalizationType,
    NumericSequence,
    NumericSequences,
    NumericTupleSequence,
    NumericTupleSequences,
    NumericTupleType,
    NumericType,
    SizesType,
    StrSequence,
    StrSequences,
)


class SetterMixIn(ABC):
    @property
    @abstractmethod
    def sizes(self) -> SizesType:
        pass

    @property
    @abstractmethod
    def sub_sizes(self) -> SizesType:
        pass

    @property
    @abstractmethod
    def multi_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def n_sequences(self) -> int:
        pass

    def _calculate_sizes(
        self, x_data: Sequence[Sequence[Any]], multi_data: bool
    ) -> tuple[SizesType, SizesType]:
        if multi_data:
            sizes = len(x_data)
            sub_sizes = [len(seq) for seq in x_data]
        else:
            sizes = len(x_data[0])
            sub_sizes = None
        return sizes, sub_sizes

    def set_color_sequences(
        self,
        colors: Union[
            ColorSequences,
            ColorSequence,
            ColorType,
        ],
        param_name: str,
        multi_param: bool = True,
        structured: bool = True,
    ) -> Any:
        validators = []
        if self.multi_data and multi_param:
            validators.append(ColorSequencesParam)
        validators.extend([ColorSequenceParam, ColorParam])

        value, errors = apply_validators(
            colors,
            validators,
            sizes=self.sizes,
            sub_sizes=self.sub_sizes,
            structured=structured,
        )
        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""

        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a color, a sequence of colors, "
                f"or sequences of colors matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a color or sequence of colors "
                f"matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_numeric_sequences(
        self,
        sequences: Union[NumericSequences, NumericSequence, NumericType],
        param_name: str,
        multi_param: bool = True,
        single_param: bool = True,
        min_value: NumericType = None,
        max_value: NumericType = None,
        structured: bool = True,
    ):
        has_range = min_value is not None or max_value is not None
        if has_range:
            min_value = min_value if min_value is not None else -np.inf
            max_value = max_value if max_value is not None else np.inf

        validators = []
        if self.multi_data and multi_param:
            validators.append(
                RangeSequencesParam if has_range else NumericSequencesParam
            )
        validators.append(
            RangeSequenceParam if has_range else NumericSequenceParam
        )
        if single_param:
            validators.append(RangeParam if has_range else NumericParam)

        extra = {}
        if has_range:
            extra["min_value"] = min_value
            extra["max_value"] = max_value

        sizes_val = (
            self.sizes
            if self.multi_data and multi_param
            else self.sizes if single_param else self.n_sequences
        )

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=sizes_val,
            sub_sizes=self.sub_sizes,
            structured=structured,
            **extra,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""

        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a numeric value, a sequence of numeric values, "
                f"or sequences of numeric values matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a numeric value, a sequence of numeric values, "
                f"or sequence of numeric values matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_literal_sequences(
        self,
        sequences: Union[LiteralSequences, LiteralSequence, LiteralType],
        options: Sequence[str],
        param_name: str,
        multi_param: bool = True,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data and multi_param:
            validators.append(LiteralSequencesParam)
        validators.extend([LiteralSequenceParam, LiteralParam])

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=self.sizes,
            sub_sizes=self.sub_sizes,
            structured=structured,
            options=options,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""

        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a literal, a sequence of literals, "
                f"or sequence of literals matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a literal, a sequence of literals, "
                f"or sequence of literals matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_marker_sequences(
        self,
        sequences: Union[MarkerSequences, MarkerSequence, MarkerType],
        param_name: str,
        multi_param: bool = True,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data and multi_param:
            validators.append(MarkerSequencesParam)
        validators.extend([MarkerSequenceParam, MarkerParam])

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=self.sizes,
            sub_sizes=self.sub_sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a marker, a sequence of markers, "
                f"or sequence of markers matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a marker, a sequence of markers, "
                f"or sequence of markers matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_edgecolor_sequences(
        self,
        sequences: Union[EdgeColorSequences, EdgeColorSequence, EdgeColorType],
        param_name: str,
        multi_param: bool = True,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data and multi_param:
            validators.append(EdgeColorSequencesParam)
        validators.extend([EdgeColorSequenceParam, EdgeColorParam])

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=self.sizes,
            sub_sizes=self.sub_sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected an edgecolor, a sequence of edgecolors, "
                f"or sequence of edgecolors matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected an edgecolor, a sequence of edgecolors, "
                f"or sequence of edgecolors matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_str_sequences(
        self,
        sequences: Union[StrSequences, StrSequence, str],
        param_name: str,
        multi_param: bool = True,
        single_param: bool = True,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data and multi_param:
            validators.append(StrSequencesParam)
        validators.append(StrSequenceParam)
        if single_param:
            validators.append(StrParam)

        sizes_val = (
            self.sizes if single_param else self.n_sequences
        )

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=sizes_val,
            sub_sizes=self.sub_sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a string, a sequence of strings, "
                f"or sequence of strings matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a string, a sequence of strings, "
                f"or sequence of strings matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_numeric_tuple_sequences(
        self,
        sequences: Union[NumericTupleSequences, NumericTupleSequence, NumericTupleType],
        tuple_sizes: Union[Sequence[int], int],
        param_name: str,
        multi_param: bool = True,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data and multi_param:
            validators.append(NumericTupleSequencesParam)
        validators.extend([NumericTupleSequenceParam, NumericTupleParam])

        value, errors = apply_validators(
            sequences,
            validators,
            sizes=self.sizes,
            sub_sizes=self.sub_sizes,
            structured=structured,
            tuple_sizes=tuple_sizes,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a numeric tuple, a sequence of numeric tuples, "
                f"or sequence of numeric tuples matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a numeric tuple, a sequence of numeric tuples, "
                f"or sequence of numeric tuples matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_colormap_sequence(
        self,
        sequence: Union[ColormapSequence, ColormapType],
        param_name: str,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data:
            validators.append(ColormapSequenceParam)
        validators.append(ColormapParam)

        value, errors = apply_validators(
            sequence,
            validators,
            sizes=self.sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a colormap, a sequence of colormaps, "
                f"or sequence of colormaps matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a colormap or sequence of colormaps "
                f"matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_norm_sequence(
        self,
        sequence: Union[NormalizationSequence, NormalizationType],
        param_name: str,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data:
            validators.append(NormalizationSequenceParam)
        validators.append(NormalizationParam)

        value, errors = apply_validators(
            sequence,
            validators,
            sizes=self.sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a normalization, a sequence of normalizations, "
                f"or sequence of normalizations matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a normalization, a sequence of normalizations, "
                f"or sequence of normalizations matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_bins_sequence(
        self,
        sequence: Union[BinsSequence, BinsType],
        param_name: str,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data:
            validators.append(BinsSequenceParam)
        validators.append(BinsParam)

        value, errors = apply_validators(
            sequence,
            validators,
            sizes=self.sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a bin, a sequence of bins, "
                f"or sequence of bins matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a bin, a sequence of bins, "
                f"or sequence of bins matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_bool_sequence(
        self,
        sequence: Union[BoolSequence, bool],
        param_name: str,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data:
            validators.append(BoolSequenceParam)
        validators.append(BoolParam)

        value, errors = apply_validators(
            sequence,
            validators,
            sizes=self.sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a boolean, a sequence of booleans, "
                f"or sequence of booleans matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a boolean, a sequence of booleans, "
                f"or sequence of booleans matching the sequence length.\nLast Error: {last_error}"
            )

        raise ArgumentStructureError(msg)

    def set_dict_sequence(
        self,
        sequence: Union[DictSequence, dict],
        param_name: str,
        structured: bool = True,
    ):
        validators = []
        if self.multi_data:
            validators.append(DictSequenceParam)
        validators.append(DictParam)

        value, errors = apply_validators(
            sequence,
            validators,
            sizes=self.sizes,
            structured=structured,
        )

        if errors == [] or value is not None:
            setattr(self, param_name, value)
            return self
        last_error = errors[-1] if errors else ""
        if self.multi_data:
            msg = (
                f"Invalid {param_name}. Expected a dictionary, a sequence of dictionaries, "
                f"or sequence of dictionaries matching the data structure.\nLast Error: {last_error}"
            )
        else:
            msg = (
                f"Invalid {param_name}. Expected a dictionary, a sequence of dictionaries, "
                f"or sequence of dictionaries matching the sequence length.\nLast Error: {last_error}"
            )
        raise ArgumentStructureError(msg)
