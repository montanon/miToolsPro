from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np
from pydantic import ValidationError

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
        if self.multi_data and multi_param:
            try:
                validated = ColorSequencesParam(
                    value=colors,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, validated)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            validated = ColorSequenceParam(
                value=colors,
                sizes=self.sizes,
                structured=structured,
            ).value
            setattr(self, param_name, validated)
            return self
        except ValidationError as e:
            last_error = str(e)
        try:
            validated = ColorParam(value=colors).value
            setattr(self, param_name, validated)
            return self
        except ValidationError as e:
            last_error = str(e)

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
        no_range = not has_range

        if self.multi_data and multi_param:
            try:
                sequences = (
                    NumericSequencesParam(
                        value=sequences,
                        sizes=self.sizes,
                        sub_sizes=self.sub_sizes,
                        structured=structured,
                    )
                    if no_range
                    else RangeSequencesParam(
                        value=sequences,
                        sizes=self.sizes,
                        sub_sizes=self.sub_sizes,
                        min_value=min_value,
                        max_value=max_value,
                        structured=structured,
                    )
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = (
                NumericSequenceParam(
                    value=sequences,
                    sizes=self.sizes if single_param else self.n_sequences,
                    structured=structured,
                )
                if no_range
                else RangeSequenceParam(
                    value=sequences,
                    sizes=self.sizes if single_param else self.n_sequences,
                    min_value=min_value,
                    max_value=max_value,
                    structured=structured,
                )
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        if single_param:
            try:
                sequences = (
                    NumericParam(value=sequences)
                    if no_range
                    else RangeParam(
                        value=sequences, min_value=min_value, max_value=max_value
                    )
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)

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
        if self.multi_data and multi_param:
            try:
                sequences = LiteralSequencesParam(
                    value=sequences,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    options=options,
                    structured=structured,
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = LiteralSequenceParam(
                value=sequences,
                sizes=self.sizes,
                options=options,
                structured=structured,
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        try:
            sequences = LiteralParam(value=sequences, options=options).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)

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
        if self.multi_data and multi_param:
            try:
                sequences = MarkerSequencesParam(
                    value=sequences,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = MarkerSequenceParam(
                value=sequences,
                sizes=self.sizes,
                structured=structured,
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        try:
            sequences = MarkerParam(value=sequences).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data and multi_param:
            try:
                sequences = EdgeColorSequencesParam(
                    value=sequences,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = EdgeColorSequenceParam(
                value=sequences,
                sizes=self.sizes,
                structured=structured,
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        try:
            sequences = EdgeColorParam(value=sequences).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data and multi_param:
            try:
                sequences = StrSequencesParam(
                    value=sequences,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = StrSequenceParam(
                value=sequences,
                sizes=self.sizes if single_param else self.n_sequences,
                structured=structured,
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        if single_param:
            try:
                sequences = StrParam(value=sequences).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
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
        if self.multi_data and multi_param:
            try:
                sequences = NumericTupleSequencesParam(
                    value=sequences,
                    sizes=self.sizes,
                    sub_sizes=self.sub_sizes,
                    tuple_sizes=tuple_sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequences)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequences = NumericTupleSequenceParam(
                value=sequences,
                sizes=self.sizes,
                tuple_sizes=tuple_sizes,
                structured=structured,
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
        try:
            sequences = NumericTupleParam(
                value=sequences, tuple_sizes=tuple_sizes
            ).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data:
            try:
                sequence = ColormapSequenceParam(
                    value=sequence,
                    sizes=self.sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequence)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequence = ColormapParam(value=sequence).value
            setattr(self, param_name, sequence)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data:
            try:
                sequence = NormalizationSequenceParam(
                    value=sequence,
                    sizes=self.sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequence)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequence = NormalizationParam(value=sequence).value
            setattr(self, param_name, sequence)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data:
            try:
                sequence = BinsSequenceParam(
                    value=sequence, sizes=self.sizes, structured=structured
                ).value
                setattr(self, param_name, sequence)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequence = BinsParam(value=sequence).value
            setattr(self, param_name, sequence)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data:
            try:
                sequence = BoolSequenceParam(
                    value=sequence,
                    sizes=self.sizes,
                    structured=structured,
                ).value
                setattr(self, param_name, sequence)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequence = BoolParam(value=sequence).value
            setattr(self, param_name, sequence)
            return self
        except ValidationError as e:
            last_error = str(e)
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
        if self.multi_data:
            try:
                sequence = DictSequenceParam(
                    value=sequence, sizes=self.sizes, structured=structured
                ).value
                setattr(self, param_name, sequence)
                return self
            except ValidationError as e:
                last_error = str(e)
        try:
            sequence = DictParam(value=sequence).value
            setattr(self, param_name, sequence)
            return self
        except ValidationError as e:
            last_error = str(e)
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
