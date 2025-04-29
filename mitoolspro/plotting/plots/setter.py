from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentStructureError
from mitoolspro.plotting.plots.validation.models import (
    BinsParam,
    BinsSequence,
    BinsSequenceParam,
    BinsType,
    BoolParam,
    BoolSequence,
    BoolSequenceParam,
    ColormapParam,
    ColormapSequence,
    ColormapSequenceParam,
    ColormapType,
    ColorParam,
    ColorSequence,
    ColorSequenceParam,
    ColorSequences,
    ColorSequencesParam,
    ColorType,
    DictParam,
    DictSequence,
    DictSequenceParam,
    EdgeColorParam,
    EdgeColorSequence,
    EdgeColorSequenceParam,
    EdgeColorSequences,
    EdgeColorSequencesParam,
    EdgeColorType,
    LiteralParam,
    LiteralSequence,
    LiteralSequenceParam,
    LiteralSequences,
    LiteralSequencesParam,
    LiteralType,
    MarkerParam,
    MarkerSequence,
    MarkerSequenceParam,
    MarkerSequences,
    MarkerSequencesParam,
    MarkerType,
    NormalizationParam,
    NormalizationSequence,
    NormalizationSequenceParam,
    NormalizationType,
    NumericParam,
    NumericSequence,
    NumericSequenceParam,
    NumericSequences,
    NumericSequencesParam,
    NumericTupleParam,
    NumericTupleSequence,
    NumericTupleSequenceParam,
    NumericTupleSequences,
    NumericTupleSequencesParam,
    NumericTupleType,
    NumericType,
    RangeParam,
    RangeSequenceParam,
    RangeSequencesParam,
    StrParam,
    StrSequence,
    StrSequenceParam,
    StrSequences,
    StrSequencesParam,
)
from mitoolspro.plotting.plots.validations import (
    validate_sequence_length,
    validate_subsequences_length,
)


class Setter(ABC):
    @property
    @abstractmethod
    def data_size(self) -> int:
        pass

    @property
    @abstractmethod
    def n_sequences(self) -> int:
        pass

    @property
    @abstractmethod
    def multi_data(self) -> bool:
        pass

    @property
    @abstractmethod
    def multi_params_structure(self) -> dict:
        pass

    def set_color_sequences(
        self,
        colors: Union[
            ColorSequences,
            ColorSequence,
            ColorType,
        ],
        param_name: str,
    ) -> Any:
        if self.multi_data:
            try:
                validated = ColorSequencesParam(value=colors).value
                validate_sequence_length(validated, self.n_sequences, param_name)
                validate_subsequences_length(validated, [1, self.data_size], param_name)
                setattr(self, param_name, validated)
                return self
            except ValidationError:
                pass
        try:
            validated = ColorSequenceParam(value=colors).value
            validate_sequence_length(
                validated,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, validated)
            return self
        except ValidationError:
            pass
        try:
            validated = ColorParam(value=colors).value
            setattr(self, param_name, validated)
            return self
        except ValidationError:
            pass

        if self.multi_data:
            msg = f"Invalid {param_name}, must be a color, sequence of colors, or sequences of colors."
        else:
            msg = f"Invalid {param_name}, must be a color or sequence of colors."

        raise ArgumentStructureError(msg)

    def set_numeric_sequences(
        self,
        sequences: Union[NumericSequences, NumericSequence, NumericType],
        param_name: str,
        min_value: NumericType = None,
        max_value: NumericType = None,
    ):
        has_range = min_value is not None or max_value is not None
        if has_range:
            min_value = min_value if min_value is not None else -np.inf
            max_value = max_value if max_value is not None else np.inf
        no_range = not has_range

        if self.multi_data:
            try:
                sequences = (
                    NumericSequencesParam(value=sequences)
                    if no_range
                    else RangeSequencesParam(
                        value=sequences, min_value=min_value, max_value=max_value
                    )
                ).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = (
                NumericSequenceParam(value=sequences)
                if no_range
                else RangeSequenceParam(
                    value=sequences, min_value=min_value, max_value=max_value
                )
            ).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
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
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a numeric value, numeric sequences, or sequence of numeric sequences."
        )

    def set_literal_sequences(
        self,
        sequences: Union[LiteralSequences, LiteralSequence, LiteralType],
        options: Sequence[str],
        param_name: str,
    ):
        if self.multi_data:
            try:
                sequences = LiteralSequencesParam(
                    value=sequences, options=options
                ).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = LiteralSequenceParam(value=sequences, options=options).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        try:
            sequences = LiteralParam(value=sequences, options=options).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a literal or sequence of literals."
        )

    def set_marker_sequences(
        self,
        sequences: Union[MarkerSequences, MarkerSequence, MarkerType],
        param_name: str,
    ):
        if self.multi_data:
            try:
                sequences = MarkerSequencesParam(value=sequences).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = MarkerSequenceParam(value=sequences).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        try:
            sequences = MarkerParam(value=sequences).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a marker, sequence of markers, or sequences of markers."
        )

    def set_edgecolor_sequences(
        self,
        sequences: Union[EdgeColorSequences, EdgeColorSequence, EdgeColorType],
        param_name: str,
    ):
        if self.multi_data:
            try:
                sequences = EdgeColorSequencesParam(value=sequences).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = EdgeColorSequenceParam(value=sequences).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        try:
            sequences = EdgeColorParam(value=sequences).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be an edgecolor, sequence of edgecolors, or sequences of edgecolors."
        )

    def set_str_sequences(
        self, sequences: Union[StrSequences, StrSequence, str], param_name: str
    ):
        if self.multi_data:
            try:
                sequences = StrSequencesParam(value=sequences).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = StrSequenceParam(value=sequences).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        try:
            sequences = StrParam(value=sequences).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a string, sequence of strings, or sequences of strings."
        )

    def set_numeric_tuple_sequences(
        self,
        sequences: Union[NumericTupleSequences, NumericTupleSequence, NumericTupleType],
        sizes: Union[Sequence[int], int],
        param_name: str,
    ):
        if self.multi_data:
            try:
                sequences = NumericTupleSequencesParam(
                    value=sequences, sizes=sizes
                ).value
                validate_sequence_length(sequences, self.n_sequences, param_name)
                validate_subsequences_length(sequences, [1, self.data_size], param_name)
                setattr(self, param_name, sequences)
                return self
            except ValidationError:
                pass
        try:
            sequences = NumericTupleSequenceParam(value=sequences, sizes=sizes).value
            validate_sequence_length(
                sequences,
                self.n_sequences if self.multi_data else self.data_size,
                param_name,
            )
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        try:
            sequences = NumericTupleParam(value=sequences, sizes=sizes).value
            setattr(self, param_name, sequences)
            return self
        except ValidationError:
            pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a numeric tuple, sequence of numeric tuples, or sequences of numeric tuples."
        )

    def set_colormap_sequence(
        self, sequence: Union[ColormapSequence, ColormapType], param_name: str
    ):
        if self.multi_data:
            try:
                sequence = ColormapSequenceParam(value=sequence).value
                validate_sequence_length(sequence, self.n_sequences, param_name)
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "sequence"
                return self
            except ValidationError:
                pass
        else:
            try:
                sequence = ColormapParam(value=sequence).value
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "value"
                return self
            except ValidationError:
                pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a colormap, sequence of colormaps, or sequences of colormaps."
        )

    def set_norm_sequence(
        self, sequence: Union[NormalizationSequence, NormalizationType], param_name: str
    ):
        if self.multi_data:
            try:
                sequence = NormalizationSequenceParam(value=sequence).value
                validate_sequence_length(sequence, self.n_sequences, param_name)
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "sequence"
                return self
            except ValidationError:
                pass
        else:
            try:
                sequence = NormalizationParam(value=sequence).value
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "value"
                return self
            except ValidationError:
                pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a normalization, sequence of normalizations, or sequences of normalizations."
        )

    def set_bins_sequence(
        self, sequence: Union[BinsSequence, BinsType], param_name: str
    ):
        if self.multi_data:
            try:
                sequence = BinsSequenceParam(value=sequence).value
                validate_sequence_length(sequence, self.n_sequences, param_name)
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "sequence"
                return self
            except ValidationError:
                pass
        else:
            try:
                sequence = BinsParam(value=sequence).value
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "value"
                return self
            except ValidationError:
                pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a bin, sequence of bins, or sequences of bins."
        )

    def set_bool_sequence(self, sequence: Union[BoolSequence, bool], param_name: str):
        if self.multi_data:
            try:
                sequence = BoolSequenceParam(value=sequence).value
                validate_sequence_length(sequence, self.n_sequences, param_name)
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "sequence"
                return self
            except ValidationError:
                pass
        else:
            try:
                sequence = BoolParam(value=sequence).value
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "value"
                return self
            except ValidationError:
                pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a boolean or sequence of booleans."
        )

    def set_dict_sequence(self, sequence: Union[DictSequence, dict], param_name: str):
        if self.multi_data:
            try:
                sequence = DictSequenceParam(value=sequence).value
                validate_sequence_length(sequence, self.n_sequences, param_name)
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "sequence"
                return self
            except ValidationError:
                pass
        else:
            try:
                sequence = DictParam(value=sequence).value
                setattr(self, param_name, sequence)
                self.multi_params_structure[param_name] = "value"
                return self
            except ValidationError:
                pass
        raise ArgumentStructureError(
            f"Invalid {param_name}, must be a dictionary or sequence of dictionaries."
        )
