import unittest
from unittest import TestCase

import numpy as np
from matplotlib.colors import Normalize
from matplotlib.markers import MarkerStyle
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentStructureError, ArgumentValueError
from mitoolspro.plotting.plots.setter import SetterMixIn
from mitoolspro.plotting.plots.validation.models import (
    BinsParam,
    BoolParam,
    ColormapParam,
    ColorParam,
    DictParam,
    EdgeColorParam,
    LiteralParam,
    MarkerParam,
    NormalizationParam,
    NumericParam,
    NumericTupleParam,
    RangeParam,
    RangeSequenceParam,
    RangeSequencesParam,
    StrParam,
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
    StrSequence,
    StrSequences,
)


class DummySetter(SetterMixIn):
    def __init__(self, data_size=4, n_sequences=1):
        self._data_size = data_size
        self._n_sequences = n_sequences
        self._multi_data = n_sequences > 1
        self._multi_params_structure = {}

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


class TestSetter(TestCase):
    def setUp(self):
        self.setter = DummySetter()
        self.multi_setter = DummySetter(data_size=4, n_sequences=2)

    def test_set_color_sequences_single_color(self):
        self.setter.set_color_sequences("red", "color")
        self.assertEqual(self.setter.color, "red")

    def test_set_color_sequences_sequence(self):
        colors = ["red", "blue", "green", "yellow"]
        self.setter.set_color_sequences(colors, "color")
        self.assertEqual(self.setter.color, colors)

    def test_set_color_sequences_multi_sequences(self):
        colors = [
            ["red", "blue", "green", "yellow"],
            ["yellow", "green", "blue", "red"],
        ]
        self.multi_setter.set_color_sequences(colors, "color")
        self.assertEqual(self.multi_setter.color, colors)

    def test_set_color_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_color_sequences(["invalid_color"], "color")
        with self.assertRaises(ArgumentStructureError):
            self.multi_setter.set_color_sequences([["red", "blue"], "blue"], "color")

    def test_set_color_sequence_single_color(self):
        self.setter.set_color_sequences("red", "color")
        self.assertEqual(self.setter.color, "red")

    def test_set_color_sequence_sequence(self):
        colors = ["red", "blue"]
        self.multi_setter.set_color_sequences(colors, "color")
        self.assertEqual(self.multi_setter.color, colors)

    def test_set_color_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_color_sequences(["invalid_color"], "color")

    def test_set_numeric_sequences_single_value(self):
        self.setter.set_numeric_sequences(0.5, "alpha")
        self.assertEqual(self.setter.alpha, 0.5)

    def test_set_numeric_sequences_sequence(self):
        alphas = [0.5, 0.6, 0.7, 0.8]
        self.setter.set_numeric_sequences(alphas, "alpha")
        np.testing.assert_array_equal(self.setter.alpha, alphas)

    def test_set_numeric_sequences_multi_sequences(self):
        alphas = [[0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5]]
        self.multi_setter.set_numeric_sequences(alphas, "alpha")
        np.testing.assert_array_equal(self.multi_setter.alpha, alphas)

    def test_set_numeric_sequences_with_range(self):
        self.setter.set_numeric_sequences(0.5, "alpha", min_value=0, max_value=1)
        self.assertEqual(self.setter.alpha, 0.5)
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_numeric_sequences(1.5, "alpha", min_value=0, max_value=1)

    def test_set_numeric_sequence_single_value(self):
        self.setter.set_numeric_sequences(0.5, "alpha")
        self.assertEqual(self.setter.alpha, 0.5)

    def test_set_numeric_sequence_sequence(self):
        alphas = [0.5, 0.6]
        self.multi_setter.set_numeric_sequences(alphas, "alpha")
        np.testing.assert_array_equal(self.multi_setter.alpha, alphas)

    def test_set_numeric_sequence_with_range(self):
        self.setter.set_numeric_sequences(0.5, "alpha", min_value=0, max_value=1)
        self.assertEqual(self.setter.alpha, 0.5)
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_numeric_sequences(1.5, "alpha", min_value=0, max_value=1)

    def test_set_literal_sequences_single_value(self):
        self.setter.set_literal_sequences("linear", ["linear", "log"], "scale")
        self.assertEqual(self.setter.scale, "linear")

    def test_set_literal_sequences_sequence(self):
        scales = ["linear", "log", "linear", "log"]
        self.setter.set_literal_sequences(scales, ["linear", "log"], "scale")
        self.assertEqual(self.setter.scale, scales)

    def test_set_literal_sequences_multi_sequences(self):
        scales = [
            ["linear", "log", "linear", "log"],
            ["log", "linear", "log", "linear"],
        ]
        self.multi_setter.set_literal_sequences(scales, ["linear", "log"], "scale")
        self.assertEqual(self.multi_setter.scale, scales)

    def test_set_literal_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_literal_sequences("invalid", ["linear", "log"], "scale")

    def test_set_literal_sequence_single_value(self):
        self.setter.set_literal_sequences("linear", ["linear", "log"], "scale")
        self.assertEqual(self.setter.scale, "linear")

    def test_set_literal_sequence_sequence(self):
        scales = ["linear", "log"]
        self.multi_setter.set_literal_sequences(scales, ["linear", "log"], "scale")
        self.assertEqual(self.multi_setter.scale, scales)

    def test_set_literal_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_literal_sequences("invalid", ["linear", "log"], "scale")

    def test_set_marker_sequences_single_value(self):
        self.setter.set_marker_sequences("o", "marker")
        self.assertEqual(self.setter.marker, "o")

    def test_set_marker_sequences_sequence(self):
        markers = ["o", "s", "D", "^"]
        self.setter.set_marker_sequences(markers, "marker")
        self.assertEqual(self.setter.marker, markers)

    def test_set_marker_sequences_multi_sequences(self):
        markers = [["o", "s", "D", "^"], ["^", "D", "s", "o"]]
        self.multi_setter.set_marker_sequences(markers, "marker")
        self.assertEqual(self.multi_setter.marker, markers)

    def test_set_marker_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_marker_sequences(["invalid"], "marker")

    def test_set_marker_sequence_single_value(self):
        self.setter.set_marker_sequences("o", "marker")
        self.assertEqual(self.setter.marker, "o")

    def test_set_marker_sequence_sequence(self):
        markers = ["o", "s"]
        self.multi_setter.set_marker_sequences(markers, "marker")
        self.assertEqual(self.multi_setter.marker, [[m] for m in markers])

    def test_set_marker_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_marker_sequences(["invalid"], "marker")

    def test_set_edgecolor_sequences_single_value(self):
        self.setter.set_edgecolor_sequences("red", "edgecolor")
        self.assertEqual(self.setter.edgecolor, "red")

    def test_set_edgecolor_sequences_sequence(self):
        edgecolors = ["red", "blue", "green", "yellow"]
        self.setter.set_edgecolor_sequences(edgecolors, "edgecolor")
        self.assertEqual(self.setter.edgecolor, edgecolors)

    def test_set_edgecolor_sequences_multi_sequences(self):
        edgecolors = [
            ["red", "blue", "green", "yellow"],
            ["yellow", "green", "blue", "red"],
        ]
        self.multi_setter.set_edgecolor_sequences(edgecolors, "edgecolor")
        self.assertEqual(self.multi_setter.edgecolor, edgecolors)

    def test_set_edgecolor_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_edgecolor_sequences(["invalid"], "edgecolor")

    def test_set_edgecolor_sequence_single_value(self):
        self.setter.set_edgecolor_sequences("red", "edgecolor")
        self.assertEqual(self.setter.edgecolor, "red")

    def test_set_edgecolor_sequence_sequence(self):
        edgecolors = ["red", "blue"]
        self.multi_setter.set_edgecolor_sequences(edgecolors, "edgecolor")
        self.assertEqual(self.multi_setter.edgecolor, edgecolors)

    def test_set_edgecolor_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_edgecolor_sequences(["invalid"], "edgecolor")

    def test_set_colormap_sequence_single_value(self):
        self.setter.set_colormap_sequence("viridis", "cmap")
        self.assertEqual(self.setter.cmap, "viridis")

    def test_set_colormap_sequence_sequence(self):
        cmaps = ["viridis", "plasma"]
        self.multi_setter.set_colormap_sequence(cmaps, "cmap")
        self.assertEqual(self.multi_setter.cmap, cmaps)

    def test_set_colormap_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_colormap_sequence(["invalid"], "cmap")

    def test_set_norm_sequence_single_value(self):
        self.setter.set_norm_sequence("linear", "norm")
        self.assertEqual(self.setter.norm, "linear")

    def test_set_norm_sequence_sequence(self):
        norms = ["linear", "log"]
        self.multi_setter.set_norm_sequence(norms, "norm")
        self.assertEqual(self.multi_setter.norm, norms)

    def test_set_norm_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_norm_sequence(["invalid"], "norm")

    def test_set_str_sequences_single_value(self):
        self.setter.set_str_sequences("label1", "label")
        self.assertEqual(self.setter.label, "label1")

    def test_set_str_sequences_sequence(self):
        labels = ["label1", "label2", "label3", "label4"]
        self.setter.set_str_sequences(labels, "label")
        self.assertEqual(self.setter.label, labels)

    def test_set_str_sequences_multi_sequences(self):
        labels = [
            ["label1", "label2", "label3", "label4"],
            ["label4", "label3", "label2", "label1"],
        ]
        self.multi_setter.set_str_sequences(labels, "label")
        self.assertEqual(self.multi_setter.label, labels)

    def test_set_str_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_str_sequences(123, "label")

    def test_set_str_sequence_single_value(self):
        self.setter.set_str_sequences("label1", "label")
        self.assertEqual(self.setter.label, "label1")

    def test_set_str_sequence_sequence(self):
        labels = ["label1", "label2"]
        self.multi_setter.set_str_sequences(labels, "label")
        self.assertEqual(self.multi_setter.label, labels)

    def test_set_str_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_str_sequences(123, "label")

    def test_set_numeric_tuple_sequences_single_value(self):
        self.setter.set_numeric_tuple_sequences((1, 2), (2,), "size")
        self.assertEqual(self.setter.size, (1, 2))

    def test_set_numeric_tuple_sequences_sequence(self):
        sizes = [(1, 2), (2, 3), (3, 4), (4, 5)]
        self.setter.set_numeric_tuple_sequences(sizes, (2,), "size")
        self.assertEqual(self.setter.size, sizes)

    def test_set_numeric_tuple_sequences_multi_sequences(self):
        sizes = [[(1, 2), (2, 3), (3, 4), (4, 5)], [(5, 6), (6, 7), (7, 8), (8, 9)]]
        self.multi_setter.set_numeric_tuple_sequences(sizes, (2,), "size")
        self.assertEqual(self.multi_setter.size, sizes)

    def test_set_numeric_tuple_sequences_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_numeric_tuple_sequences([(1,)], (2,), "size")

    def test_set_numeric_tuple_sequence_single_value(self):
        self.setter.set_numeric_tuple_sequences((1, 2), (2,), "size")
        self.assertEqual(self.setter.size, (1, 2))

    def test_set_numeric_tuple_sequence_sequence(self):
        sizes = [(1, 2), (2, 3)]
        self.multi_setter.set_numeric_tuple_sequences(sizes, (2,), "size")
        self.assertEqual(self.multi_setter.size, sizes)

    def test_set_numeric_tuple_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_numeric_tuple_sequences([(1,)], (2,), "size")

    def test_set_bins_sequence_single_value(self):
        self.setter.set_bins_sequence(10, "bins")
        self.assertEqual(self.setter.bins, 10)

    def test_set_bins_sequence_sequence(self):
        bins = [10, 20]
        self.multi_setter.set_bins_sequence(bins, "bins")
        self.assertEqual(self.multi_setter.bins, bins)

    def test_set_bins_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_bins_sequence(["invalid"], "bins")

    def test_set_bool_sequence_single_value(self):
        self.setter.set_bool_sequence(True, "visible")
        self.assertEqual(self.setter.visible, True)

    def test_set_bool_sequence_sequence(self):
        visible = [True, False]
        self.multi_setter.set_bool_sequence(visible, "visible")
        self.assertEqual(self.multi_setter.visible, visible)

    def test_set_bool_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_bool_sequence(["invalid"], "visible")

    def test_set_dict_sequence_single_value(self):
        self.setter.set_dict_sequence({"key": "value"}, "kwargs")
        self.assertEqual(self.setter.kwargs, {"key": "value"})

    def test_set_dict_sequence_sequence(self):
        kwargs = [{"key1": "value1"}, {"key2": "value2"}]
        self.multi_setter.set_dict_sequence(kwargs, "kwargs")
        self.assertEqual(self.multi_setter.kwargs, kwargs)

    def test_set_dict_sequence_invalid(self):
        with self.assertRaises(ArgumentStructureError):
            self.setter.set_dict_sequence(["invalid"], "kwargs")


if __name__ == "__main__":
    unittest.main()
