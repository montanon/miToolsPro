import unittest
from unittest import TestCase

from matplotlib.colors import to_rgba
from pydantic import ValidationError

from mitoolspro.exceptions import ArgumentStructureError
from mitoolspro.plotting.plots.setter import SetterMixIn


class TestSetColorSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_color(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_color_sequences("red", "color", structured=True)
        self.assertEqual(plotter.color, "red")

    def test_single_sequence_color_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_color_sequences(
            ["red", "blue", "green", "yellow"], "color", structured=True
        )
        self.assertEqual(
            plotter.color,
            ["red", "blue", "green", "yellow"],
        )

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(["red", "blue"], "color", structured=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(
                ["red", "blue", "green", "yellow", "purple"], "color", structured=True
            )

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red", "blue"]], "color", structured=True)

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_color(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_color_sequences("red", "color", structured=True)
        self.assertEqual(plotter.color, "red")

    def test_multi_sequence_color_sequence_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_color_sequences(["red", "blue"], "color", structured=True)
        self.assertEqual(plotter.color, ["red", "blue"])

    def test_multi_sequence_nested_colors_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_color_sequences(
            [["red", "green"], ["blue", "yellow"]], "color", structured=True
        )
        self.assertEqual(
            plotter.color,
            [["red", "green"], ["blue", "yellow"]],
        )

    def test_multi_sequence_mixed_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(
                [["red"], ["blue", "yellow"]], "color", structured=True
            )

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(["red"], "color", structured=True)

    def test_multi_sequence_color_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4], [2, 3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(
                ["red", "blue", "green", "yellow"], "color", structured=True
            )

    def test_multi_sequence_color_nested_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_color_sequences(
            [["red", "green"], ["blue", "yellow"]], "color", structured=True
        )
        self.assertEqual(
            plotter.color,
            [["red", "green"], ["blue", "yellow"]],
        )

    def test_multi_sequence_mismatched_sequence_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences(["red"], "color")

    def test_multi_sequence_mixed_nested(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red"], ["blue", "yellow"]], "color")
        plotter.set_color_sequences([["red", "blue"], ["blue", "yellow"]], "color")
        self.assertEqual(plotter.color, [["red", "blue"], ["blue", "yellow"]])
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red", "blue"], ["yellow"]], "color")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red", "blue"], ["yellow"]], "color")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red", "blue"], "yellow"], "color")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences([["red", "blue", "green"], ["yellow"]], "color")

    def test_structured_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_color_sequences(
            [["red", "green"], ["blue", "yellow", "cyan"]], "color", structured=True
        )
        self.assertEqual(
            plotter.color,
            [["red", "green"], ["blue", "yellow", "cyan"]],
        )

    # --------------- Invalid Cases ----------------
    def test_invalid_color_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_color_sequences("invalid_color", "color")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_color_sequences("red", "color")

    def test_mixed_color_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_color_sequences([1, "red", (0, 0, 0)], "color")
        self.assertEqual(plotter.color, [1, "red", (0, 0, 0)])


class TestSetNumericSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_value(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_numeric_sequences(5, "alpha")
        self.assertEqual(plotter.alpha, 5)

    def test_single_sequence_numeric_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_numeric_sequences([0.1, 0.2, 0.3, 0.4], "alpha")
        self.assertEqual(plotter.alpha, [0.1, 0.2, 0.3, 0.4])

    def test_single_sequence_out_of_range(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences(
                [0.1, 1.5, 0.3, 0.4], "alpha", min_value=0, max_value=1
            )

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences([0.1, 0.2], "alpha")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences([0.1, 0.2, 0.3, 0.4, 0.5], "alpha")

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences([[0.1, 0.2], [0.3, 0.4]], "alpha")

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_value(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_sequences(0.5, "alpha")
        self.assertEqual(plotter.alpha, 0.5)

    def test_multi_sequence_numeric_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_sequences([0.3, 0.6], "alpha")
        self.assertEqual(plotter.alpha, [0.3, 0.6])

    def test_multi_sequence_nested_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_sequences([[0.1, 0.2], [0.3, 0.4]], "alpha")
        self.assertEqual(plotter.alpha, [[0.1, 0.2], [0.3, 0.4]])

    def test_multi_sequence_mixed_valid(self):
        plotter = self.MockPlotter([[1, 2], [3]], multi_data=True)
        plotter.set_numeric_sequences([[0.1, 0.2], [0.3]], "alpha", structured=False)
        self.assertEqual(plotter.alpha, [[0.1, 0.2], [0.3]])

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences([0.1, 0.2, 0.3], "alpha")

    def test_multi_sequence_out_of_range(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences(
                [[0.1, 0.2], [1.5, 0.4]], "alpha", min_value=0, max_value=1
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_numeric_sequences(
            [[0.1, 0.2], [0.3, 0.4, 0.5]], "alpha", structured=True
        )
        self.assertEqual(plotter.alpha, [[0.1, 0.2], [0.3, 0.4, 0.5]])

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences(
                [[0.1, 0.2], [0.3, 0.4]], "alpha", structured=True
            )

    # --------------- Invalid Cases ----------------
    def test_invalid_numeric_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_sequences("invalid", "alpha")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_numeric_sequences(0.5, "alpha")

    def test_mixed_numeric_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_numeric_sequences([1, 0.5, 3.2], "alpha")
        self.assertEqual(plotter.alpha, [1, 0.5, 3.2])


class TestSetLiteralSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    VALID_OPTIONS = ["red", "blue", "green", "yellow"]

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_literal(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_literal_sequences("red", self.VALID_OPTIONS, "label")
        self.assertEqual(plotter.label, "red")

    def test_single_sequence_literal_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_literal_sequences(
            ["red", "blue", "green", "yellow"], self.VALID_OPTIONS, "label"
        )
        self.assertEqual(plotter.label, ["red", "blue", "green", "yellow"])

    def test_single_sequence_invalid_literal(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences("purple", self.VALID_OPTIONS, "label")

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(["red", "blue"], self.VALID_OPTIONS, "label")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(
                ["red", "blue", "green", "yellow", "purple"],
                self.VALID_OPTIONS,
                "label",
            )

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(
                [["red", "blue"]], self.VALID_OPTIONS, "label"
            )

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_literal(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_literal_sequences("red", self.VALID_OPTIONS, "label")
        self.assertEqual(plotter.label, "red")

    def test_multi_sequence_literal_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_literal_sequences(["red", "blue"], self.VALID_OPTIONS, "label")
        self.assertEqual(plotter.label, ["red", "blue"])

    def test_multi_sequence_nested_literals_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_literal_sequences(
            [["red", "green"], ["blue", "yellow"]],
            self.VALID_OPTIONS,
            "label",
        )
        self.assertEqual(plotter.label, [["red", "green"], ["blue", "yellow"]])

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(["red"], self.VALID_OPTIONS, "label")

    def test_multi_sequence_invalid_literal(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(
                ["red", "purple"], self.VALID_OPTIONS, "label"
            )

    def test_multi_sequence_nested_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(
                [["red", "blue", "green"], ["yellow"]],
                self.VALID_OPTIONS,
                "label",
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_literal_sequences(
            [["red", "green"], ["blue", "yellow", "green"]],
            self.VALID_OPTIONS,
            "label",
            structured=True,
        )
        self.assertEqual(
            plotter.label,
            [["red", "green"], ["blue", "yellow", "green"]],
        )

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences(
                [["red", "green"], ["blue"]],
                self.VALID_OPTIONS,
                "label",
                structured=True,
            )

    # --------------- Invalid Cases ----------------
    def test_invalid_literal_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_literal_sequences("invalid", self.VALID_OPTIONS, "label")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_literal_sequences("red", self.VALID_OPTIONS, "label")

    def test_mixed_literal_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_literal_sequences(
            ["red", "blue", "green"], self.VALID_OPTIONS, "label"
        )
        self.assertEqual(plotter.label, ["red", "blue", "green"])


class TestSetMarkerSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    VALID_MARKERS = ["o", "s", "x", "d", "^", "v", "<", ">"]

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_marker(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_marker_sequences("o", "marker")
        self.assertEqual(plotter.marker, "o")

    def test_single_sequence_marker_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_marker_sequences(["o", "s", "x", "d"], "marker")
        self.assertEqual(plotter.marker, ["o", "s", "x", "d"])

    def test_single_sequence_invalid_marker(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences("invalid_marker", "marker")

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences(["o", "s"], "marker")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences(["o", "s", "x", "d", "^"], "marker")

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences([["o", "s"]], "marker")

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_marker(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_marker_sequences("o", "marker")
        self.assertEqual(plotter.marker, "o")

    def test_multi_sequence_marker_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_marker_sequences(["o", "s"], "marker")
        self.assertEqual(plotter.marker, ["o", "s"])

    def test_multi_sequence_nested_markers_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_marker_sequences(
            [["o", "s"], ["x", "d"]],
            "marker",
        )
        self.assertEqual(plotter.marker, [["o", "s"], ["x", "d"]])

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences(["o"], "marker")

    def test_multi_sequence_invalid_marker(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences(["o", "invalid_marker"], "marker")

    def test_multi_sequence_nested_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences(
                [["o", "s", "x"], ["d"]],
                "marker",
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_marker_sequences(
            [["o", "s"], ["x", "d", "^"]], "marker", structured=True
        )
        self.assertEqual(
            plotter.marker,
            [["o", "s"], ["x", "d", "^"]],
        )

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences([["o", "s"], ["x"]], "marker", structured=True)

    # --------------- Invalid Cases ----------------
    def test_invalid_marker_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_marker_sequences("invalid", "marker")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_marker_sequences("o", "marker")

    def test_mixed_marker_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_marker_sequences(["o", "s", "x"], "marker")
        self.assertEqual(plotter.marker, ["o", "s", "x"])


class TestSetEdgeColorSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_edgecolor(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_edgecolor_sequences("red", "edgecolor")
        self.assertEqual(plotter.edgecolor, "red")

    def test_single_sequence_edgecolor_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_edgecolor_sequences(["red", "blue", "green", "yellow"], "edgecolor")
        self.assertEqual(
            plotter.edgecolor,
            ["red", "blue", "green", "yellow"],
        )

    def test_single_sequence_invalid_edgecolor(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences("invalid_color", "edgecolor")

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(["red", "blue"], "edgecolor")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(
                ["red", "blue", "green", "yellow", "purple"], "edgecolor"
            )

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences([["red", "blue"]], "edgecolor")

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_edgecolor(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_edgecolor_sequences("red", "edgecolor")
        self.assertEqual(plotter.edgecolor, "red")

    def test_multi_sequence_edgecolor_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_edgecolor_sequences(["red", "blue"], "edgecolor")
        self.assertEqual(plotter.edgecolor, ["red", "blue"])

    def test_multi_sequence_nested_edgecolors_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_edgecolor_sequences(
            [["red", "green"], ["blue", "yellow"]],
            "edgecolor",
        )
        self.assertEqual(
            plotter.edgecolor,
            [["red", "green"], ["blue", "yellow"]],
        )

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(["red"], "edgecolor")

    def test_multi_sequence_invalid_edgecolor(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(["red", "invalid_color"], "edgecolor")

    def test_multi_sequence_nested_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(
                [["red", "green", "blue"], ["yellow"]],
                "edgecolor",
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_edgecolor_sequences(
            [["red", "green"], ["blue", "yellow", "cyan"]], "edgecolor", structured=True
        )
        self.assertEqual(
            plotter.edgecolor,
            [["red", "green"], ["blue", "yellow", "cyan"]],
        )

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences(
                [["red", "green"], ["blue"]], "edgecolor", structured=True
            )

    # --------------- Invalid Cases ----------------
    def test_invalid_edgecolor_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_edgecolor_sequences("invalid", "edgecolor")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_edgecolor_sequences("red", "edgecolor")

    def test_mixed_edgecolor_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_edgecolor_sequences(["red", (0, 0, 0), "#0000FF"], "edgecolor")
        self.assertEqual(
            plotter.edgecolor,
            ["red", (0, 0, 0), "#0000FF"],
        )


class TestSetStrSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_string(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_str_sequences("Label A", "label")
        self.assertEqual(plotter.label, "Label A")

    def test_single_sequence_string_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_str_sequences(["Label A", "Label B", "Label C", "Label D"], "label")
        self.assertEqual(plotter.label, ["Label A", "Label B", "Label C", "Label D"])

    def test_single_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(["Label A", "Label B"], "label")
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(
                ["Label A", "Label B", "Label C", "Label D", "Label E"], "label"
            )

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences([["Label A", "Label B"]], "label")

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_string(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_str_sequences("Common Label", "label")
        self.assertEqual(plotter.label, "Common Label")

    def test_multi_sequence_string_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_str_sequences(["Label A", "Label B"], "label")
        self.assertEqual(plotter.label, ["Label A", "Label B"])

    def test_multi_sequence_nested_strings_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_str_sequences(
            [["Label A1", "Label A2"], ["Label B1", "Label B2"]],
            "label",
        )
        self.assertEqual(
            plotter.label,
            [["Label A1", "Label A2"], ["Label B1", "Label B2"]],
        )

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(["Label A"], "label")

    def test_multi_sequence_invalid_string(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(["Label A", 123], "label")

    def test_multi_sequence_nested_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(
                [["Label A", "Label B", "Label C"], ["Label D"]],
                "label",
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_str_sequences(
            [["Label A", "Label B"], ["Label C", "Label D", "Label E"]],
            "label",
            structured=True,
        )
        self.assertEqual(
            plotter.label,
            [["Label A", "Label B"], ["Label C", "Label D", "Label E"]],
        )

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(
                [["Label A", "Label B"], ["Label C"]], "label", structured=True
            )

    # --------------- Invalid Cases ----------------
    def test_invalid_string_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_str_sequences(123, "label")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_str_sequences("Label", "label")

    def test_mixed_string_types(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        plotter.set_str_sequences(["Label A", "Label B", "Label C"], "label")
        self.assertEqual(plotter.label, ["Label A", "Label B", "Label C"])


class TestSetNumericTupleSequences(TestCase):
    class MockPlotter(SetterMixIn):
        def __init__(self, x_data, multi_data):
            self._x_data = x_data
            self._multi_data = multi_data

        @property
        def x_data(self):
            return self._x_data

        @property
        def multi_data(self):
            return self._multi_data

    # --------------- Single Sequence (multi_data=False) Tests ----------------
    def test_single_sequence_single_numeric_tuple(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_numeric_tuple_sequences((1, 2), 2, "coordinates")
        self.assertEqual(plotter.coordinates, (1, 2))

    def test_single_sequence_numeric_tuple_sequence(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        plotter.set_numeric_tuple_sequences(
            [(1, 2), (3, 4), (5, 6), (7, 8)], 2, "coordinates"
        )
        self.assertEqual(plotter.coordinates, [(1, 2), (3, 4), (5, 6), (7, 8)])

    def test_single_sequence_invalid_tuple_length(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences((1, 2, 3), 2, "coordinates")

    def test_single_sequence_invalid_nested(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences([[(1, 2)]], 2, "coordinates")

    def test_single_sequence_mismatched_tuple_sizes(self):
        plotter = self.MockPlotter([[1, 2, 3, 4]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences([(1, 2), (3, 4, 5)], 2, "coordinates")

    # --------------- Multi-Sequence (multi_data=True) Tests ----------------
    def test_multi_sequence_single_numeric_tuple(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_tuple_sequences((1, 2), 2, "coordinates")
        self.assertEqual(plotter.coordinates, (1, 2))

    def test_multi_sequence_numeric_tuple_sequence(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_tuple_sequences([(1, 2), (3, 4)], 2, "coordinates")
        self.assertEqual(plotter.coordinates, [(1, 2), (3, 4)])

    def test_multi_sequence_nested_numeric_tuples_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        plotter.set_numeric_tuple_sequences(
            [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
            2,
            "coordinates",
        )
        self.assertEqual(
            plotter.coordinates,
            [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
        )

    def test_multi_sequence_invalid_length(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences([(1, 2)], 2, "coordinates")

    def test_multi_sequence_invalid_tuple(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences([(1, 2), (3, 4, 5)], 2, "coordinates")

    def test_multi_sequence_nested_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences(
                [[(1, 2), (3, 4, 5)], [(5, 6)]],
                2,
                "coordinates",
            )

    # --------------- Structured = True (Strict Structure) Tests ----------------
    def test_structured_true_valid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        plotter.set_numeric_tuple_sequences(
            [[(1, 2), (3, 4)], [(5, 6), (7, 8), (9, 10)]],
            2,
            "coordinates",
            structured=True,
        )
        self.assertEqual(
            plotter.coordinates,
            [[(1, 2), (3, 4)], [(5, 6), (7, 8), (9, 10)]],
        )

    def test_structured_true_invalid(self):
        plotter = self.MockPlotter([[1, 2], [3, 4, 5]], multi_data=True)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences(
                [[(1, 2), (3, 4)], [(5, 6)]], 2, "coordinates", structured=True
            )

    # --------------- Invalid Cases ----------------
    def test_invalid_numeric_tuple_value(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences("invalid", 2, "coordinates")

    def test_empty_data(self):
        plotter = self.MockPlotter([], multi_data=False)
        with self.assertRaises(IndexError):
            plotter.set_numeric_tuple_sequences((1, 2), 2, "coordinates")

    def test_mixed_numeric_tuple_sizes(self):
        plotter = self.MockPlotter([[1, 2, 3]], multi_data=False)
        with self.assertRaises(ArgumentStructureError):
            plotter.set_numeric_tuple_sequences([(1, 2), (3, 4, 5)], 2, "coordinates")


if __name__ == "__main__":
    unittest.main()
