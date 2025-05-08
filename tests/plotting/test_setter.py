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


if __name__ == "__main__":
    unittest.main()
