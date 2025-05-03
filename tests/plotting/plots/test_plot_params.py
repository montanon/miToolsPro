import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Transform
from pandas import Series
from pydantic import ValidationError

from mitoolspro.plotting.plots.plot_params import FigureParamsMixIn, ParamsMixIn


class TestParamsMixIn(TestCase):
    def setUp(self):
        self.ax = MagicMock(spec=Axes)
        self.figure = MagicMock(spec=Figure)
        self.ax.figure = self.figure
        self.params = ParamsMixIn(ax=self.ax)

    def test_initialization(self):
        params = ParamsMixIn()
        self.assertIsNone(params.ax)
        self.assertIsNone(params.figure)
        self.assertEqual(params.figsize, (10, 8))
        self.assertFalse(params.tight_layout)
        self.assertIsNone(params.style)

        params = ParamsMixIn(ax=self.ax)
        self.assertEqual(params.ax, self.ax)
        self.assertEqual(params.figure, self.figure)

    def test_set_alpha(self):
        self.params.set_alpha(0.5)
        self.assertEqual(self.params.alpha, 0.5)
        with self.assertRaises(ValidationError):
            self.params.set_alpha(1.5)

    def test_set_aspect(self):
        self.params.set_aspect("auto")
        self.assertEqual(self.params.aspect, "auto")
        self.params.set_aspect("equal")
        self.assertEqual(self.params.aspect, "equal")
        self.params.set_aspect(2.0)
        self.assertEqual(self.params.aspect, 2.0)
        with self.assertRaises(ValidationError):
            self.params.set_aspect("invalid")
        with self.assertRaises(ValidationError):
            self.params.set_aspect(-1.0)

    def test_set_title(self):
        self.params.set_title("Test Title", color="red")
        self.assertEqual(self.params.title["label"], "Test Title")
        self.assertEqual(self.params.title["color"], "red")

    def test_set_suptitle(self):
        self.params.set_suptitle("Test Suptitle", color="blue")
        self.assertEqual(self.params.suptitle["t"], "Test Suptitle")
        self.assertEqual(self.params.suptitle["color"], "blue")

    def test_set_transform(self):
        transform = MagicMock(spec=Transform)
        self.params.set_transform(transform)
        self.assertEqual(self.params.transform, transform)

    def test_set_axes_labels(self):
        self.params.set_xlabel("X Label", color="red")
        self.assertEqual(self.params.xlabel["xlabel"], "X Label")
        self.assertEqual(self.params.xlabel["color"], "red")

        self.params.set_ylabel("Y Label", color="blue")
        self.assertEqual(self.params.ylabel["ylabel"], "Y Label")
        self.assertEqual(self.params.ylabel["color"], "blue")

        self.params.set_axes_labels("New X", "New Y", color="green")
        self.assertEqual(self.params.xlabel["xlabel"], "New X")
        self.assertEqual(self.params.ylabel["ylabel"], "New Y")
        self.assertEqual(self.params.xlabel["color"], "green")
        self.assertEqual(self.params.ylabel["color"], "green")

    def test_set_scales(self):
        self.params.set_xscale("log")
        self.assertEqual(self.params.xscale, "log")
        self.params.set_yscale("logit")
        self.assertEqual(self.params.yscale, "logit")
        with self.assertRaises(ValidationError):
            self.params.set_xscale("invalid")

        self.params.set_scales("linear", "symlog")
        self.assertEqual(self.params.xscale, "linear")
        self.assertEqual(self.params.yscale, "symlog")

    def test_set_limits(self):
        self.params.set_xlim((0, 10))
        self.assertEqual(self.params.xlim, (0, 10))
        self.params.set_ylim((0, 20))
        self.assertEqual(self.params.ylim, (0, 20))
        with self.assertRaises(ValidationError):
            self.params.set_xlim((0, 1, 2))

        self.params.set_limits((1, 2), (3, 4))
        self.assertEqual(self.params.xlim, (1, 2))
        self.assertEqual(self.params.ylim, (3, 4))

    def test_set_ticks(self):
        self.params.set_xticks([1, 2, 3])
        self.assertEqual(self.params.xticks, [1, 2, 3])
        self.params.set_yticks([4, 5, 6])
        self.assertEqual(self.params.yticks, [4, 5, 6])

        self.params.set_ticks([7, 8, 9], [10, 11, 12])
        self.assertEqual(self.params.xticks, [7, 8, 9])
        self.assertEqual(self.params.yticks, [10, 11, 12])

    def test_set_ticklabels(self):
        self.params.set_xticklabels(["A", "B", "C"])
        self.assertEqual(self.params.xticklabels, ["A", "B", "C"])
        self.params.set_yticklabels(["D", "E", "F"])
        self.assertEqual(self.params.yticklabels, ["D", "E", "F"])

        self.params.set_ticklabels(["G", "H", "I"], ["J", "K", "L"])
        self.assertEqual(self.params.xticklabels, ["G", "H", "I"])
        self.assertEqual(self.params.yticklabels, ["J", "K", "L"])

    def test_set_tickparams(self):
        self.params.set_xtickparams({"color": "red"})
        self.assertEqual(self.params.xtickparams, {"color": "red"})
        self.params.set_ytickparams({"color": "blue"})
        self.assertEqual(self.params.ytickparams, {"color": "blue"})

        self.params.set_tickparams({"size": 10}, {"size": 12})
        self.assertEqual(self.params.xtickparams, {"size": 10})
        self.assertEqual(self.params.ytickparams, {"size": 12})

    def test_set_spines(self):
        self.params.set_spines(
            left={"visible": True, "color": "red"},
            right={"visible": False},
            bottom={"linewidth": 2},
            top={"linestyle": "--"},
        )
        self.assertIsNotNone(self.params.spines)
        spine_params = self.params.spines.model_dump()
        self.assertTrue(spine_params["left"]["visible"])
        self.assertEqual(spine_params["left"]["color"], "red")
        self.assertFalse(spine_params["right"]["visible"])
        self.assertEqual(spine_params["bottom"]["linewidth"], 2)
        self.assertEqual(spine_params["top"]["linestyle"], "--")

    def test_set_legend(self):
        self.params.set_legend(
            show=True,
            labels=["A", "B"],
            loc="upper right",
            ncol=2,
            title="Legend",
            frameon=False,
        )
        self.assertTrue(self.params.legend["show"])
        self.assertEqual(self.params.legend["kwargs"]["labels"], ["A", "B"])
        self.assertEqual(self.params.legend["kwargs"]["loc"], "upper right")
        self.assertEqual(self.params.legend["kwargs"]["ncol"], 2)
        self.assertEqual(self.params.legend["kwargs"]["title"], "Legend")
        self.assertFalse(self.params.legend["kwargs"]["frameon"])

    def test_set_texts(self):
        self.params.set_texts({"x": 1, "y": 2, "text": "Test"})
        self.assertEqual(len(self.params.texts), 1)
        self.assertEqual(self.params.texts[0]["text"], "Test")

        self.params.set_texts([{"x": 3, "y": 4, "text": "Test2"}])
        self.assertEqual(len(self.params.texts), 1)
        self.assertEqual(self.params.texts[0]["text"], "Test2")

    def test_set_grid(self):
        self.params.set_grid(visible=True, which="major", axis="both", color="gray")
        self.assertTrue(self.params.grid["visible"])
        self.assertEqual(self.params.grid["which"], "major")
        self.assertEqual(self.params.grid["axis"], "both")
        self.assertEqual(self.params.grid["color"], "gray")

        with self.assertRaises(ValidationError):
            self.params.set_grid(which="invalid")
        with self.assertRaises(ValidationError):
            self.params.set_grid(axis="invalid")

    def test_set_colors(self):
        self.params.set_facecolor("red")
        self.assertEqual(self.params.facecolor, "red")
        self.params.set_background("blue")
        self.assertEqual(self.params.background, "blue")
        self.params.set_figure_background("green")
        self.assertEqual(self.params.figure_background, "green")

    def test_set_figsize(self):
        self.params.set_figsize((12, 8))
        self.assertEqual(self.params.figsize, (12, 8))
        with self.assertRaises(ValidationError):
            self.params.set_figsize((1, 2, 3))

    def test_set_tight_layout(self):
        self.params.set_tight_layout(True)
        self.assertTrue(self.params.tight_layout)

    def test_set_style(self):
        with patch("matplotlib.pyplot.style.available", ["seaborn"]):
            self.params.set_style("seaborn")
            self.assertEqual(self.params.style, "seaborn")
            with self.assertRaises(ValidationError):
                self.params.set_style("invalid")

    def test_reset_params(self):
        self.params.set_alpha(0.5)
        self.params.set_title("Test")
        self.params.set_xlabel("X")
        self.params.reset_params()
        self.assertIsNone(self.params.alpha)
        self.assertEqual(self.params.title, "")
        self.assertEqual(self.params.xlabel, "")

    def test_clear(self):
        self.params.clear()
        self.assertIsNone(self.params.figure)
        self.assertIsNone(self.params.ax)

    def test_to_serializable(self):
        test_dict = {"a": 1, "b": np.array([1, 2, 3])}
        result = self.params._to_serializable(test_dict)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], [1, 2, 3])

        test_series = Series([1, 2, 3])
        result = self.params._to_serializable(test_series)
        self.assertEqual(result, [1, 2, 3])

        test_colormap = MagicMock(spec=Colormap)
        test_colormap.name = "viridis"
        result = self.params._to_serializable(test_colormap)
        self.assertEqual(result, "viridis")

        test_normalize = MagicMock(spec=Normalize)
        test_normalize.__class__.__name__ = "Normalize"
        result = self.params._to_serializable(test_normalize)
        self.assertEqual(result, "normalize")

        test_path = MagicMock(spec=Path)
        test_path.__str__.return_value = "/test/path"
        result = self.params._to_serializable(test_path)
        self.assertEqual(result, "/test/path")

        test_marker = MagicMock(spec=MarkerStyle)
        test_marker.get_marker.return_value = "o"
        test_marker.get_fillstyle.return_value = "full"
        test_marker.get_capstyle.return_value = "round"
        test_marker.get_joinstyle.return_value = "round"
        result = self.params._to_serializable(test_marker)
        self.assertEqual(
            result,
            {
                "marker": "o",
                "fillstyle": "full",
                "capstyle": "round",
                "joinstyle": "round",
            },
        )

    def test_apply_common_properties(self):
        self.params.set_title("Test Title")
        self.params.set_xlabel("X Label")
        self.params.set_ylabel("Y Label")
        self.params.set_xscale("log")
        self.params.set_yscale("log")
        self.params.set_xlim((0, 10))
        self.params.set_ylim((0, 20))
        self.params.set_xticks([1, 2, 3])
        self.params.set_yticks([4, 5, 6])
        self.params.set_xticklabels(["A", "B", "C"])
        self.params.set_yticklabels(["D", "E", "F"])
        self.params.set_xtickparams({"color": "red"})
        self.params.set_ytickparams({"color": "blue"})
        self.params.set_texts([{"x": 1, "y": 2, "text": "Test"}])
        self.params.set_legend(show=True, labels=["A", "B"])
        self.params.set_background("white")
        self.params.set_figure_background("gray")
        self.params.set_suptitle("Super Title")

        self.params._apply_common_properties()

        self.ax.set_title.assert_called_once_with(**self.params.title)
        self.ax.set_xlabel.assert_called_once_with(**self.params.xlabel)
        self.ax.set_ylabel.assert_called_once_with(**self.params.ylabel)
        self.ax.set_xscale.assert_called_once_with(self.params.xscale)
        self.ax.set_yscale.assert_called_once_with(self.params.yscale)
        self.ax.set_xlim.assert_called_once_with(self.params.xlim)
        self.ax.set_ylim.assert_called_once_with(self.params.ylim)
        self.ax.set_xticks.assert_called_once_with(self.params.xticks)
        self.ax.set_yticks.assert_called_once_with(self.params.yticks)
        self.ax.set_xticklabels.assert_called_once_with(self.params.xticklabels)
        self.ax.set_yticklabels.assert_called_once_with(self.params.yticklabels)
        self.ax.tick_params.assert_any_call(axis="x", **self.params.xtickparams)
        self.ax.tick_params.assert_any_call(axis="y", **self.params.ytickparams)
        self.ax.text.assert_called_once_with(**self.params.texts[0])
        self.ax.legend.assert_called_once_with(**self.params.legend["kwargs"])
        self.ax.set_facecolor.assert_called_once_with(self.params.background)
        self.figure.set_facecolor.assert_called_once_with(self.params.figure_background)
        self.figure.suptitle.assert_called_once_with(**self.params.suptitle)


class TestFigureParams(TestCase):
    def setUp(self):
        self.figure = MagicMock(spec=Figure)
        self.figure.get_size_inches.return_value = (12, 8)
        self.params = FigureParamsMixIn(figure=self.figure)

    def test_initialization(self):
        params = FigureParamsMixIn()
        self.assertIsNone(params.figure)
        self.assertEqual(params.figsize, (10, 8))
        self.assertFalse(params.tight_layout)
        self.assertIsNone(params.style)
        self.assertIsNone(params.figure_background)
        self.assertIsNone(params.suptitle)

        params = FigureParamsMixIn(figure=self.figure)
        self.assertEqual(params.figure, self.figure)
        self.assertEqual(params.figsize, (12, 8))

    def test_set_figsize(self):
        self.params.set_figsize((14, 10))
        self.assertEqual(self.params.figsize, (14, 10))
        with self.assertRaises(ValidationError):
            self.params.set_figsize((1, 2, 3))

    def test_set_style(self):
        with patch("matplotlib.pyplot.style.available", ["seaborn"]):
            self.params.set_style("seaborn")
            self.assertEqual(self.params.style, "seaborn")
            with self.assertRaises(ValidationError):
                self.params.set_style("invalid")

    def test_set_tight_layout(self):
        self.params.set_tight_layout(True)
        self.assertTrue(self.params.tight_layout)

    def test_set_figure_background(self):
        self.params.set_figure_background("red")
        self.assertEqual(self.params.figure_background, "red")
        with self.assertRaises(ValidationError):
            self.params.set_figure_background("invalid_color")

    def test_set_suptitle(self):
        self.params.set_suptitle("Test Title", color="red")
        self.assertEqual(self.params.suptitle["t"], "Test Title")
        self.assertEqual(self.params.suptitle["color"], "red")

    def test_reset_params(self):
        self.params.set_figsize((14, 10))
        self.params.set_style("ggplot")
        self.params.set_tight_layout(True)
        self.params.set_figure_background("red")
        self.params.set_suptitle("Test Title")

        self.params.reset_params()

        self.assertEqual(self.params.figsize, (12, 8))
        self.assertIsNone(self.params.style)
        self.assertFalse(self.params.tight_layout)
        self.assertIsNone(self.params.figure_background)
        self.assertIsNone(self.params.suptitle)

    def test_prepare_draw(self):
        with (
            patch("matplotlib.pyplot.style.use") as mock_style_use,
            patch("matplotlib.pyplot.figure") as mock_figure,
            patch("matplotlib.pyplot.rcParams.copy") as mock_copy,
        ):
            mock_copy.return_value = {"style": "default"}
            mock_figure.return_value = self.figure

            self.params.set_style("ggplot")
            self.params._prepare_draw()

            mock_style_use.assert_called_once_with("ggplot")
            mock_copy.assert_called_once()
            self.assertEqual(self.params._default_style, {"style": "default"})

            self.params.figure = None
            self.params._prepare_draw(clear=True)
            mock_figure.assert_called_once_with(figsize=(12, 8))

    def test_finalize_draw(self):
        with (
            patch("matplotlib.pyplot.tight_layout") as mock_tight_layout,
            patch("matplotlib.pyplot.rcParams.update") as mock_update,
        ):
            self.params.set_tight_layout(True)
            self.params.set_style("ggplot")
            self.params._default_style = {"style": "default"}

            result = self.params._finalize_draw(show=True)

            mock_tight_layout.assert_called_once()
            self.figure.show.assert_called_once()
            mock_update.assert_called_once_with({"style": "default"})
            self.assertEqual(result, self.figure)

    def test_clear(self):
        with patch("matplotlib.pyplot.close") as mock_close:
            self.params.clear()
            mock_close.assert_called_once_with(self.figure)
            self.assertIsNone(self.params.figure)

    def test_to_serializable(self):
        test_dict = {"a": 1, "b": np.array([1, 2, 3])}
        result = self.params._to_serializable(test_dict)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], [1, 2, 3])

        test_series = Series([1, 2, 3])
        result = self.params._to_serializable(test_series)
        self.assertEqual(result, [1, 2, 3])

        test_colormap = MagicMock(spec=Colormap)
        test_colormap.name = "viridis"
        result = self.params._to_serializable(test_colormap)
        self.assertEqual(result, "viridis")

        test_normalize = MagicMock(spec=Normalize)
        test_normalize.__class__.__name__ = "Normalize"
        result = self.params._to_serializable(test_normalize)
        self.assertEqual(result, "normalize")

        test_path = MagicMock(spec=Path)
        test_path.__str__.return_value = "/test/path"
        result = self.params._to_serializable(test_path)
        self.assertEqual(result, "/test/path")

        test_marker = MagicMock(spec=MarkerStyle)
        test_marker.get_marker.return_value = "o"
        test_marker.get_fillstyle.return_value = "full"
        test_marker.get_capstyle.return_value = "round"
        test_marker.get_joinstyle.return_value = "round"
        result = self.params._to_serializable(test_marker)
        self.assertEqual(
            result,
            {
                "marker": "o",
                "fillstyle": "full",
                "capstyle": "round",
                "joinstyle": "round",
            },
        )


if __name__ == "__main__":
    unittest.main()
