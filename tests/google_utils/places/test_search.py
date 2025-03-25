import tempfile
import unittest
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.search import (
    _generate_file_path,
    _generate_plot_paths,
    _generate_results_plots,
    _generate_sampling_plots,
    _plot_polygon_with_circles,
    places_search_step,
    search_places_in_polygon,
)


class TestSearchFunctions(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_folder = Path(self.temp_dir.name) / "test_project"
        self.plots_folder = Path(self.temp_dir.name) / "test_plots"
        self.project_folder.mkdir(exist_ok=True)
        self.plots_folder.mkdir(exist_ok=True)
        self.tag = "test_search"
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.radius_in_meters = 1000.0
        self.step_in_degrees = 0.1
        self.client = GooglePlacesClient()
        self.included_types = ["restaurant", "cafe"]
        self.threshold = 20
        self.has_places = True
        self.show = False
        self.recalculate = False

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_places_search_step_valid_inputs(self):
        found_places, circles, saturated_area, saturated_circles = places_search_step(
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            tag=self.tag,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            client=self.client,
            included_types=self.included_types,
            threshold=self.threshold,
            has_places=self.has_places,
            show=self.show,
            recalculate=self.recalculate,
        )
        self.assertIsInstance(found_places, pd.DataFrame)
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertTrue(
            isinstance(saturated_area, (Polygon, MultiPolygon))
            or saturated_area.is_empty
        )
        self.assertIsInstance(saturated_circles, GeoDataFrame)

    def test_places_search_step_invalid_project_folder(self):
        with self.assertRaises(ArgumentValueError):
            places_search_step(
                project_folder=Path("nonexistent"),
                plots_folder=self.plots_folder,
                tag=self.tag,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                client=self.client,
            )

    def test_places_search_step_invalid_plots_folder(self):
        with self.assertRaises(ArgumentValueError):
            places_search_step(
                project_folder=self.project_folder,
                plots_folder=Path("nonexistent"),
                tag=self.tag,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                client=self.client,
            )

    def test_places_search_step_invalid_polygon(self):
        with self.assertRaises(ArgumentTypeError):
            places_search_step(
                project_folder=self.project_folder,
                plots_folder=self.plots_folder,
                tag=self.tag,
                polygon="invalid_polygon",
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                client=self.client,
            )

    def test_search_places_in_polygon_valid_inputs(self):
        circles, found_places = search_places_in_polygon(
            root_folder=self.project_folder,
            plot_folder=self.plots_folder,
            tag=self.tag,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            condition_rule="center",
            client=self.client,
            included_types=self.included_types,
            recalculate=self.recalculate,
            has_places=self.has_places,
            show=self.show,
        )
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertIsInstance(found_places, pd.DataFrame)

    def test_search_places_in_polygon_invalid_root_folder(self):
        with self.assertRaises(ArgumentValueError):
            search_places_in_polygon(
                root_folder=Path("nonexistent"),
                plot_folder=self.plots_folder,
                tag=self.tag,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule="center",
                client=self.client,
            )

    def test_search_places_in_polygon_invalid_plot_folder(self):
        with self.assertRaises(ArgumentValueError):
            search_places_in_polygon(
                root_folder=self.project_folder,
                plot_folder=Path("nonexistent"),
                tag=self.tag,
                polygon=self.polygon,
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule="center",
                client=self.client,
            )

    def test_search_places_in_polygon_invalid_polygon(self):
        with self.assertRaises(ArgumentTypeError):
            search_places_in_polygon(
                root_folder=self.project_folder,
                plot_folder=self.plots_folder,
                tag=self.tag,
                polygon="invalid_polygon",
                radius_in_meters=self.radius_in_meters,
                step_in_degrees=self.step_in_degrees,
                condition_rule="center",
                client=self.client,
            )

    def test_generate_file_path(self):
        file_path = _generate_file_path(
            self.project_folder,
            self.tag,
            self.radius_in_meters,
            self.step_in_degrees,
            "test.parquet",
        )
        self.assertIsInstance(file_path, Path)
        self.assertEqual(
            file_path.name,
            f"{self.tag}_{self.radius_in_meters}_radius_{self.step_in_degrees}_step_test.parquet",
        )

    def test_generate_plot_paths(self):
        plot_paths = _generate_plot_paths(self.plots_folder, self.tag)
        self.assertIsInstance(plot_paths, dict)
        self.assertEqual(len(plot_paths), 4)
        self.assertIn("circles", plot_paths)
        self.assertIn("circles_zoom", plot_paths)
        self.assertIn("places", plot_paths)
        self.assertIn("places_zoom", plot_paths)
        for path in plot_paths.values():
            self.assertIsInstance(path, Path)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_polygon_with_circles(self, mock_savefig, mock_show):
        circles = [self.polygon]
        output_path = self.plots_folder / "test_plot.png"
        _plot_polygon_with_circles(
            polygon=self.polygon,
            circles=circles,
            output_path=output_path,
            show=True,
        )
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_sampling_plots(self, mock_savefig, mock_show):
        circles = GeoDataFrame({"geometry": [self.polygon]}, crs="EPSG:4326")
        plot_paths = _generate_plot_paths(self.plots_folder, self.tag)
        _generate_sampling_plots(
            polygon=self.polygon,
            circles=circles.geometry,
            plot_paths=plot_paths,
            radius_in_meters=self.radius_in_meters,
            show=True,
        )
        self.assertEqual(mock_savefig.call_count, 2)
        self.assertEqual(mock_show.call_count, 2)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_generate_results_plots(self, mock_savefig, mock_show):
        circles = GeoDataFrame({"geometry": [self.polygon]}, crs="EPSG:4326")
        found_places = pd.DataFrame(
            {
                "longitude": [0.5],
                "latitude": [0.5],
            }
        )
        plot_paths = _generate_plot_paths(self.plots_folder, self.tag)
        _generate_results_plots(
            polygon=self.polygon,
            circles=circles,
            found_places=found_places,
            plot_paths=plot_paths,
            radius_in_meters=self.radius_in_meters,
            show=True,
        )
        self.assertEqual(mock_savefig.call_count, 4)
        self.assertEqual(mock_show.call_count, 2)

    def test_places_search_step_with_multipolygon(self):
        multipolygon = MultiPolygon([self.polygon])
        found_places, circles, saturated_area, saturated_circles = places_search_step(
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            tag=self.tag,
            polygon=multipolygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            client=self.client,
        )
        self.assertIsInstance(found_places, pd.DataFrame)
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertTrue(
            isinstance(saturated_area, (Polygon, MultiPolygon))
            or saturated_area.is_empty
        )
        self.assertIsInstance(saturated_circles, GeoDataFrame)

    def test_places_search_step_with_no_places(self):
        found_places, circles, saturated_area, saturated_circles = places_search_step(
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            tag=self.tag,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            client=self.client,
            has_places=False,
        )
        self.assertIsInstance(found_places, pd.DataFrame)
        self.assertTrue(found_places.empty)
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertTrue(saturated_area.is_empty)
        self.assertIsInstance(saturated_circles, GeoDataFrame)

    def test_places_search_step_with_recalculate(self):
        found_places, circles, saturated_area, saturated_circles = places_search_step(
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            tag=self.tag,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            client=self.client,
            recalculate=True,
        )
        self.assertIsInstance(found_places, pd.DataFrame)
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertTrue(
            isinstance(saturated_area, (Polygon, MultiPolygon))
            or saturated_area.is_empty
        )
        self.assertIsInstance(saturated_circles, GeoDataFrame)

    def test_places_search_step_with_different_threshold(self):
        found_places, circles, saturated_area, saturated_circles = places_search_step(
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            tag=self.tag,
            polygon=self.polygon,
            radius_in_meters=self.radius_in_meters,
            step_in_degrees=self.step_in_degrees,
            client=self.client,
            threshold=10,
        )
        self.assertIsInstance(found_places, pd.DataFrame)
        self.assertIsInstance(circles, GeoDataFrame)
        self.assertTrue(
            isinstance(saturated_area, (Polygon, MultiPolygon))
            or saturated_area.is_empty
        )
        self.assertIsInstance(saturated_circles, GeoDataFrame)


if __name__ == "__main__":
    unittest.main()
