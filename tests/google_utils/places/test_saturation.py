import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.google_utils.places.saturation import (
    compute_saturated_area,
    compute_saturated_circles,
    filter_saturated_circles,
)


class TestSaturation(unittest.TestCase):
    def setUp(self):
        self.sample_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])

        circles = [
            Polygon([(0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1), (0.1, 0.1)]),
            Polygon([(0.3, 0.3), (0.3, 0.4), (0.4, 0.4), (0.4, 0.3), (0.3, 0.3)]),
            Polygon([(0.5, 0.5), (0.5, 0.6), (0.6, 0.6), (0.6, 0.5), (0.5, 0.5)]),
        ]
        self.sample_circles = gpd.GeoDataFrame(geometry=circles, index=[0, 1, 2])

        self.sample_places = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "circle": [0, 0, 1, 1, 2],
                "longitude": [0.15, 0.15, 0.35, 0.35, 0.55],
                "latitude": [0.15, 0.15, 0.35, 0.35, 0.55],
            }
        )

    def test_filter_saturated_circles_empty_circles(self):
        empty_circles = gpd.GeoDataFrame()
        with self.assertRaisesRegex(ArgumentValueError, "'circles' cannot be empty"):
            filter_saturated_circles(self.sample_places, empty_circles, threshold=1)

    def test_filter_saturated_circles_negative_threshold(self):
        with self.assertRaisesRegex(
            ArgumentValueError, "'threshold' must be a positive integer or 0"
        ):
            filter_saturated_circles(
                self.sample_places, self.sample_circles, threshold=-1
            )

    def test_filter_saturated_circles_invalid_index(self):
        invalid_places = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "circle": [10, 11, 12],
                "longitude": [0.15, 0.15, 0.35],
                "latitude": [0.15, 0.15, 0.35],
            }
        )
        with self.assertRaises(ArgumentValueError):
            filter_saturated_circles(invalid_places, self.sample_circles, threshold=1)

    def test_filter_saturated_circles_threshold_zero(self):
        result = filter_saturated_circles(
            self.sample_places, self.sample_circles, threshold=0
        )
        self.assertEqual(len(result), len(self.sample_circles))

    def test_filter_saturated_circles_threshold_one(self):
        result = filter_saturated_circles(
            self.sample_places, self.sample_circles, threshold=1
        )
        self.assertEqual(len(result), 3)

    def test_filter_saturated_circles_threshold_two(self):
        result = filter_saturated_circles(
            self.sample_places, self.sample_circles, threshold=2
        )
        self.assertEqual(len(result), 2)

    def test_compute_saturated_circles(self):
        with self.subTest():
            with self.temp_dir() as tmp_path:
                result = compute_saturated_circles(
                    self.sample_polygon,
                    self.sample_places,
                    self.sample_circles,
                    threshold=1,
                    show=False,
                    output_path=tmp_path / "test.png",
                )
                self.assertIsInstance(result, gpd.GeoDataFrame)
                self.assertEqual(len(result), 3)
                self.assertTrue((tmp_path / "test.png").exists())

    def test_compute_saturated_circles_no_output(self):
        result = compute_saturated_circles(
            self.sample_polygon,
            self.sample_places,
            self.sample_circles,
            threshold=1,
            show=False,
        )
        self.assertIsInstance(result, gpd.GeoDataFrame)
        self.assertEqual(len(result), 3)

    def test_compute_saturated_area(self):
        with self.subTest():
            with self.temp_dir() as tmp_path:
                result = compute_saturated_area(
                    self.sample_polygon,
                    self.sample_circles,
                    show=False,
                    output_path=tmp_path / "test.png",
                )
                self.assertIsInstance(result, (Polygon, MultiPolygon))
                self.assertTrue((tmp_path / "test.png").exists())

    def test_compute_saturated_area_no_output(self):
        result = compute_saturated_area(
            self.sample_polygon, self.sample_circles, show=False
        )
        self.assertIsInstance(result, (Polygon, MultiPolygon))

    @staticmethod
    def temp_dir():
        import os
        import shutil
        import tempfile

        class TempDir:
            def __init__(self):
                self.temp_dir = tempfile.mkdtemp()

            def __enter__(self):
                return Path(self.temp_dir)

            def __exit__(self, exc_type, exc_val, exc_tb):
                shutil.rmtree(self.temp_dir)

        return TempDir()


if __name__ == "__main__":
    unittest.main()
