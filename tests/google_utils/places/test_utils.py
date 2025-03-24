import math
import unittest
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.google_utils.places.utils import (
    calculate_degree_steps,
    create_subsampled_circles,
    generate_unique_place_id,
    get_circles_search,
    meters_to_degree,
    sample_polygon_with_circles,
    sample_polygons_with_circles,
)


class TestMetersToDegree(unittest.TestCase):
    def test_valid_inputs(self):
        self.assertGreater(meters_to_degree(1000, 0), 0)
        self.assertGreater(meters_to_degree(1000, 45), 0)
        self.assertGreater(meters_to_degree(1000, -45), 0)
        self.assertEqual(meters_to_degree(0, 0), 0)

    def test_invalid_distance(self):
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(-1000, 0)
        with self.assertRaises(ArgumentValueError):
            meters_to_degree("invalid", 0)

    def test_invalid_latitude(self):
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(1000, 91)
        with self.assertRaises(ArgumentValueError):
            meters_to_degree(1000, -91)


class TestCalculateDegreeSteps(unittest.TestCase):
    def test_valid_inputs(self):
        steps = calculate_degree_steps([1000, 2000, 4000])
        self.assertEqual(len(steps), 3)
        self.assertTrue(all(step > 0 for step in steps))
        self.assertEqual(steps[0], 0.00375)

    def test_single_radius(self):
        steps = calculate_degree_steps([1000])
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], 0.00375)

    def test_invalid_radiuses(self):
        with self.assertRaises(ArgumentValueError):
            calculate_degree_steps([])
        with self.assertRaises(ArgumentValueError):
            calculate_degree_steps([0, 1000])
        with self.assertRaises(ArgumentValueError):
            calculate_degree_steps([-1000, 1000])


class TestSamplePolygonWithCircles(unittest.TestCase):
    def setUp(self):
        self.simple_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])

    def test_valid_inputs(self):
        circles = sample_polygon_with_circles(
            self.simple_polygon, radius_in_meters=1000, step_in_degrees=0.1
        )
        self.assertIsInstance(circles, list)
        self.assertTrue(all(isinstance(circle, Polygon) for circle in circles))

    def test_invalid_polygon(self):
        with self.assertRaises(ArgumentTypeError):
            sample_polygon_with_circles("invalid", 1000, 0.1)
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(Polygon(), 1000, 0.1)

    def test_invalid_step(self):
        with self.assertRaises(ArgumentValueError):
            sample_polygon_with_circles(self.simple_polygon, 1000, 0)


class TestSamplePolygonsWithCircles(unittest.TestCase):
    def setUp(self):
        self.simple_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.multi_polygon = MultiPolygon(
            [
                Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),
                Polygon([(2, 0), (2, 1), (3, 1), (3, 0), (2, 0)]),
            ]
        )

    def test_single_polygon(self):
        circles = sample_polygons_with_circles(
            self.simple_polygon, radius_in_meters=1000, step_in_degrees=0.1
        )
        self.assertIsInstance(circles, list)
        self.assertTrue(all(isinstance(circle, Polygon) for circle in circles))

    def test_multi_polygon(self):
        circles = sample_polygons_with_circles(
            self.multi_polygon, radius_in_meters=1000, step_in_degrees=0.1
        )
        self.assertIsInstance(circles, list)
        self.assertTrue(all(isinstance(circle, Polygon) for circle in circles))

    def test_polygon_list(self):
        circles = sample_polygons_with_circles(
            [self.simple_polygon, self.simple_polygon],
            radius_in_meters=1000,
            step_in_degrees=0.1,
        )
        self.assertIsInstance(circles, list)
        self.assertTrue(all(isinstance(circle, Polygon) for circle in circles))

    def test_invalid_input(self):
        with self.assertRaises(ArgumentTypeError):
            sample_polygons_with_circles("invalid", 1000, 0.1)


class TestGetCirclesSearch(unittest.TestCase):
    def setUp(self):
        self.simple_polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.temp_path = Path("test_circles.geojson")

    def tearDown(self):
        if self.temp_path.exists():
            self.temp_path.unlink()

    def test_create_new_file(self):
        gdf = get_circles_search(
            self.temp_path,
            self.simple_polygon,
            radius_in_meters=1000,
            step_in_degrees=0.1,
        )
        self.assertIsInstance(gdf, GeoDataFrame)
        self.assertIn("searched", gdf.columns)
        self.assertTrue(self.temp_path.exists())

    def test_read_existing_file(self):
        gdf1 = get_circles_search(
            self.temp_path,
            self.simple_polygon,
            radius_in_meters=1000,
            step_in_degrees=0.1,
        )
        gdf2 = get_circles_search(
            self.temp_path,
            self.simple_polygon,
            radius_in_meters=1000,
            step_in_degrees=0.1,
        )
        self.assertEqual(len(gdf1), len(gdf2))

    def test_recalculate(self):
        gdf1 = get_circles_search(
            self.temp_path,
            self.simple_polygon,
            radius_in_meters=1000,
            step_in_degrees=0.1,
        )
        gdf2 = get_circles_search(
            self.temp_path,
            self.simple_polygon,
            radius_in_meters=1000,
            step_in_degrees=0.1,
            recalculate=True,
        )
        self.assertEqual(len(gdf1), len(gdf2))


class TestCreateSubsampledCircles(unittest.TestCase):
    def setUp(self):
        self.center_point = Point(0, 0)

    def test_valid_inputs(self):
        circles = create_subsampled_circles(
            self.center_point, large_radius=1000, small_radius=100, radial_samples=8
        )
        self.assertIsInstance(circles, list)
        self.assertTrue(all(isinstance(circle, Polygon) for circle in circles))
        self.assertGreater(len(circles), 1)

    def test_invalid_center(self):
        with self.assertRaises(ArgumentTypeError):
            create_subsampled_circles("invalid", 1000, 100, 8)

    def test_invalid_radii(self):
        with self.assertRaises(ArgumentValueError):
            create_subsampled_circles(self.center_point, 0, 100, 8)
        with self.assertRaises(ArgumentValueError):
            create_subsampled_circles(self.center_point, 1000, 0, 8)

    def test_invalid_samples(self):
        with self.assertRaises(ArgumentValueError):
            create_subsampled_circles(self.center_point, 1000, 100, 0)


class TestGenerateUniquePlaceId(unittest.TestCase):
    def test_format(self):
        place_id = generate_unique_place_id()
        self.assertIsInstance(place_id, str)
        self.assertEqual(len(place_id), 20)
        self.assertTrue(place_id.isdigit())

    def test_uniqueness(self):
        ids = [generate_unique_place_id() for _ in range(100)]
        self.assertEqual(len(set(ids)), 100)


if __name__ == "__main__":
    unittest.main()
