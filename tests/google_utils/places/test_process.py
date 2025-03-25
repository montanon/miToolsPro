import unittest
from pathlib import Path

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from tqdm import tqdm

from mitoolspro.google_utils.places.client import (
    GooglePlacesClient,
    create_dummy_response,
)
from mitoolspro.google_utils.places.models import NewPlace
from mitoolspro.google_utils.places.process import (
    global_requests_counter,
    global_requests_counter_limit,
    process_circles,
    process_single_circle,
    should_do_search,
    should_process_circles,
    should_save_state,
    update_progress_bar,
)


class TestProcessFunctions(unittest.TestCase):
    def setUp(self):
        self.test_circle = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        self.test_circles = GeoDataFrame(
            {"geometry": [self.test_circle], "searched": [False]},
            crs="EPSG:4326",
        )
        self.test_file_path = Path("test_places.parquet")
        self.test_circles_path = Path("test_circles.geojson")
        self.test_client = GooglePlacesClient()
        self.test_pbar = tqdm(total=1, desc="Test Progress")

    def tearDown(self):
        if self.test_file_path.exists():
            self.test_file_path.unlink()
        if self.test_circles_path.exists():
            self.test_circles_path.unlink()
        self.test_pbar.close()

    def test_process_circles_empty_circles(self):
        empty_circles = GeoDataFrame(columns=["geometry", "searched"], crs="EPSG:4326")
        result = process_circles(
            empty_circles,
            1000,
            self.test_file_path,
            self.test_circles_path,
            self.test_client,
        )
        self.assertTrue(result.empty)
        self.assertEqual(len(result.columns), len(NewPlace.__annotations__) + 1)

    def test_process_circles_existing_file(self):
        # Create initial test data using dummy response
        query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": 0.0, "longitude": 0.0},
                    "radius": 1000.0,
                }
            }
        }
        dummy_response = create_dummy_response(query, has_places=True)
        initial_places = self.test_client.get_response_places(0, dummy_response.places)
        initial_places.to_parquet(self.test_file_path)

        # Process circles and get result
        result = process_circles(
            self.test_circles,
            1000,
            self.test_file_path,
            self.test_circles_path,
            self.test_client,
        )

        # Verify that initial places are preserved
        for _, initial_place in initial_places.iterrows():
            matching_place = result[result["id"] == initial_place["id"]]
            self.assertFalse(
                matching_place.empty,
                f"Initial place {initial_place['id']} not found in result",
            )
            pd.testing.assert_series_equal(matching_place.iloc[0], initial_place)

        # Verify that there are additional places in the result
        self.assertGreater(
            len(result), len(initial_places), "No additional places were added"
        )

    def test_process_single_circle_invalid_geometry(self):
        with self.assertRaises(Exception):
            process_single_circle(
                0,
                "invalid_geometry",
                1000,
                pd.DataFrame(),
                self.test_circles,
                self.test_file_path,
                self.test_circles_path,
                self.test_pbar,
                self.test_client,
            )

    def test_process_single_circle_success(self):
        result = process_single_circle(
            0,
            self.test_circle,
            1000,
            pd.DataFrame(),
            self.test_circles,
            self.test_file_path,
            self.test_circles_path,
            self.test_pbar,
            self.test_client,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_should_process_circles_true(self):
        circles = GeoDataFrame(
            {"geometry": [self.test_circle], "searched": [False]}, crs="EPSG:4326"
        )
        self.assertTrue(should_process_circles(circles, False))

    def test_should_process_circles_false(self):
        circles = GeoDataFrame(
            {"geometry": [self.test_circle], "searched": [True]}, crs="EPSG:4326"
        )
        self.assertFalse(should_process_circles(circles, False))

    def test_should_process_circles_recalculate(self):
        circles = GeoDataFrame(
            {"geometry": [self.test_circle], "searched": [True]}, crs="EPSG:4326"
        )
        self.assertTrue(should_process_circles(circles, True))

    def test_should_save_state_regular_interval(self):
        self.assertTrue(should_save_state(200, 1000))
        self.assertFalse(should_save_state(199, 1000))

    def test_should_save_state_last_circle(self):
        self.assertTrue(should_save_state(999, 1000))
        self.assertFalse(should_save_state(998, 1000))

    def test_should_save_state_request_limit(self):
        global_requests_counter.value = global_requests_counter_limit.value - 1
        self.assertTrue(should_save_state(1, 1000))

    def test_should_do_search(self):
        global_requests_counter.value = 0
        self.assertTrue(should_do_search())

        global_requests_counter.value = global_requests_counter_limit.value
        self.assertFalse(should_do_search())

    def test_update_progress_bar(self):
        circles = GeoDataFrame(
            {
                "geometry": [self.test_circle, self.test_circle],
                "searched": [True, False],
            },
            crs="EPSG:4326",
        )
        found_places = pd.DataFrame(
            {
                "id": ["20250325102746920384", "20250325102746920385"],
                "types": [["restaurant"], ["cafe"]],
                "location": [
                    {"latitude": 0.0, "longitude": 0.0},
                    {"latitude": 0.0, "longitude": 0.0},
                ],
                "displayName": [
                    {"text": "Place1", "languageCode": "en"},
                    {"text": "Place2", "languageCode": "en"},
                ],
                "primaryType": ["restaurant", "cafe"],
                "primaryTypeDisplayName": [
                    {"text": "Restaurant", "languageCode": "en"},
                    {"text": "Cafe", "languageCode": "en"},
                ],
                "formattedAddress": ["Address1", "Address2"],
                "addressComponents": [
                    [
                        {
                            "longText": "City",
                            "shortText": "C",
                            "types": ["locality"],
                            "languageCode": "en",
                        }
                    ],
                    [
                        {
                            "longText": "City",
                            "shortText": "C",
                            "types": ["locality"],
                            "languageCode": "en",
                        }
                    ],
                ],
                "googleMapsUri": [
                    "http://maps.google.com/1",
                    "http://maps.google.com/2",
                ],
                "priceLevel": ["1", "2"],
                "rating": [4.5, 4.0],
                "userRatingCount": [100, 50],
                "circle": [0, 0],
            }
        )

        update_progress_bar(self.test_pbar, circles, found_places)
        self.assertEqual(self.test_pbar.n, 1)

    def test_process_circles_with_included_types(self):
        included_types = ["restaurant", "cafe"]
        process_circles(
            self.test_circles,
            1000,
            self.test_file_path,
            self.test_circles_path,
            self.test_client,
            included_types=included_types,
        )
        self.assertTrue(self.test_file_path.exists())

    def test_process_circles_with_has_places_false(self):
        process_circles(
            self.test_circles,
            1000,
            self.test_file_path,
            self.test_circles_path,
            self.test_client,
            has_places=False,
        )
        self.assertTrue(self.test_file_path.exists())


if __name__ == "__main__":
    unittest.main()
