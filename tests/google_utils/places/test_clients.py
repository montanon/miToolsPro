import unittest
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from shapely.geometry import Point

from mitoolspro.google_utils.places.client import (
    FIELD_MASK,
    RESTAURANT_TYPES,
    GooglePlacesClient,
    create_dummy_place,
    create_dummy_response,
)
from mitoolspro.google_utils.places.models import (
    AccessibilityOptions,
    AddressComponent,
    Coordinate,
    DummyResponse,
    LocalizedText,
    NewPlace,
    NewPlacesResponse,
    OpeningHours,
    Period,
    PlusCode,
    TimePeriod,
    Viewport,
)


class TestGooglePlacesClient(TestCase):
    def setUp(self):
        self.client = GooglePlacesClient()
        self.test_point = Point(150.644, -34.397)
        self.test_radius = 1000.0
        self.test_response_id = "test_circle_1"

    def test_initialization(self):
        client = GooglePlacesClient()
        self.assertEqual(client.places_types, RESTAURANT_TYPES)
        self.assertEqual(client.field_mask, FIELD_MASK)
        self.assertIsNone(client.api_key)

        client_with_key = GooglePlacesClient(api_key="test_key")
        self.assertEqual(client_with_key.api_key, "test_key")

        custom_types = ["restaurant", "cafe"]
        client_with_types = GooglePlacesClient(places_types=custom_types)
        self.assertEqual(client_with_types.places_types, custom_types)

    def test_build_headers(self):
        client = GooglePlacesClient()
        headers = client._build_headers()
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["X-Goog-FieldMask"], FIELD_MASK)
        self.assertNotIn("X-Goog-Api-Key", headers)

        client_with_key = GooglePlacesClient(api_key="test_key")
        headers = client_with_key._build_headers()
        self.assertEqual(headers["X-Goog-Api-Key"], "test_key")

    def test_search_nearby_without_api_key(self):
        client = GooglePlacesClient()
        response = client.search_nearby(
            center_point=self.test_point,
            radius_in_meters=self.test_radius,
        )
        self.assertIsInstance(response, NewPlacesResponse)
        self.assertTrue(hasattr(response, "places"))

    @patch("requests.post")
    def test_search_nearby_with_api_key(self, mock_post):
        client = GooglePlacesClient(api_key="test_key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "places": [
                create_dummy_place(
                    {
                        "locationRestriction": {
                            "circle": {
                                "center": {"latitude": -34.397, "longitude": 150.644},
                                "radius": 1000.0,
                            }
                        }
                    }
                )
            ]
        }
        mock_post.return_value = mock_response

        response = client.search_nearby(
            center_point=self.test_point,
            radius_in_meters=self.test_radius,
        )
        self.assertIsInstance(response, NewPlacesResponse)
        self.assertTrue(hasattr(response, "places"))
        mock_post.assert_called_once()

    def test_search_nearby_with_custom_types(self):
        client = GooglePlacesClient()
        custom_types = ["restaurant", "cafe"]
        response = client.search_nearby(
            center_point=self.test_point,
            radius_in_meters=self.test_radius,
            included_types=custom_types,
        )
        self.assertIsInstance(response, NewPlacesResponse)
        self.assertTrue(hasattr(response, "places"))

    def test_get_response_places(self):
        client = GooglePlacesClient()
        dummy_place = create_dummy_place(
            {
                "locationRestriction": {
                    "circle": {
                        "center": {"latitude": -34.397, "longitude": 150.644},
                        "radius": 1000.0,
                    }
                }
            }
        )
        places = [dummy_place]
        df = client.get_response_places(self.test_response_id, places)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["circle"].iloc[0], self.test_response_id)

    def test_search_for_places(self):
        client = GooglePlacesClient()
        df = client.search_for_places(
            center_point=self.test_point,
            radius_in_meters=self.test_radius,
            response_id=self.test_response_id,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue("circle" in df.columns)
        self.assertEqual(df["circle"].iloc[0], self.test_response_id)

    def test_search_for_places_invalid_center_point(self):
        client = GooglePlacesClient()
        with self.assertRaises(Exception):
            client.search_for_places(
                center_point="invalid_point",
                radius_in_meters=self.test_radius,
                response_id=self.test_response_id,
            )

    def test_search_for_places_with_error(self):
        client = GooglePlacesClient()
        with patch.object(client, "search_nearby", side_effect=Exception("Test error")):
            df = client.search_for_places(
                center_point=self.test_point,
                radius_in_meters=self.test_radius,
                response_id=self.test_response_id,
            )
            self.assertIsNone(df)

    def test_dummy_response_creation(self):
        query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": -34.397, "longitude": 150.644},
                    "radius": 1000.0,
                }
            }
        }
        response = create_dummy_response(query, has_places=True)
        self.assertIsInstance(response, DummyResponse)
        self.assertTrue(hasattr(response, "data"))
        self.assertTrue("places" in response.data)
        self.assertTrue(len(response.data["places"]) > 0)

    def test_dummy_place_creation(self):
        query = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": -34.397, "longitude": 150.644},
                    "radius": 1000.0,
                }
            }
        }
        place = create_dummy_place(query)

        # Verify it's a NewPlace instance
        self.assertIsInstance(place, NewPlace)

        # Verify all required fields are present and have correct types
        self.assertIsInstance(place.id, str)
        self.assertIsInstance(place.name, str)
        self.assertIsInstance(place.types, list)
        self.assertIsInstance(place.formattedAddress, str)
        self.assertIsInstance(place.shortFormattedAddress, str)
        self.assertIsInstance(place.adrFormatAddress, str)
        self.assertIsInstance(place.addressComponents, list)
        self.assertIsInstance(place.plusCode, PlusCode)
        self.assertIsInstance(place.location, Coordinate)
        self.assertIsInstance(place.viewport, Viewport)
        self.assertIsInstance(place.googleMapsUri, str)
        self.assertIsInstance(place.websiteUri, str)
        self.assertIsInstance(place.businessStatus, str)
        self.assertIsInstance(place.rating, float)
        self.assertIsInstance(place.userRatingCount, int)
        self.assertIsInstance(place.priceLevel, str)
        self.assertIsInstance(place.nationalPhoneNumber, str)
        self.assertIsInstance(place.internationalPhoneNumber, str)
        self.assertIsInstance(place.utcOffsetMinutes, int)
        self.assertIsInstance(place.iconMaskBaseUri, str)
        self.assertIsInstance(place.iconBackgroundColor, str)
        self.assertIsInstance(place.displayName, LocalizedText)
        self.assertIsInstance(place.primaryType, str)
        self.assertIsInstance(place.primaryTypeDisplayName, LocalizedText)
        self.assertIsInstance(place.currentOpeningHours, OpeningHours)
        self.assertIsInstance(place.regularOpeningHours, OpeningHours)
        self.assertIsInstance(place.accessibilityOptions, AccessibilityOptions)

        # Verify nested objects have correct structure
        self.assertIsInstance(place.addressComponents[0], AddressComponent)
        self.assertIsInstance(place.location.latitude, float)
        self.assertIsInstance(place.location.longitude, float)
        self.assertIsInstance(place.viewport.low, Coordinate)
        self.assertIsInstance(place.viewport.high, Coordinate)
        self.assertIsInstance(place.displayName.text, str)
        self.assertIsInstance(place.displayName.languageCode, str)
        self.assertIsInstance(place.currentOpeningHours.periods[0], Period)
        self.assertIsInstance(place.currentOpeningHours.periods[0].open, TimePeriod)
        self.assertIsInstance(place.currentOpeningHours.periods[0].close, TimePeriod)

        # Verify data constraints
        self.assertTrue(0 <= place.rating <= 5)
        self.assertTrue(place.userRatingCount >= 0)
        self.assertTrue(place.priceLevel in ["1", "2", "3", "4", "5"])
        self.assertTrue(place.utcOffsetMinutes in range(-720, 721, 60))
        self.assertTrue(all(t in RESTAURANT_TYPES for t in place.types))
        self.assertTrue(place.primaryType in place.types)


if __name__ == "__main__":
    unittest.main()
