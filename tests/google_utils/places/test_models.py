import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

import geopandas as gpd
from shapely import Point, Polygon

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.google_utils.places.models import (
    AccessibilityOptions,
    AddressComponent,
    CircleCenterInsidePolygon,
    CircleInsidePolygon,
    CircleIntersectsPolygon,
    CircleType,
    CityGeojson,
    Coordinate,
    DateStamp,
    DummyResponse,
    LocalizedText,
    NewNearbySearchRequest,
    NewPlace,
    NewPlacesResponse,
    OpeningHours,
    Period,
    PlusCode,
    TimePeriod,
    Viewport,
    intersection_condition_factory,
)


class TestCoordinate(TestCase):
    def setUp(self):
        self.coordinate = Coordinate(latitude=35.6895, longitude=139.6917)

    def test_initialization(self):
        self.assertEqual(self.coordinate.latitude, 35.6895)
        self.assertEqual(self.coordinate.longitude, 139.6917)

    def test_invalid_latitude(self):
        with self.assertRaises(ArgumentValueError):
            Coordinate(latitude=91.0, longitude=139.6917)
        with self.assertRaises(ArgumentValueError):
            Coordinate(latitude=-91.0, longitude=139.6917)

    def test_invalid_longitude(self):
        with self.assertRaises(ArgumentValueError):
            Coordinate(latitude=35.6895, longitude=181.0)
        with self.assertRaises(ArgumentValueError):
            Coordinate(latitude=35.6895, longitude=-181.0)


class TestViewport(TestCase):
    def setUp(self):
        self.viewport = Viewport(
            low=Coordinate(latitude=35.6895, longitude=139.6917),
            high=Coordinate(latitude=35.6896, longitude=139.6918),
        )

    def test_initialization(self):
        self.assertEqual(self.viewport.low.latitude, 35.6895)
        self.assertEqual(self.viewport.low.longitude, 139.6917)
        self.assertEqual(self.viewport.high.latitude, 35.6896)
        self.assertEqual(self.viewport.high.longitude, 139.6918)

    def test_invalid_coordinates(self):
        with self.assertRaises(ArgumentValueError):
            Viewport(
                low=Coordinate(latitude=35.6896, longitude=139.6918),
                high=Coordinate(latitude=35.6895, longitude=139.6917),
            )


class TestPlusCode(TestCase):
    def setUp(self):
        self.plus_code = PlusCode(
            globalCode="87G8P27V+JG",
            compoundCode="P27V+JG Tokyo, Japan",
        )

    def test_initialization(self):
        self.assertEqual(self.plus_code.globalCode, "87G8P27V+JG")
        self.assertEqual(self.plus_code.compoundCode, "P27V+JG Tokyo, Japan")

    def test_optional_fields(self):
        plus_code = PlusCode()
        self.assertIsNone(plus_code.globalCode)
        self.assertIsNone(plus_code.compoundCode)


class TestAddressComponent(TestCase):
    def setUp(self):
        self.address_component = AddressComponent(
            longText="Tokyo",
            shortText="TKY",
            types=["locality", "political"],
            languageCode="en",
        )

    def test_initialization(self):
        self.assertEqual(self.address_component.longText, "Tokyo")
        self.assertEqual(self.address_component.shortText, "TKY")
        self.assertEqual(self.address_component.types, ["locality", "political"])
        self.assertEqual(self.address_component.languageCode, "en")


class TestDateStamp(TestCase):
    def setUp(self):
        self.date_stamp = DateStamp(year=2024, month=3, day=15)

    def test_initialization(self):
        self.assertEqual(self.date_stamp.year, 2024)
        self.assertEqual(self.date_stamp.month, 3)
        self.assertEqual(self.date_stamp.day, 15)

    def test_invalid_month(self):
        with self.assertRaises(ArgumentValueError):
            DateStamp(year=2024, month=13, day=15)
        with self.assertRaises(ArgumentValueError):
            DateStamp(year=2024, month=0, day=15)

    def test_invalid_day(self):
        with self.assertRaises(ArgumentValueError):
            DateStamp(year=2024, month=3, day=32)
        with self.assertRaises(ArgumentValueError):
            DateStamp(year=2024, month=3, day=0)


class TestTimePeriod(TestCase):
    def setUp(self):
        self.time_period = TimePeriod(
            day=1,
            hour=9,
            minute=0,
            date=DateStamp(year=2024, month=3, day=15),
        )

    def test_initialization(self):
        self.assertEqual(self.time_period.day, 1)
        self.assertEqual(self.time_period.hour, 9)
        self.assertEqual(self.time_period.minute, 0)
        self.assertEqual(self.time_period.date.year, 2024)
        self.assertEqual(self.time_period.date.month, 3)
        self.assertEqual(self.time_period.date.day, 15)

    def test_invalid_hour(self):
        with self.assertRaises(ArgumentValueError):
            TimePeriod(day=1, hour=24, minute=0)
        with self.assertRaises(ArgumentValueError):
            TimePeriod(day=1, hour=-1, minute=0)

    def test_invalid_minute(self):
        with self.assertRaises(ArgumentValueError):
            TimePeriod(day=1, hour=9, minute=60)
        with self.assertRaises(ArgumentValueError):
            TimePeriod(day=1, hour=9, minute=-1)


class TestPeriod(TestCase):
    def setUp(self):
        self.period = Period(
            open=TimePeriod(day=1, hour=9, minute=0),
            close=TimePeriod(day=1, hour=17, minute=0),
        )

    def test_initialization(self):
        self.assertEqual(self.period.open.day, 1)
        self.assertEqual(self.period.open.hour, 9)
        self.assertEqual(self.period.open.minute, 0)
        self.assertEqual(self.period.close.day, 1)
        self.assertEqual(self.period.close.hour, 17)
        self.assertEqual(self.period.close.minute, 0)

    def test_optional_fields(self):
        period = Period()
        self.assertIsNone(period.open)
        self.assertIsNone(period.close)


class TestOpeningHours(TestCase):
    def setUp(self):
        self.opening_hours = OpeningHours(
            openNow=True,
            periods=[
                Period(
                    open=TimePeriod(day=1, hour=9, minute=0),
                    close=TimePeriod(day=1, hour=17, minute=0),
                )
            ],
            weekdayDescriptions=["Monday: 9:00 AM – 5:00 PM"],
            nextOpenTime="2024-03-18T09:00:00",
        )

    def test_initialization(self):
        self.assertTrue(self.opening_hours.openNow)
        self.assertEqual(len(self.opening_hours.periods), 1)
        self.assertEqual(
            self.opening_hours.weekdayDescriptions, ["Monday: 9:00 AM – 5:00 PM"]
        )
        self.assertEqual(self.opening_hours.nextOpenTime, "2024-03-18T09:00:00")

    def test_optional_fields(self):
        opening_hours = OpeningHours()
        self.assertIsNone(opening_hours.openNow)
        self.assertIsNone(opening_hours.periods)
        self.assertIsNone(opening_hours.weekdayDescriptions)
        self.assertIsNone(opening_hours.nextOpenTime)


class TestAccessibilityOptions(TestCase):
    def setUp(self):
        self.accessibility = AccessibilityOptions(
            wheelchairAccessibleSeating=True,
            wheelchairAccessibleParking=True,
            wheelchairAccessibleEntrance=True,
            wheelchairAccessibleRestroom=True,
        )

    def test_initialization(self):
        self.assertTrue(self.accessibility.wheelchairAccessibleSeating)
        self.assertTrue(self.accessibility.wheelchairAccessibleParking)
        self.assertTrue(self.accessibility.wheelchairAccessibleEntrance)
        self.assertTrue(self.accessibility.wheelchairAccessibleRestroom)

    def test_optional_fields(self):
        accessibility = AccessibilityOptions()
        self.assertIsNone(accessibility.wheelchairAccessibleSeating)
        self.assertIsNone(accessibility.wheelchairAccessibleParking)
        self.assertIsNone(accessibility.wheelchairAccessibleEntrance)
        self.assertIsNone(accessibility.wheelchairAccessibleRestroom)


class TestLocalizedText(TestCase):
    def setUp(self):
        self.localized_text = LocalizedText(
            text="Tokyo Tower",
            languageCode="en",
        )

    def test_initialization(self):
        self.assertEqual(self.localized_text.text, "Tokyo Tower")
        self.assertEqual(self.localized_text.languageCode, "en")

    def test_optional_fields(self):
        localized_text = LocalizedText()
        self.assertIsNone(localized_text.text)
        self.assertIsNone(localized_text.languageCode)


class TestNewPlace(TestCase):
    def setUp(self):
        self.place = NewPlace(
            id="ChIJN1t_tDeuEmsRUsoyG83frY4",
            name="Google Japan",
            types=["point_of_interest", "establishment"],
            formattedAddress="6 Chome-7-15 Roppongi, Minato City, Tokyo 106-0032, Japan",
            shortFormattedAddress="6 Chome-7-15 Roppongi, Minato City, Tokyo",
            adrFormatAddress="6 Chome-7-15 Roppongi, Minato City, Tokyo 106-0032, Japan",
            addressComponents=[
                AddressComponent(
                    longText="Roppongi",
                    shortText="Roppongi",
                    types=["neighborhood", "political"],
                    languageCode="en",
                )
            ],
            plusCode=PlusCode(
                globalCode="87G8P27V+JG",
                compoundCode="P27V+JG Tokyo, Japan",
            ),
            location=Coordinate(latitude=35.6895, longitude=139.6917),
            viewport=Viewport(
                low=Coordinate(latitude=35.6894, longitude=139.6916),
                high=Coordinate(latitude=35.6896, longitude=139.6918),
            ),
            googleMapsUri="https://maps.google.com/?cid=123456789",
            websiteUri="https://www.google.com",
            businessStatus="OPERATIONAL",
            rating=4.5,
            userRatingCount=1000,
            priceLevel="$$",
            nationalPhoneNumber="+81 3-1234-5678",
            internationalPhoneNumber="+81 3-1234-5678",
            utcOffsetMinutes=540,
            iconMaskBaseUri="https://maps.gstatic.com/mapfiles/place_api/icons/v1/png_71/restaurant-71.png",
            iconBackgroundColor="#FF9E67",
            displayName=LocalizedText(text="Google Japan", languageCode="en"),
            primaryType="point_of_interest",
            primaryTypeDisplayName=LocalizedText(
                text="Point of Interest", languageCode="en"
            ),
            currentOpeningHours=OpeningHours(
                openNow=True,
                periods=[
                    Period(
                        open=TimePeriod(day=1, hour=9, minute=0),
                        close=TimePeriod(day=1, hour=17, minute=0),
                    )
                ],
            ),
            regularOpeningHours=OpeningHours(
                openNow=True,
                periods=[
                    Period(
                        open=TimePeriod(day=1, hour=9, minute=0),
                        close=TimePeriod(day=1, hour=17, minute=0),
                    )
                ],
            ),
            accessibilityOptions=AccessibilityOptions(
                wheelchairAccessibleSeating=True,
                wheelchairAccessibleParking=True,
                wheelchairAccessibleEntrance=True,
                wheelchairAccessibleRestroom=True,
            ),
        )

    def test_initialization(self):
        self.assertEqual(self.place.id, "ChIJN1t_tDeuEmsRUsoyG83frY4")
        self.assertEqual(self.place.name, "Google Japan")
        self.assertEqual(self.place.types, ["point_of_interest", "establishment"])
        self.assertEqual(
            self.place.formattedAddress,
            "6 Chome-7-15 Roppongi, Minato City, Tokyo 106-0032, Japan",
        )
        self.assertEqual(
            self.place.shortFormattedAddress,
            "6 Chome-7-15 Roppongi, Minato City, Tokyo",
        )
        self.assertEqual(
            self.place.adrFormatAddress,
            "6 Chome-7-15 Roppongi, Minato City, Tokyo 106-0032, Japan",
        )
        self.assertEqual(len(self.place.addressComponents), 1)
        self.assertEqual(self.place.addressComponents[0].longText, "Roppongi")
        self.assertEqual(self.place.plusCode.globalCode, "87G8P27V+JG")
        self.assertEqual(self.place.location.latitude, 35.6895)
        self.assertEqual(self.place.location.longitude, 139.6917)
        self.assertEqual(self.place.viewport.low.latitude, 35.6894)
        self.assertEqual(self.place.viewport.high.latitude, 35.6896)
        self.assertEqual(
            self.place.googleMapsUri, "https://maps.google.com/?cid=123456789"
        )
        self.assertEqual(self.place.websiteUri, "https://www.google.com")
        self.assertEqual(self.place.businessStatus, "OPERATIONAL")
        self.assertEqual(self.place.rating, 4.5)
        self.assertEqual(self.place.userRatingCount, 1000)
        self.assertEqual(self.place.priceLevel, "$$")
        self.assertEqual(self.place.nationalPhoneNumber, "+81 3-1234-5678")
        self.assertEqual(self.place.internationalPhoneNumber, "+81 3-1234-5678")
        self.assertEqual(self.place.utcOffsetMinutes, 540)
        self.assertEqual(
            self.place.iconMaskBaseUri,
            "https://maps.gstatic.com/mapfiles/place_api/icons/v1/png_71/restaurant-71.png",
        )
        self.assertEqual(self.place.iconBackgroundColor, "#FF9E67")
        self.assertEqual(self.place.displayName.text, "Google Japan")
        self.assertEqual(self.place.primaryType, "point_of_interest")
        self.assertEqual(self.place.primaryTypeDisplayName.text, "Point of Interest")
        self.assertTrue(self.place.currentOpeningHours.openNow)
        self.assertTrue(self.place.regularOpeningHours.openNow)
        self.assertTrue(self.place.accessibilityOptions.wheelchairAccessibleSeating)

    def test_post_init(self):
        self.assertEqual(self.place.latitude, 35.6895)
        self.assertEqual(self.place.longitude, 139.6917)
        self.assertEqual(self.place.place_name, "Google Japan")


class TestNewPlacesResponse(TestCase):
    def setUp(self):
        self.place = NewPlace(
            id="ChIJN1t_tDeuEmsRUsoyG83frY4",
            name="Google Japan",
            types=["point_of_interest", "establishment"],
            location=Coordinate(latitude=35.6895, longitude=139.6917),
        )
        self.response = NewPlacesResponse(places=[self.place])

    def test_initialization(self):
        self.assertEqual(len(self.response.places), 1)
        self.assertEqual(self.response.places[0].id, "ChIJN1t_tDeuEmsRUsoyG83frY4")
        self.assertEqual(self.response.places[0].name, "Google Japan")

    def test_empty_places(self):
        response = NewPlacesResponse()
        self.assertEqual(len(response.places), 0)


class TestDummyResponse(TestCase):
    def setUp(self):
        self.data = {"key": "value"}
        self.response = DummyResponse(data=self.data, status_code=200)

    def test_initialization(self):
        self.assertEqual(self.response.data, {"key": "value"})
        self.assertEqual(self.response.status_code, 200)
        self.assertEqual(self.response.reason, "OK")

    def test_json(self):
        self.assertEqual(self.response.json(), {"key": "value"})

    def test_error_status(self):
        response = DummyResponse(data={"error": "Not found"}, status_code=404)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.reason, "Error")


class TestNewNearbySearchRequest(TestCase):
    def setUp(self):
        self.location = Point(139.6917, 35.6895)
        self.request = NewNearbySearchRequest(
            location=self.location,
            distance_in_meters=1000,
            max_result_count=20,
            included_types=["restaurant"],
            language_code="en",
        )

    def test_initialization(self):
        self.assertEqual(self.request.location, self.location)
        self.assertEqual(self.request.distance_in_meters, 1000)
        self.assertEqual(self.request.max_result_count, 20)
        self.assertEqual(self.request.included_types, ["restaurant"])
        self.assertEqual(self.request.language_code, "en")

    def test_invalid_distance(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=0,
                max_result_count=20,
            )
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=-1000,
                max_result_count=20,
            )

    def test_invalid_max_result_count(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=1000,
                max_result_count=0,
            )
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=1000,
                max_result_count=-20,
            )

    def test_invalid_language_code(self):
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=1000,
                language_code="eng",
            )
        with self.assertRaises(ArgumentValueError):
            NewNearbySearchRequest(
                location=self.location,
                distance_in_meters=1000,
                language_code="e",
            )

    def test_location_restriction(self):
        restriction = self.request.location_restriction
        self.assertEqual(restriction["circle"]["center"]["latitude"], 35.6895)
        self.assertEqual(restriction["circle"]["center"]["longitude"], 139.6917)
        self.assertEqual(restriction["circle"]["radius"], 1000)

    def test_json_query(self):
        query = self.request.json_query()
        self.assertEqual(query["includedTypes"], ["restaurant"])
        self.assertEqual(query["maxResultCount"], 20)
        self.assertEqual(query["languageCode"], "en")
        self.assertEqual(
            query["locationRestriction"]["circle"]["center"]["latitude"], 35.6895
        )
        self.assertEqual(
            query["locationRestriction"]["circle"]["center"]["longitude"], 139.6917
        )
        self.assertEqual(query["locationRestriction"]["circle"]["radius"], 1000)


class TestCityGeojson(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_geojson = Path(self.temp_dir.name) / "test_city.geojson"

        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"Wards": "Test Ward"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                    },
                }
            ],
        }

        import json

        with open(self.test_geojson, "w") as f:
            json.dump(geojson_data, f)

        self.city = CityGeojson(self.test_geojson, "test_city")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initialization(self):
        self.assertEqual(self.city.name, "test_city")
        self.assertEqual(self.city.plots_width, 14)
        self.assertEqual(self.city.plots_aspect_ratio, 16.0 / 9.0)
        self.assertEqual(self.city.plots_height, 14 / (16.0 / 9.0))

    def test_invalid_path(self):
        with self.assertRaises(ArgumentValueError):
            CityGeojson("nonexistent.geojson", "test_city")

    def test_plot_polygons(self):
        ax = self.city.plot_polygons()
        self.assertEqual(ax.get_xlabel(), "Longitude")
        self.assertEqual(ax.get_ylabel(), "Latitude")
        self.assertEqual(ax.get_title(), "Test City Wards Polygons")

    def test_plot_unary_polygon(self):
        ax = self.city.plot_unary_polygon()
        self.assertEqual(ax.get_xlabel(), "Longitude")
        self.assertEqual(ax.get_ylabel(), "Latitude")
        self.assertEqual(ax.get_title(), "Test City Polygon")


class TestIntersectionConditions(TestCase):
    def setUp(self):
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        self.circle = Point(0.5, 0.5).buffer(0.2)

    def test_circle_inside_polygon(self):
        condition = CircleInsidePolygon()
        self.assertTrue(condition.check(self.polygon, self.circle))
        circle_outside = Point(2, 2).buffer(0.2)
        self.assertFalse(condition.check(self.polygon, circle_outside))

    def test_circle_center_inside_polygon(self):
        condition = CircleCenterInsidePolygon()
        self.assertTrue(condition.check(self.polygon, self.circle))
        circle_outside = Point(2, 2).buffer(0.2)
        self.assertFalse(condition.check(self.polygon, circle_outside))

    def test_circle_intersects_polygon(self):
        condition = CircleIntersectsPolygon()
        self.assertTrue(condition.check(self.polygon, self.circle))
        circle_outside = Point(2, 2).buffer(0.2)
        self.assertFalse(condition.check(self.polygon, circle_outside))

    def test_intersection_condition_factory(self):
        condition = intersection_condition_factory("circle")
        self.assertIsInstance(condition, CircleInsidePolygon)
        condition = intersection_condition_factory("center")
        self.assertIsInstance(condition, CircleCenterInsidePolygon)
        condition = intersection_condition_factory("intersection")
        self.assertIsInstance(condition, CircleIntersectsPolygon)
        with self.assertRaises(ArgumentValueError):
            intersection_condition_factory("invalid")


if __name__ == "__main__":
    unittest.main()
