import random
import traceback
from typing import Any, Dict, List, Optional, Union

import requests
from pandas import DataFrame
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

from mitoolspro.exceptions import ArgumentTypeError
from mitoolspro.google_utils.places.models import (
    AccessibilityOptions,
    AddressComponent,
    Coordinate,
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
)
from mitoolspro.google_utils.places.utils import (
    generate_unique_place_id,
    meters_to_degree,
)
from mitoolspro.utils.context_vars import ContextVar

global_requests_counter = ContextVar("GLOBAL_REQUESTS_COUNTER", default_value=0)
GOOGLE_PLACES_API_URL = "https://places.googleapis.com/v1/places:searchNearby"
RESTAURANT_TYPES = [
    "american_restaurant",
    "bakery",
    "bar",
    "barbecue_restaurant",
    "brazilian_restaurant",
    "breakfast_restaurant",
    "brunch_restaurant",
    "cafe",
    "chinese_restaurant",
    "coffee_shop",
    "fast_food_restaurant",
    "french_restaurant",
    "greek_restaurant",
    "hamburger_restaurant",
    "ice_cream_shop",
    "indian_restaurant",
    "indonesian_restaurant",
    "italian_restaurant",
    "japanese_restaurant",
    "korean_restaurant",
    "lebanese_restaurant",
    "meal_delivery",
    "meal_takeaway",
    "mediterranean_restaurant",
    "mexican_restaurant",
    "middle_eastern_restaurant",
    "pizza_restaurant",
    "ramen_restaurant",
    "restaurant",
    "sandwich_shop",
    "seafood_restaurant",
    "spanish_restaurant",
    "steak_house",
    "sushi_restaurant",
    "thai_restaurant",
    "turkish_restaurant",
    "vegan_restaurant",
    "vegetarian_restaurant",
    "vietnamese_restaurant",
]

FIELD_MASK = (
    "places.accessibilityOptions,places.addressComponents,places.adrFormatAddress,places.businessStatus,"
    + "places.displayName,places.formattedAddress,places.googleMapsUri,places.iconBackgroundColor,"
    + "places.iconMaskBaseUri,places.id,places.location,places.name,places.primaryType,places.primaryTypeDisplayName,places.plusCode,"
    + "places.shortFormattedAddress,places.subDestinations,places.types,places.utcOffsetMinutes,places.viewport,"
    + "places.currentOpeningHours,places.currentSecondaryOpeningHours,places.internationalPhoneNumber,places.nationalPhoneNumber,"
    + "places.priceLevel,places.rating,places.regularOpeningHours,places.regularSecondaryOpeningHours,places.userRatingCount,places.websiteUri"
)


class GooglePlacesClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        places_types: Optional[List[str]] = None,
        field_mask: Optional[str] = None,
    ):
        self.api_key = api_key
        self.places_types = places_types or RESTAURANT_TYPES
        self.field_mask = field_mask or FIELD_MASK

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-FieldMask": self.field_mask,
        }
        if self.api_key:
            headers["X-Goog-Api-Key"] = self.api_key
        return headers

    def search_nearby(
        self,
        center_point: Point,
        radius_in_meters: float,
        included_types: Optional[List[str]] = None,
        has_places: Optional[bool] = None,
    ) -> NewPlacesResponse:
        query_object = NewNearbySearchRequest(
            location=center_point,
            distance_in_meters=radius_in_meters,
            included_types=included_types or self.places_types,
        )

        query = query_object.json_query()
        headers = self._build_headers()

        if not self.api_key:
            return create_dummy_response(query, has_places)

        try:
            response = requests.post(
                GOOGLE_PLACES_API_URL,
                headers=headers,
                json=query,
                timeout=10,
            )
            response.raise_for_status()
            return NewPlacesResponse.model_validate(response.json())
        except requests.RequestException as e:
            raise RuntimeError(f"Google Places request failed: {e}")

    def _dummy_response(
        self, query: Dict[str, Any], has_places: Optional[bool] = None
    ) -> NewPlacesResponse:
        return create_dummy_response(query, has_places)

    def get_response_places(
        self,
        response_id: str,
        places: List[NewPlace],
    ) -> DataFrame:
        places = DataFrame([place.model_dump(mode="python") for place in places])
        places["circle"] = response_id
        return places

    def search_for_places(
        self,
        center_point: Point,
        radius_in_meters: float,
        response_id: str,
        included_types: Optional[List[str]] = None,
        has_places: bool = True,
    ) -> DataFrame:
        if not isinstance(center_point, Point):
            raise ArgumentTypeError("Invalid 'center_point' is not of type Point.")

        try:
            places_response = self.search_nearby(
                center_point=center_point,
                radius_in_meters=radius_in_meters,
                included_types=included_types,
                has_places=has_places,
            )
            places_df = self.get_response_places(response_id, places_response.places)
            return places_df
        except Exception as e:
            logger.error(
                "[search_for_places] Unrecoverable error: %s\n%s",
                e,
                traceback.format_exc(),
            )
            return None


def create_dummy_response(
    query: Dict[str, Any],
    has_places: bool = None,
) -> NewPlacesResponse:
    has_places = (
        random.choice([True, False, False]) if has_places is None else has_places
    )
    places = []
    if has_places:
        places_n = random.randint(1, 21)
        places = [create_dummy_place(query) for _ in range(places_n)]
    return NewPlacesResponse(places=places)


def create_dummy_place(query: Dict[str, Any]) -> NewPlace:
    latitude = query["locationRestriction"]["circle"]["center"]["latitude"]
    longitude = query["locationRestriction"]["circle"]["center"]["longitude"]
    radius = query["locationRestriction"]["circle"]["radius"]
    distance_in_deg = meters_to_degree(radius, latitude)
    random_types = random.sample(
        RESTAURANT_TYPES,
        random.randint(1, min(len(RESTAURANT_TYPES), random.randint(1, 5))),
    )
    unique_id = generate_unique_place_id()
    random_latitude = random.uniform(
        latitude - distance_in_deg, latitude + distance_in_deg
    )
    random_longitude = random.uniform(
        longitude - distance_in_deg, longitude + distance_in_deg
    )
    place_name = f"Place {unique_id}"

    return NewPlace(
        id=unique_id,
        name=place_name,
        types=random_types,
        formattedAddress=f"{unique_id} Some Address",
        shortFormattedAddress=f"{unique_id} Short Address",
        adrFormatAddress=f"{unique_id} ADR Format Address",
        addressComponents=[
            AddressComponent(
                longText="City",
                shortText="C",
                types=["locality"],
                languageCode="en",
            )
        ],
        plusCode=PlusCode(
            globalCode=f"87G8P27V+{unique_id[:4]}",
            compoundCode=f"P27V+{unique_id[:4]} City, Country",
        ),
        location=Coordinate(
            latitude=random_latitude,
            longitude=random_longitude,
        ),
        viewport=Viewport(
            low=Coordinate(
                latitude=random_latitude - 0.001,
                longitude=random_longitude - 0.001,
            ),
            high=Coordinate(
                latitude=random_latitude + 0.001,
                longitude=random_longitude + 0.001,
            ),
        ),
        googleMapsUri=f"https://maps.google.com/?q={random_latitude},{random_longitude}",
        websiteUri=f"https://example.com/{unique_id}",
        businessStatus="OPERATIONAL",
        rating=random.uniform(1.0, 5.0),
        userRatingCount=random.randint(1, 500),
        priceLevel=str(random.choice([1, 2, 3, 4, 5])),
        nationalPhoneNumber=f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        internationalPhoneNumber=f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        utcOffsetMinutes=random.choice(
            [
                -480,
                -420,
                -360,
                -300,
                -240,
                -180,
                -120,
                -60,
                0,
                60,
                120,
                180,
                240,
                300,
                360,
                420,
                480,
            ]
        ),
        iconMaskBaseUri="https://maps.gstatic.com/mapfiles/place_api/icons/v1/png_71/restaurant-71.png",
        iconBackgroundColor="#FF9E67",
        displayName=LocalizedText(
            text=place_name,
            languageCode="en",
        ),
        primaryType=random.choice(random_types),
        primaryTypeDisplayName=LocalizedText(
            text=random.choice(random_types),
            languageCode="en",
        ),
        currentOpeningHours=OpeningHours(
            openNow=random.choice([True, False]),
            periods=[
                Period(
                    open=TimePeriod(day=1, hour=9, minute=0),
                    close=TimePeriod(day=1, hour=17, minute=0),
                )
            ],
            weekdayDescriptions=["Monday: 9:00 AM – 5:00 PM"],
            nextOpenTime="2024-03-25T09:00:00Z",
        ),
        regularOpeningHours=OpeningHours(
            openNow=random.choice([True, False]),
            periods=[
                Period(
                    open=TimePeriod(day=1, hour=9, minute=0),
                    close=TimePeriod(day=1, hour=17, minute=0),
                )
            ],
            weekdayDescriptions=["Monday: 9:00 AM – 5:00 PM"],
            nextOpenTime="2024-03-25T09:00:00Z",
        ),
        accessibilityOptions=AccessibilityOptions(
            wheelchairAccessibleSeating=random.choice([True, False]),
            wheelchairAccessibleParking=random.choice([True, False]),
            wheelchairAccessibleEntrance=random.choice([True, False]),
            wheelchairAccessibleRestroom=random.choice([True, False]),
        ),
    )
