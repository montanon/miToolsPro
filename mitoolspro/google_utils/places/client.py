import random
import traceback
from typing import Any, Dict, List, Optional, Union

import requests
from pandas import DataFrame
from shapely.geometry import Point

from mitoolspro.exceptions import ArgumentTypeError
from mitoolspro.google_utils.places.models import (
    DummyResponse,
    NewNearbySearchRequest,
    NewPlace,
    NewPlacesResponse,
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


def create_dummy_response(
    query: Dict[str, Any],
    has_places: bool = None,
) -> DummyResponse:
    has_places = (
        random.choice([True, False, False]) if has_places is None else has_places
    )
    data = {}
    if has_places:
        places_n = random.randint(1, 21)
        data["places"] = [create_dummy_place(query) for _ in range(places_n)]
    return DummyResponse(data=data)


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
    ) -> Union[List[NewPlace], DummyResponse]:
        query_object = NewNearbySearchRequest(
            location=center_point,
            distance_in_meters=radius_in_meters,
            included_types=included_types or self.places_types,
        )

        query = query_object.json_query()
        headers = self._build_headers()

        if not self.api_key:
            dummy_response = self._dummy_response(query, has_places)
            return self._parse_response(dummy_response.json())

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
    ) -> DummyResponse:
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
            print(
                f"[search_for_places] Unrecoverable error: {e}\n{traceback.format_exc()}"
            )
            return None


def create_dummy_place(query: Dict[str, Any]) -> Dict[str, Any]:
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
    place_data = {
        "id": unique_id,
        "types": random_types,
        "location": {
            "latitude": random_latitude,
            "longitude": random_longitude,
        },
    }
    place_data.update(
        {
            "displayName": {"text": f"Name {unique_id}"},
            "primaryType": random.choice(random_types),
            "primaryTypeDisplayName": {"text": random.choice(random_types)},
            "formattedAddress": f"{unique_id} Some Address",
            "addressComponents": [
                {
                    "longText": "City",
                    "shortText": "C",
                    "types": ["locality"],
                    "languageCode": "en",
                }
            ],
            "googleMapsUri": f"https://maps.google.com/?q={random_latitude},{random_longitude}",
            "priceLevel": str(random.choice([1, 2, 3, 4, 5])),
            "rating": random.uniform(1.0, 5.0),
            "userRatingCount": random.randint(1, 500),
        }
    )
    return place_data
