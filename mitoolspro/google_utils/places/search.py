from typing import List, Optional, Tuple, Union

from pandas import DataFrame
from shapely.geometry import Point, Polygon

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.models import DummyResponse, NewPlace


def get_response_places(
    response_id: str,
    places: List[NewPlace],
) -> DataFrame:
    places_series = []
    for place in places:
        places_series = place.to_series()
        places_series["circle"] = response_id
        places_series.append(places_series)
    return DataFrame(places_series)


def search_for_places(
    center_point: Point,
    radius_in_meters: float,
    response_id: str,
    client: GooglePlacesClient,
    included_types: Optional[List[str]] = None,
    has_places: bool = True,
) -> Tuple[bool, Optional[DataFrame]]:
    if not isinstance(center_point, Point):
        raise ArgumentTypeError("Invalid 'center_point' is not of type Point.")

    try:
        places = client.search_nearby(
            center_point=center_point,
            radius_in_meters=radius_in_meters,
            included_types=included_types,
            has_places=has_places,
        )
        places_df = get_response_places(response_id, places)
        return places_df
    except Exception as e:
        print(f"[search_for_places] Unrecoverable error: {e}")
        return None
