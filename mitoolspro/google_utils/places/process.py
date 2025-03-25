from pathlib import Path
from typing import List, Optional

import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Polygon
from tqdm import tqdm

from mitoolspro.exceptions import ArgumentTypeError
from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.models import NewPlace
from mitoolspro.utils.context_vars import ContextVar

global_requests_counter = ContextVar("GLOBAL_REQUESTS_COUNTER", default_value=0)
global_requests_counter_limit = ContextVar(
    "GLOBAL_REQUESTS_COUNTER_LIMIT", default_value=200
)


def process_circles(
    circles: GeoDataFrame,
    radius_in_meters: float,
    file_path: Path,
    circles_path: Path,
    client: GooglePlacesClient,
    included_types: Optional[List[str]] = None,
    has_places: bool = True,
    recalculate: bool = False,
) -> DataFrame:
    if file_path.exists() and not recalculate:
        found_places = pd.read_parquet(file_path)
    else:
        found_places = DataFrame(
            columns=["circle", *list(NewPlace.__annotations__.keys())]
        )

    if circles.empty:
        return found_places

    if should_process_circles(circles, recalculate):
        with tqdm(total=len(circles), desc="Processing circles") as pbar:
            for response_id, circle in circles[~circles["searched"]].iterrows():
                found_places = process_single_circle(
                    response_id=response_id,
                    circle=circle["geometry"],
                    radius_in_meters=radius_in_meters,
                    found_places=found_places,
                    circles=circles,
                    file_path=file_path,
                    circles_path=circles_path,
                    pbar=pbar,
                    client=client,
                    included_types=included_types,
                    has_places=has_places,
                )
    else:
        found_places = pd.read_parquet(file_path)

    return found_places


def process_single_circle(
    response_id: int,
    circle: Polygon,
    radius_in_meters: float,
    found_places: DataFrame,
    circles: GeoDataFrame,
    file_path: Path,
    circles_path: Path,
    pbar: tqdm,
    client: GooglePlacesClient,
    included_types: Optional[List[str]] = None,
    has_places: bool = True,
) -> DataFrame:
    if not isinstance(circle, Polygon):
        raise ArgumentTypeError("Invalid 'circle' geometry")

    if not should_do_search():
        return found_places

    places_df = client.search_for_places(
        center_point=circle.centroid,
        radius_in_meters=radius_in_meters,
        response_id=response_id,
        included_types=included_types,
        has_places=has_places,
    )

    if places_df is not None and not places_df.empty:
        found_places = pd.concat([found_places, places_df], axis=0, ignore_index=True)

    circles.loc[response_id, "searched"] = True
    global_requests_counter.value += 1

    if should_save_state(response_id, circles.shape[0]):
        found_places.to_parquet(file_path)
        circles.to_file(circles_path, driver="GeoJSON")

    update_progress_bar(pbar, circles, found_places)
    return found_places


def should_process_circles(circles: GeoDataFrame, recalculate: bool) -> bool:
    return (~circles["searched"]).any() or recalculate


def should_save_state(
    response_id: int, total_circles: int, n_amount: int = 200
) -> bool:
    return (
        (response_id % n_amount == 0)
        or (response_id == total_circles - 1)
        or (global_requests_counter.value == global_requests_counter_limit.value - 1)
    )


def should_do_search() -> None:
    return global_requests_counter.value < global_requests_counter_limit.value


def update_progress_bar(
    pbar: tqdm, circles: GeoDataFrame, found_places: DataFrame
) -> None:
    remaining_circles = circles["searched"].value_counts().get(False, 0)
    searched_circles = circles["searched"].sum()
    found_places_count = found_places["id"].nunique()
    pbar.update()
    pbar.set_postfix(
        {
            "Remaining Circles": remaining_circles,
            "Found Places": found_places_count,
            "Searched Circles": searched_circles,
        }
    )
