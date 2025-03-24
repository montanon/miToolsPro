from pathlib import Path
from typing import Optional, Union

from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import MultiPolygon, Polygon

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.google_utils.places.plots import (
    plot_saturated_area,
    plot_saturated_circles,
)


def filter_saturated_circles(
    found_places: DataFrame,
    circles: GeoDataFrame,
    threshold: int,
) -> GeoDataFrame:
    if circles.empty:
        raise ArgumentValueError("'circles' cannot be empty.")
    if threshold < 0:
        raise ArgumentValueError("'threshold' must be a positive integer or 0.")
    places_by_circle = (
        found_places.groupby("circle")["id"].nunique().sort_values(ascending=False)
    )
    saturated_circle_indices = places_by_circle[places_by_circle >= threshold].index
    try:
        saturated_circles = circles.loc[saturated_circle_indices]
        return saturated_circles
    except KeyError as e:
        raise ArgumentValueError(
            f"Invalid 'circles' and 'found_places' Circles indexes: {e}"
        )


def compute_saturated_circles(
    polygon: Polygon,
    found_places: DataFrame,
    circles: GeoDataFrame,
    threshold: int,
    show: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> GeoDataFrame:
    saturated_circles = filter_saturated_circles(found_places, circles, threshold)
    points = found_places.loc[
        found_places["circle"].isin(saturated_circles.index),
        ["longitude", "latitude"],
    ].values.tolist()

    plot_saturated_circles(
        polygon=polygon,
        circles=saturated_circles.geometry.tolist(),
        points=points,
        output_file_path=output_path,
        show=show,
    )
    return saturated_circles


def compute_saturated_area(
    polygon: Polygon,
    saturated_circles: GeoDataFrame,
    show: bool = False,
    output_path: Union[str, Path] = None,
) -> Union[Polygon, MultiPolygon]:
    saturated_area = saturated_circles.geometry.unary_union
    plot_saturated_area(polygon, saturated_area, show=show, output_path=output_path)
    return saturated_area
