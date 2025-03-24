import random
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame
from shapely.geometry import Polygon

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.places_old import (
    _plot_polygon_with_circles,
    _plot_polygon_with_circles_and_points,
)
from mitoolspro.google_utils.places.process import process_circles
from mitoolspro.google_utils.places.utils import get_circles_search, meters_to_degree


def search_places_in_polygon(
    root_folder: PathLike,
    plot_folder: PathLike,
    tag: str,
    polygon: Polygon,
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: str,
    client: GooglePlacesClient,
    included_types: Optional[List[str]] = None,
    recalculate: bool = False,
    has_places: bool = True,
    show: bool = False,
) -> Tuple[GeoDataFrame, DataFrame]:
    if not isinstance(root_folder, Path) or not root_folder.exists():
        raise ArgumentValueError("`root_folder` must be a valid Path object.")
    if not isinstance(plot_folder, Path) or not plot_folder.exists():
        raise ArgumentValueError("`plot_folder` must be a valid Path object.")
    if not isinstance(polygon, Polygon):
        raise ArgumentTypeError(
            f"Invalid 'polygon' of type {type(polygon)} is not of type Polygon."
        )

    circles_path = _generate_file_path(
        root_folder, tag, radius_in_meters, step_in_degrees, "circles.geojson"
    )
    places_path = _generate_file_path(
        root_folder, tag, radius_in_meters, step_in_degrees, "places.parquet"
    )
    plot_paths = _generate_plot_paths(plot_folder, tag)

    circles = get_circles_search(
        circles_path=circles_path,
        polygon=polygon,
        radius_in_meters=radius_in_meters,
        step_in_degrees=step_in_degrees,
        condition_rule=condition_rule,
        recalculate=recalculate,
    )

    if show or recalculate:
        _generate_sampling_plots(
            polygon, circles.geometry, plot_paths, radius_in_meters, show
        )

    found_places = process_circles(
        circles=circles,
        radius_in_meters=radius_in_meters,
        file_path=places_path,
        circles_path=circles_path,
        included_types=included_types,
        recalculate=recalculate,
        has_places=has_places,
        client=client,
    )

    if show or recalculate:
        _generate_results_plots(
            polygon, circles.geometry, found_places, plot_paths, radius_in_meters, show
        )

    return circles, found_places


def _generate_sampling_plots(
    polygon: Polygon,
    circles: GeoSeries,
    plot_paths: Dict[str, Path],
    radius_in_meters: float,
    show: bool,
) -> None:
    if not isinstance(polygon, Polygon):
        raise ArgumentTypeError("Invalid 'polygon' is not of type Polygon.")
    if not isinstance(circles, GeoSeries):
        raise ArgumentTypeError("Invalid 'circles' is not of type GeoSeries.")
    _plot_polygon_with_circles(polygon, circles, plot_paths["circles"], show)

    random_circle = random.choice(circles.geometry.tolist())
    zoom_level = 5 * meters_to_degree(radius_in_meters, random_circle.centroid.y)
    _plot_polygon_with_circles(
        polygon, circles, plot_paths["circles_zoom"], show, random_circle, zoom_level
    )


def _generate_results_plots(
    polygon: GeoDataFrame,
    circles: GeoDataFrame,
    found_places: GeoDataFrame,
    plot_paths: Dict[str, Path],
    radius_in_meters: float,
    show: bool,
) -> None:
    points = found_places[["longitude", "latitude"]].values.tolist()
    _plot_polygon_with_circles_and_points(
        polygon, circles, points, plot_paths["places"], show
    )

    random_circle = random.choice(circles.geometry.tolist())
    zoom_level = 5 * meters_to_degree(radius_in_meters, random_circle.centroid.y)
    _plot_polygon_with_circles_and_points(
        polygon,
        circles,
        points,
        plot_paths["places_zoom"],
        show,
        random_circle,
        zoom_level,
    )


def _generate_plot_paths(plot_folder: Path, tag: str) -> Dict[str, Path]:
    return {
        "circles": plot_folder / f"{tag}_polygon_with_circles_plot.png",
        "circles_zoom": plot_folder / f"{tag}_polygon_with_circles_zoom_plot.png",
        "places": plot_folder / f"{tag}_polygon_with_circles_and_places_plot.png",
        "places_zoom": plot_folder
        / f"{tag}_polygon_with_circles_and_places_zoom_plot.png",
    }


def _generate_file_path(
    folder: PathLike, tag: str, radius: float, step: float, suffix: str
) -> Path:
    return Path(folder) / f"{tag}_{radius}_radius_{step}_step_{suffix}"
