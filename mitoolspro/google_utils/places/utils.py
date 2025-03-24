import math
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, List, NewType, Optional, Union

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.google_utils.places.models import intersection_condition_factory

CircleType = NewType("CircleType", Polygon)


def meters_to_degree(distance_in_meters: float, reference_latitude: float) -> float:
    if not isinstance(distance_in_meters, (int, float)) or distance_in_meters < 0:
        raise ArgumentValueError("Invalid Distance, must be a positive number")
    if reference_latitude >= 90 or reference_latitude <= -90:
        raise ArgumentValueError("Invalid Latitude, must be between -90° and 90°")
    meters_per_degree_latitude = 111_132.95
    meters_per_degree_longitude = 111_132.95 * math.cos(
        math.radians(reference_latitude)
    )
    lat_degrees = distance_in_meters / meters_per_degree_latitude
    lon_degrees = distance_in_meters / meters_per_degree_longitude
    return max(lat_degrees, lon_degrees)


def calculate_degree_steps(
    meter_radiuses: List[float], step_in_degrees: float = 0.00375
) -> List[float]:
    if not meter_radiuses:
        raise ArgumentValueError("Radius values must be a non-empty list of numbers.")
    if any(r <= 0 for r in meter_radiuses):
        raise ArgumentValueError("All radius values must be positive.")
    degree_steps = [step_in_degrees]  # Start with the initial step
    for i in range(1, len(meter_radiuses)):
        step = degree_steps[-1] * (meter_radiuses[i] / meter_radiuses[i - 1])
        degree_steps.append(step)
    return degree_steps


def sample_polygon_with_circles(
    polygon: Polygon,
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: str = "center",
) -> List[CircleType]:
    if not isinstance(polygon, Polygon):
        raise ArgumentTypeError("Invalid 'polygon' is not of type Polygon.")
    if not polygon.is_valid:
        raise ArgumentValueError("Invalid Polygon")
    if polygon.is_empty:
        raise ArgumentValueError("Empty Polygon")
    if step_in_degrees <= 0:
        raise ArgumentValueError("Invalid Step, must be a positive number")
    condition = intersection_condition_factory(condition_rule)
    minx, miny, maxx, maxy = polygon.bounds
    latitudes = np.arange(miny, maxy, step_in_degrees)
    longitudes = np.arange(minx, maxx, step_in_degrees)
    circles = []
    for lat, lon in product(latitudes, longitudes):
        deg_radius = meters_to_degree(
            distance_in_meters=radius_in_meters, reference_latitude=lat
        )
        circle = Point(lon, lat).buffer(deg_radius)
        if condition.check(polygon=polygon, circle=circle):
            circles.append(circle)
    return circles


def sample_polygons_with_circles(
    polygons: Union[Iterable[Polygon], Polygon],
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: Optional[str] = "center",
) -> List[CircleType]:
    if not isinstance(polygons, (Polygon, MultiPolygon, Iterable)):
        raise ArgumentTypeError(
            "Invalid 'polygons' is not of type Polygon, MultiPolygon or an iterable of them."
        )
    elif isinstance(polygons, Polygon):
        polygons = [polygons]
    elif isinstance(polygons, MultiPolygon):
        polygons = list(polygons.geoms)
    elif not all(isinstance(polygon, (Polygon, MultiPolygon)) for polygon in polygons):
        raise ArgumentTypeError(
            "Invalid 'polygons' is not of type Polygon or MultiPolygon."
        )
    circles = []
    for polygon in polygons:
        circles.extend(
            sample_polygon_with_circles(
                polygon=polygon,
                radius_in_meters=radius_in_meters,
                step_in_degrees=step_in_degrees,
                condition_rule=condition_rule,
            )
        )
    return circles


def get_circles_search(
    circles_path: Path,
    polygon: Polygon,
    radius_in_meters: float,
    step_in_degrees: float,
    condition_rule: str = "center",
    recalculate: bool = False,
) -> GeoDataFrame:
    if not circles_path or not isinstance(circles_path, Path):
        raise ArgumentValueError("`circles_path` must be a valid Path object.")
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise ArgumentTypeError(
            "`polygon` must be a Shapely Polygon or MultiPolygon object."
        )
    if recalculate or not circles_path.exists():
        circles = sample_polygons_with_circles(
            polygons=polygon,
            radius_in_meters=radius_in_meters,
            step_in_degrees=step_in_degrees,
            condition_rule=condition_rule,
        )
        circles = (
            GeoDataFrame(geometry=circles).reset_index(drop=True).assign(searched=False)
        )
        circles.to_file(circles_path, driver="GeoJSON")
    else:
        circles = gpd.read_file(circles_path)

    return circles


def create_subsampled_circles(
    large_circle_center: Point,
    large_radius: float,
    small_radius: float,
    radial_samples: int,
    factor: float = 1.0,
) -> List[Polygon]:
    if not isinstance(large_circle_center, Point):
        raise ArgumentTypeError("Invalid 'large_circle_center' is not of type Point.")
    if large_radius <= 0 or small_radius <= 0:
        raise ArgumentValueError("Radius values must be positive.")
    if radial_samples <= 0:
        raise ArgumentValueError("radial_samples must be a positive integer.")
    large_radius_deg = meters_to_degree(large_radius, large_circle_center.y)
    small_radius_deg = meters_to_degree(small_radius, large_circle_center.y)
    large_circle = large_circle_center.buffer(large_radius_deg)
    subsampled_circles = [large_circle_center.buffer(small_radius_deg)]
    angle_step = 2 * np.pi / radial_samples
    for i in range(radial_samples):
        angle = i * angle_step
        dx = factor * small_radius_deg * np.cos(angle)
        dy = factor * small_radius_deg * np.sin(angle)
        new_center = Point(large_circle_center.x + dx, large_circle_center.y + dy)
        if large_circle.contains(new_center):
            subsampled_circles.append(new_center.buffer(small_radius_deg))
    return subsampled_circles


def generate_unique_place_id():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


def convert_shp_to_geojson(shp_path: str, output_path: str) -> None:
    gdf = gpd.read_file(shp_path)
    gdf.to_file(output_path, driver="GeoJSON")
