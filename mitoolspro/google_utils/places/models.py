import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Protocol

import geopandas as gpd
import seaborn as sns
from geopandas import GeoDataFrame, GeoSeries
from matplotlib.pyplot import Axes
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator
from shapely import Point, Polygon
from shapely.ops import unary_union

from mitoolspro.exceptions import (
    ArgumentValueError,
)

CircleType = NewType("CircleType", Point)


class Coordinate(BaseModel):
    latitude: float
    longitude: float

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ArgumentValueError("Latitude must be between -90° and 90°")
        return v

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ArgumentValueError("Longitude must be between -180° and 180°")
        return v


class Viewport(BaseModel):
    low: Coordinate
    high: Coordinate

    @field_validator("high")
    @classmethod
    def validate_coordinates(cls, v: Coordinate, info: ValidationInfo) -> Coordinate:
        if info.data and "low" in info.data:
            low = info.data["low"]
            if v.latitude <= low.latitude or v.longitude <= low.longitude:
                raise ArgumentValueError(
                    "High coordinate must be greater than low coordinate"
                )
        return v


class PlusCode(BaseModel):
    globalCode: Optional[str] = None
    compoundCode: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.globalCode is None and self.compoundCode is None:
            raise ValidationError(
                "At least one of globalCode or compoundCode must be provided"
            )


class AddressComponent(BaseModel):
    longText: str
    shortText: str
    types: List[str]
    languageCode: str


class DateStamp(BaseModel):
    year: int
    month: int
    day: int

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: int) -> int:
        if not 1 <= v <= 12:
            raise ValueError("Month must be between 1 and 12")
        return v

    @field_validator("day")
    @classmethod
    def validate_day(cls, v: int) -> int:
        if not 1 <= v <= 31:
            raise ValueError("Day must be between 1 and 31")
        return v


class TimePeriod(BaseModel):
    day: int
    hour: int
    minute: int
    date: Optional[DateStamp] = None

    @field_validator("hour")
    @classmethod
    def validate_hour(cls, v: int) -> int:
        if not 0 <= v < 24:
            raise ArgumentValueError("Hour must be between 0 and 23")
        return v

    @field_validator("minute")
    @classmethod
    def validate_minute(cls, v: int) -> int:
        if not 0 <= v < 60:
            raise ArgumentValueError("Minute must be between 0 and 59")
        return v


class Period(BaseModel):
    open: TimePeriod = None
    close: TimePeriod = None


class OpeningHours(BaseModel):
    openNow: Optional[bool] = None
    periods: Optional[List[Period]] = None
    weekdayDescriptions: Optional[List[str]] = None
    nextOpenTime: Optional[str] = None


class AccessibilityOptions(BaseModel):
    wheelchairAccessibleSeating: Optional[bool] = None
    wheelchairAccessibleParking: Optional[bool] = None
    wheelchairAccessibleEntrance: Optional[bool] = None
    wheelchairAccessibleRestroom: Optional[bool] = None


class LocalizedText(BaseModel):
    text: Optional[str] = None
    languageCode: Optional[str] = None


class NewPlace(BaseModel):
    id: str
    name: Optional[str] = None
    types: List[str]
    formattedAddress: Optional[str] = None
    shortFormattedAddress: Optional[str] = None
    adrFormatAddress: Optional[str] = None
    addressComponents: Optional[List[AddressComponent]] = None

    plusCode: Optional[PlusCode] = None
    location: Coordinate
    viewport: Optional[Viewport] = None

    googleMapsUri: Optional[str] = None
    websiteUri: Optional[str] = None

    businessStatus: Optional[str] = None
    rating: Optional[float] = None
    userRatingCount: Optional[int] = None
    priceLevel: Optional[str] = None

    nationalPhoneNumber: Optional[str] = None
    internationalPhoneNumber: Optional[str] = None

    utcOffsetMinutes: Optional[int] = None

    iconMaskBaseUri: Optional[str] = None
    iconBackgroundColor: Optional[str] = None

    displayName: Optional[LocalizedText] = None
    primaryType: Optional[str] = None
    primaryTypeDisplayName: Optional[LocalizedText] = None

    currentOpeningHours: Optional[OpeningHours] = None
    regularOpeningHours: Optional[OpeningHours] = None

    accessibilityOptions: Optional[AccessibilityOptions] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        self.latitude = self.location.latitude
        self.longitude = self.location.longitude
        self.place_name = self.displayName.text


class NewPlacesResponse(BaseModel):
    places: List[NewPlace] = Field(default_factory=list)


class DummyResponse:
    def __init__(self, data: Dict[str, Any] = None, status_code: int = 200):
        self.data = data or {}
        self.status_code = status_code
        self.reason = "OK" if status_code == 200 else "Error"

    def json(self) -> Dict[str, Any]:
        return self.data


class NewNearbySearchRequest(BaseModel):
    location: Point
    distance_in_meters: float = Field(..., gt=0)
    max_result_count: int = Field(default=20, gt=0)
    included_types: List[str] = Field(default_factory=list)
    language_code: str = Field(default="en", min_length=2, max_length=2)

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: Point) -> Point:
        if not isinstance(v, Point):
            raise TypeError(f"Expected a Point, got {type(v).__name__}")
        return v

    @property
    def location_restriction(self) -> Dict[str, Dict[str, Any]]:
        center = self.location.centroid
        return {
            "circle": {
                "center": {
                    "latitude": center.y,
                    "longitude": center.x,
                },
                "radius": self.distance_in_meters,
            }
        }

    def json_query(self) -> Dict[str, Any]:
        return {
            "includedTypes": self.included_types,
            "maxResultCount": self.max_result_count,
            "locationRestriction": self.location_restriction,
            "languageCode": self.language_code,
        }

    model_config = {"arbitrary_types_allowed": True}


class CityGeojson:
    TOKYO_WARDS_NAMES = [
        "Chiyoda Ward",
        "Koto Ward",
        "Nakano",
        "Meguro",
        "Shinagawa Ward",
        "Ota-ku",
        "Setagaya",
        "Suginami",
        "Nerima Ward",
        "Itabashi Ward",
        "Adachi Ward",
        "Katsushika",
        "Edogawa Ward",
        "Sumida Ward",
        "Chuo-ku",
        "Minato-ku",
        "North Ward",
        "Toshima ward",
        "Shibuya Ward",
        "Arakawa",
        "Bunkyo Ward",
        "Shinjuku ward",
        "Taito",
    ]

    def __init__(self, geojson_path: PathLike, name: str):
        self.geojson_path = self._validate_path(geojson_path)
        self.data = self._load_geojson(self.geojson_path)
        self.name = name
        self.plots_width = 14
        self.plots_aspect_ratio = 16.0 / 9.0
        self.plots_height = self.plots_width / self.plots_aspect_ratio
        self.polygons = self._process_polygons()
        self.merged_polygon = self.polygons.unary_union
        self.bounds = self.polygons.bounds.iloc[0].values

    @staticmethod
    def _validate_path(geojson_path: PathLike) -> Path:
        try:
            path = Path(geojson_path).resolve(strict=True)
        except Exception as e:
            raise ArgumentValueError(f"Invalid GeoJSON path: {geojson_path}. {e}")
        return path

    @staticmethod
    def _load_geojson(geojson_path: Path) -> GeoDataFrame:
        try:
            return gpd.read_file(geojson_path)
        except Exception as e:
            raise ArgumentValueError(
                f"Failed to load GeoJSON file: {geojson_path}. Error: {e}"
            )

    def _process_polygons(self) -> GeoSeries:
        if self.geojson_path.name == "translated_tokyo_wards.geojson":
            return self._merge_tokyo_wards()
        return self.data["geometry"]

    def _merge_tokyo_wards(self) -> GeoSeries:
        polygons = [
            unary_union(self.data.loc[self.data["Wards"] == ward, "geometry"])
            for ward in self.TOKYO_WARDS_NAMES
        ]
        return GeoSeries(polygons).explode(index_parts=True).reset_index(drop=True)

    def plot_unary_polygon(self) -> Axes:
        return self._plot_geoseries(
            GeoSeries(self.merged_polygon), f"{self.name.title()} Polygon"
        )

    def plot_polygons(self) -> Axes:
        return self._plot_geoseries(
            GeoSeries(self.polygons), f"{self.name.title()} Wards Polygons"
        )

    def _plot_geoseries(self, geoseries: GeoSeries, title: str) -> Axes:
        ax = geoseries.plot(
            facecolor="none",
            edgecolor=sns.color_palette("Paired")[0],
            figsize=(self.plots_width, self.plots_height),
        )
        ax.set_ylabel("Latitude")
        ax.set_xlabel("Longitude")
        ax.set_title(title)
        return ax


class ConditionProtocol(Protocol):
    def check(self, polygon: Polygon, circle: CircleType) -> bool: ...


class CircleInsidePolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return circle.within(polygon)


class CircleCenterInsidePolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return polygon.contains(circle.centroid)


class CircleIntersectsPolygon:
    def check(self, polygon: Polygon, circle: CircleType) -> bool:
        return polygon.intersects(circle)


def intersection_condition_factory(condition_type: str) -> ConditionProtocol:
    if condition_type == "circle":
        return CircleInsidePolygon()
    elif condition_type == "center":
        return CircleCenterInsidePolygon()
    elif condition_type == "intersection":
        return CircleIntersectsPolygon()
    else:
        raise ArgumentValueError(f"Unknown condition type: {condition_type}")
