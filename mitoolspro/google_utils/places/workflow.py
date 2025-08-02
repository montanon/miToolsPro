from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from pandas import DataFrame
from shapely.geometry import Polygon
import logging

logger = logging.getLogger(__name__)

from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.models import CityGeojson
from mitoolspro.google_utils.places.search import places_search_step
from mitoolspro.google_utils.places.utils import calculate_degree_steps


class PlacesSamplingWorkflow:
    def __init__(
        self,
        city_name: str,
        geojson_path: Path,
        project_folder: Path,
        plots_folder: Optional[Path] = None,
        meter_radiuses: Optional[List[float]] = None,
        step_in_degrees: float = 0.00375,
        client: Optional[GooglePlacesClient] = None,
        included_types: Optional[List[str]] = None,
        threshold: int = 20,
        show: bool = False,
        recalculate: bool = False,
    ):
        self.city_name = city_name
        self.geojson_path = Path(geojson_path)
        self.project_folder = Path(project_folder)
        self.plots_folder = plots_folder or self.project_folder / "plots"
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.plots_folder.mkdir(parents=True, exist_ok=True)

        self.meter_radiuses = meter_radiuses or [250, 100, 50, 25, 12.5, 5, 2.5, 1]
        self.step_in_degrees = step_in_degrees
        self.client = client or GooglePlacesClient(api_key=None)
        self.included_types = included_types
        self.threshold = threshold
        self.show = show
        self.recalculate = recalculate

        self.city = CityGeojson(self.geojson_path, city_name)
        self.radius_step_pairs = list(
            zip(
                self.meter_radiuses,
                calculate_degree_steps(self.meter_radiuses, self.step_in_degrees),
            )
        )

        self.all_places = pd.DataFrame(
            columns=["circle", *list(self.client.places_types)]
        )
        self.total_sampled_circles = 0
        self.area_polygon = self.city.merged_polygon

    def _plot_city_shapes(self):
        city_plot = self.plots_folder / f"{self.city.name}_polygon_plot.png"
        wards_plot = self.plots_folder / f"{self.city.name}_wards_polygons_plot.png"
        if self.show:
            ax1 = self.city.plot_polygons()
            if not wards_plot.exists() or self.recalculate:
                ax1.get_figure().savefig(wards_plot, dpi=500)
            plt.show()
            ax2 = self.city.plot_unary_polygon()
            if not city_plot.exists() or self.recalculate:
                ax2.get_figure().savefig(city_plot, dpi=500)
            plt.show()

    def run(self):
        self._plot_city_shapes()
        area_polygon = self.area_polygon
        for i, (radius, step) in enumerate(self.radius_step_pairs):
            tag = f"Step-{i + 1}_{self.city_name}"
            logger.info("\n>>> Sampling: %s", tag)

            found_places, circles, area_polygon, saturated_circles = places_search_step(
                project_folder=self.project_folder,
                plots_folder=self.plots_folder,
                tag=tag,
                polygon=area_polygon,
                radius_in_meters=radius,
                step_in_degrees=step,
                client=self.client,
                included_types=self.included_types,
                recalculate=self.recalculate,
                show=self.show,
                threshold=self.threshold,
            )

            self.total_sampled_circles += circles.shape[0]

            logger.info(
                "→ Found Places: %s | Sampled Circles: %s | Saturated Circles: %s",
                found_places.shape[0],
                circles.shape[0],
                saturated_circles.shape[0],
            )

            self.all_places = pd.concat(
                [self.all_places, found_places], axis=0, ignore_index=True
            )

    def save_results(self):
        def _drop_columns(df: DataFrame, cols: List[str]) -> DataFrame:
            return df.drop(columns=[c for c in cols if c in df.columns])

        all_path = self.project_folder / f"{self.city.name}_all_found_places"
        uniq_path = self.project_folder / f"{self.city.name}_unique_found_places"

        all_cleaned = _drop_columns(
            self.all_places, ["iconMaskBaseUri", "googleMapsUri", "websiteUri"]
        ).reset_index(drop=True)

        all_cleaned.to_parquet(f"{all_path}.parquet")
        all_cleaned.to_excel(f"{all_path}.xlsx", index=False)

        unique_places = all_cleaned.drop_duplicates(subset=["id"]).reset_index(
            drop=True
        )
        unique_places.to_parquet(f"{uniq_path}.parquet")
        unique_places.to_excel(f"{uniq_path}.xlsx", index=False)

        logger.info("\n✔ Saved %s places (%s unique)", len(all_cleaned), len(unique_places))


if __name__ == "__main__":
    import os

    workflow = PlacesSamplingWorkflow(
        city_name="tokyo",
        geojson_path="/Users/sebastian/Desktop/MontagnaInc/Research/Cities_Restaurants/translated_tokyo_wards.geojson",
        project_folder=Path("Here"),
        client=GooglePlacesClient(api_key=os.getenv("GOOGLE_PLACES_API_KEY")),
        meter_radiuses=[500],
        step_in_degrees=0.0066,
        show=True,
        recalculate=True,
    )
    workflow.run()
