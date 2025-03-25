import unittest
from pathlib import Path
from unittest import TestCase

import pandas as pd
from shapely.geometry import Polygon

from mitoolspro.google_utils.places.client import GooglePlacesClient
from mitoolspro.google_utils.places.workflow import PlacesSamplingWorkflow


class TestPlacesSamplingWorkflow(TestCase):
    def setUp(self):
        self.temp_dir = Path("temp_test_dir")
        self.temp_dir.mkdir(exist_ok=True)
        self.geojson_path = self.temp_dir / "test_city.geojson"
        self.project_folder = self.temp_dir / "project"
        self.plots_folder = self.temp_dir / "plots"
        self.client = GooglePlacesClient(api_key="test_key")
        self.client.places_types = ["restaurant", "cafe", "bar"]

        # Create a real GeoJSON file with a simple polygon
        mock_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "test_city"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]],
                    },
                }
            ],
        }
        import json

        with open(self.geojson_path, "w") as f:
            json.dump(mock_geojson, f)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_initialization_with_default_values(self):
        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
        )

        self.assertEqual(workflow.city_name, "test_city")
        self.assertEqual(workflow.geojson_path, self.geojson_path)
        self.assertEqual(workflow.project_folder, self.project_folder)
        self.assertEqual(workflow.plots_folder, self.project_folder / "plots")
        self.assertEqual(workflow.meter_radiuses, [250, 100, 50, 25, 12.5, 5, 2.5, 1])
        self.assertEqual(workflow.step_in_degrees, 0.00375)
        self.assertEqual(workflow.threshold, 20)
        self.assertFalse(workflow.show)
        self.assertFalse(workflow.recalculate)
        self.assertEqual(workflow.total_sampled_circles, 0)
        self.assertTrue(workflow.all_places.empty)

    def test_initialization_with_custom_values(self):
        custom_radiuses = [500, 250]
        custom_step = 0.005
        custom_threshold = 30
        custom_types = ["restaurant", "cafe"]

        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
            plots_folder=self.plots_folder,
            meter_radiuses=custom_radiuses,
            step_in_degrees=custom_step,
            client=self.client,
            included_types=custom_types,
            threshold=custom_threshold,
            show=True,
            recalculate=True,
        )

        self.assertEqual(workflow.meter_radiuses, custom_radiuses)
        self.assertEqual(workflow.step_in_degrees, custom_step)
        self.assertEqual(workflow.client, self.client)
        self.assertEqual(workflow.included_types, custom_types)
        self.assertEqual(workflow.threshold, custom_threshold)
        self.assertTrue(workflow.show)
        self.assertTrue(workflow.recalculate)

    def test_save_results(self):
        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
        )

        test_data = {
            "circle": [1, 2],
            "id": ["place1", "place2"],
            "restaurant": [1, 0],
            "cafe": [0, 1],
            "bar": [0, 0],
            "iconMaskBaseUri": ["uri1", "uri2"],
            "googleMapsUri": ["maps1", "maps2"],
            "websiteUri": ["web1", "web2"],
        }

        workflow.all_places = pd.DataFrame(test_data)
        workflow.save_results()

        all_path = self.project_folder / "test_city_all_found_places"
        uniq_path = self.project_folder / "test_city_unique_found_places"

        self.assertTrue(all_path.with_suffix(".parquet").exists())
        self.assertTrue(all_path.with_suffix(".xlsx").exists())
        self.assertTrue(uniq_path.with_suffix(".parquet").exists())
        self.assertTrue(uniq_path.with_suffix(".xlsx").exists())

        # Verify the saved data
        saved_df = pd.read_parquet(all_path.with_suffix(".parquet"))
        self.assertEqual(len(saved_df), 2)
        self.assertNotIn("iconMaskBaseUri", saved_df.columns)
        self.assertNotIn("googleMapsUri", saved_df.columns)
        self.assertNotIn("websiteUri", saved_df.columns)

        unique_df = pd.read_parquet(uniq_path.with_suffix(".parquet"))
        self.assertEqual(len(unique_df), 2)

    def test_radius_step_pairs_calculation(self):
        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
            meter_radiuses=[500, 250],
            step_in_degrees=0.005,
        )

        self.assertEqual(len(workflow.radius_step_pairs), 2)
        self.assertEqual(workflow.radius_step_pairs[0][0], 500)
        self.assertEqual(workflow.radius_step_pairs[1][0], 250)

    def test_run_with_empty_results(self):
        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
            meter_radiuses=[500],
        )

        # Override the places_search_step method to return empty results
        def empty_search(*args, **kwargs):
            return pd.DataFrame(), pd.DataFrame(), workflow.area_polygon, pd.DataFrame()

        # Monkey patch the method at the module level
        import mitoolspro.google_utils.places.workflow as workflow_module

        original_search = workflow_module.places_search_step
        workflow_module.places_search_step = empty_search

        try:
            workflow.run()

            self.assertEqual(workflow.total_sampled_circles, 0)
            self.assertTrue(workflow.all_places.empty)
        finally:
            # Restore the original method
            workflow_module.places_search_step = original_search

    def test_run_with_sample_results(self):
        workflow = PlacesSamplingWorkflow(
            city_name="test_city",
            geojson_path=self.geojson_path,
            project_folder=self.project_folder,
            meter_radiuses=[500],
        )

        # Override the places_search_step method to return sample results
        def sample_search(*args, **kwargs):
            return (
                pd.DataFrame(
                    {"id": ["place1"], "restaurant": [1], "cafe": [0], "bar": [0]}
                ),
                pd.DataFrame({"circle": [1]}),
                workflow.area_polygon,
                pd.DataFrame({"circle": [1]}),
            )

        # Monkey patch the method at the module level
        import mitoolspro.google_utils.places.workflow as workflow_module

        original_search = workflow_module.places_search_step
        workflow_module.places_search_step = sample_search

        try:
            workflow.run()

            self.assertEqual(workflow.total_sampled_circles, 1)
            self.assertEqual(len(workflow.all_places), 1)
            self.assertEqual(workflow.all_places.iloc[0]["id"], "place1")
        finally:
            # Restore the original method
            workflow_module.places_search_step = original_search


if __name__ == "__main__":
    unittest.main()
