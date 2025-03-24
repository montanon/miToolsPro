import unittest
from unittest import TestCase

import pandas as pd
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.pandas_utils.transform_frame import (
    get_entities_data,
    get_entity_data,
    long_to_wide_dataframe,
    reshape_countries_indicators,
    reshape_country_indicators,
    reshape_group_data,
    reshape_groups_subgroups,
    wide_to_long_dataframe,
)


class TestReshapeCountryIndicators(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_country_indicators(
            data=self.df,
            country="A",
            indicator_column="value",
            country_column="group",
            region_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_country_indicators(
            data=df,
            country="A",
            indicator_column="value",
            country_column="group",
            region_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {"X": [12.5, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_country_indicators(
                data=df,
                country="A",
                indicator_column="value",
                country_column="group",
                region_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_no_matching_filter_value(self):
        with self.assertRaises(ArgumentValueError) as context:
            reshape_country_indicators(
                data=self.df,
                country="C",
                indicator_column="value",
                country_column="group",
                region_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'C' in column 'group'.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            reshape_country_indicators(
                data=empty_df,
                country="A",
                indicator_column="value",
                country_column="group",
                region_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'A' in column 'group'.", str(context.exception)
        )

    def test_preserves_column_order(self):
        result = reshape_country_indicators(
            data=self.df,
            country="A",
            indicator_column="value",
            country_column="group",
            region_column="subgroup",
            time_column="time",
        )
        self.assertEqual(list(result.columns), ["X", "Y", "Z"])

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_country_indicators(
            data=large_df,
            country="A",
            indicator_column="value",
            country_column="group",
            region_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result is non-empty

    def test_single_subgroup(self):
        df = self.df[self.df["subgroup"] == "X"]
        result = reshape_country_indicators(
            data=df,
            country="A",
            indicator_column="value",
            country_column="group",
            region_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0]}, index=["2021-01", "2021-02", "2021-03"]
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected, check_dtype=False)


class TestReshapeGroupData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_group_data(
            dataframe=self.df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_group_data(
            dataframe=df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {"X": [12.5, None, 60.0], "Y": [20.0, None, None], "Z": [30.0, None, None]},
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=df,
                filter_value="A",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_no_matching_filter_value(self):
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=self.df,
                filter_value="C",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'C' in column 'group'.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            reshape_group_data(
                dataframe=empty_df,
                filter_value="A",
                value_column="value",
                group_column="group",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "No data found for group 'A' in column 'group'.", str(context.exception)
        )

    def test_preserves_column_order(self):
        result = reshape_group_data(
            dataframe=self.df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertEqual(list(result.columns), ["X", "Y", "Z"])

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_group_data(
            dataframe=large_df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result is non-empty

    def test_single_subgroup(self):
        df = self.df[self.df["subgroup"] == "X"]
        result = reshape_group_data(
            dataframe=df,
            filter_value="A",
            value_column="value",
            group_column="group",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {"X": [10.0, None, 60.0]}, index=["2021-01", "2021-02", "2021-03"]
        )
        expected.columns.name = "subgroup"
        expected.index.name = "A"
        assert_frame_equal(result, expected, check_dtype=False)


class TestReshapeCountriesIndicators(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_countries_indicators(
            data=self.df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_countries_indicators(
            data=df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                ("A", "X"): [12.5, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_countries_indicators(
                data=df,
                country_column="group",
                indicator_column="value",
                region_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError):
            reshape_countries_indicators(
                data=empty_df,
                country_column="group",
                indicator_column="value",
                region_column="subgroup",
                time_column="time",
            )

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_countries_indicators(
            data=large_df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[1] > 0)  # Ensure columns exist
        self.assertTrue(result.shape[0] > 0)  # Ensure rows exist

    def test_single_group(self):
        single_group_df = DataFrame(
            {
                "group": ["A", "A", "A", "A", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Z"],
                "value": [10, 20, 30, 60, None],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-03",
                    "2021-02",
                ],
            }
        )
        result = reshape_countries_indicators(
            data=single_group_df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.index.name = "A"
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_column_type_mismatch(self):
        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])  # Convert time to datetime
        result = reshape_countries_indicators(
            data=df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.index.dtype == "datetime64[ns]")

    def test_preserves_column_order(self):
        result = reshape_countries_indicators(
            data=self.df,
            country_column="group",
            indicator_column="value",
            region_column="subgroup",
            time_column="time",
        )
        self.assertEqual(
            result.columns.get_level_values(1).tolist(), ["X", "Y", "Z", "X", "Y"]
        )


class TestReshapeGroupsSubgroups(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Y", "X"],
                "value": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-02",
                    "2021-02",
                    "2021-03",
                ],
            }
        )

    def test_valid_input(self):
        result = reshape_groups_subgroups(
            dataframe=self.df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["A", "X", 15, "2021-01"]  # Add duplicate to test aggregation
        result = reshape_groups_subgroups(
            dataframe=df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                ("A", "X"): [12.5, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
                ("B", "X"): [None, 40.0, None],
                ("B", "Y"): [None, 50.0, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["subgroup"])
        with self.assertRaises(ArgumentValueError) as context:
            reshape_groups_subgroups(
                dataframe=df,
                group_column="group",
                value_column="value",
                subgroup_column="subgroup",
                time_column="time",
            )
        self.assertIn(
            "Columns {'subgroup'} not found in the DataFrame.", str(context.exception)
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError):
            reshape_groups_subgroups(
                dataframe=empty_df,
                group_column="group",
                value_column="value",
                subgroup_column="subgroup",
                time_column="time",
            )

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = reshape_groups_subgroups(
            dataframe=large_df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.shape[1] > 0)  # Ensure columns exist
        self.assertTrue(result.shape[0] > 0)  # Ensure rows exist

    def test_single_group(self):
        single_group_df = DataFrame(
            {
                "group": ["A", "A", "A", "A", "A"],
                "subgroup": ["X", "Y", "Z", "X", "Z"],
                "value": [10, 20, 30, 60, None],
                "time": [
                    "2021-01",
                    "2021-01",
                    "2021-01",
                    "2021-03",
                    "2021-02",
                ],
            }
        )
        result = reshape_groups_subgroups(
            dataframe=single_group_df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("A", "X"): [10.0, None, 60.0],
                ("A", "Y"): [20.0, None, None],
                ("A", "Z"): [30.0, None, None],
            },
            index=["2021-01", "2021-02", "2021-03"],
        )
        expected.index.name = "A"
        expected.columns.names = ["group", "subgroup"]
        assert_frame_equal(result, expected)

    def test_column_type_mismatch(self):
        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])  # Convert time to datetime
        result = reshape_groups_subgroups(
            dataframe=df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertTrue(result.index.dtype == "datetime64[ns]")

    def test_preserves_column_order(self):
        result = reshape_groups_subgroups(
            dataframe=self.df,
            group_column="group",
            value_column="value",
            subgroup_column="subgroup",
            time_column="time",
        )
        self.assertEqual(
            result.columns.get_level_values(1).tolist(), ["X", "Y", "Z", "X", "Y"]
        )


class TestGetEntityData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "country": ["USA", "USA", "USA", "CAN", "CAN", "USA"],
                "indicator1": [100, 200, 300, 400, 500, 600],
                "indicator2": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q3",
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q4",
                ],
            }
        )

    def test_valid_input(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                "indicator1": [100, 200, 300, 600],
                "indicator2": [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "USA"
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["USA", 150, 15, "2021-Q1"]  # Add duplicate for aggregation
        result = get_entity_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                "indicator1": [125.0, 200.0, 300.0, 600.0],
                "indicator2": [12.5, 20.0, 30.0, 60.0],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "USA"
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["indicator1"])
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=df,
                data_columns=["indicator1", "indicator2"],
                entity="USA",
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "Columns {'indicator1'} not found in the DataFrame.", str(context.exception)
        )

    def test_no_matching_entity(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=self.df,
                data_columns=["indicator1", "indicator2"],
                entity="MEX",
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "No data found for entity 'MEX' in column 'country'.",
            str(context.exception),
        )

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            get_entity_data(
                dataframe=empty_df,
                data_columns=["indicator1", "indicator2"],
                entity="USA",
                entity_column="country",
                time_column="time",
            )
        self.assertIn("No data found for entity 'USA'", str(context.exception))

    def test_column_type_mismatch(self):
        df = self.df.copy()
        df["time"] = pd.to_datetime(df["time"])  # Convert time to datetime
        result = get_entity_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.index.dtype == "datetime64[ns]")

    def test_reindexing_with_missing_times(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity="CAN",
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                "indicator1": [400, 500, None, None],
                "indicator2": [40, 50, None, None],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "CAN"
        assert_frame_equal(result, expected)

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = get_entity_data(
            dataframe=large_df,
            data_columns=["indicator1", "indicator2"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result has rows
        self.assertTrue(result.shape[1] == 2)  # Ensure the result has two columns

    def test_sorts_column_order(self):
        result = get_entity_data(
            dataframe=self.df,
            data_columns=["indicator2", "indicator1"],
            entity="USA",
            entity_column="country",
            time_column="time",
        )
        self.assertEqual(list(result.columns), ["indicator1", "indicator2"])


class TestGetEntitiesData(TestCase):
    def setUp(self):
        self.df = DataFrame(
            {
                "country": ["USA", "USA", "USA", "CAN", "CAN", "USA"],
                "indicator1": [100, 200, 300, 400, 500, 600],
                "indicator2": [10, 20, 30, 40, 50, 60],
                "time": [
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q3",
                    "2021-Q1",
                    "2021-Q2",
                    "2021-Q4",
                ],
            }
        )

    def test_valid_input(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
        )
        expected = DataFrame(
            {
                ("CAN", "indicator1"): [400, 500, None, None],
                ("CAN", "indicator2"): [40, 50, None, None],
                ("USA", "indicator1"): [100, 200, 300, 600],
                ("USA", "indicator2"): [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_custom_aggregation_function(self):
        df = self.df.copy()
        df.loc[len(df)] = ["USA", 150, 15, "2021-Q1"]  # Add duplicate for aggregation
        result = get_entities_data(
            dataframe=df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
            agg_func="mean",
        )
        expected = DataFrame(
            {
                ("CAN", "indicator1"): [400.0, 500.0, None, None],
                ("CAN", "indicator2"): [40.0, 50.0, None, None],
                ("USA", "indicator1"): [125.0, 200.0, 300.0, 600.0],
                ("USA", "indicator2"): [12.5, 20.0, 30.0, 60.0],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_missing_required_columns(self):
        df = self.df.drop(columns=["indicator1"])
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
            )
        self.assertIn(
            "Columns {'indicator1'} not found in the DataFrame.", str(context.exception)
        )

    def test_specific_entities(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
            entities=["USA"],
        )
        expected = DataFrame(
            {
                ("USA", "indicator1"): [100, 200, 300, 600],
                ("USA", "indicator2"): [10, 20, 30, 60],
            },
            index=["2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4"],
        )
        expected.index.name = "time"
        expected.columns.names = ["country", "indicator"]
        assert_frame_equal(result, expected)

    def test_no_matching_entities(self):
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=self.df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
                entities=["MEX"],
            )
        self.assertIn("Error processing entity 'MEX'", str(context.exception))

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=self.df.columns)
        with self.assertRaises(ArgumentValueError) as context:
            get_entities_data(
                dataframe=empty_df,
                data_columns=["indicator1", "indicator2"],
                entity_column="country",
                time_column="time",
            )

    def test_large_dataframe(self):
        large_df = pd.concat([self.df] * 10000, ignore_index=True)
        result = get_entities_data(
            dataframe=large_df,
            data_columns=["indicator1", "indicator2"],
            entity_column="country",
            time_column="time",
        )
        self.assertTrue(result.shape[0] > 0)  # Ensure the result has rows
        self.assertTrue(result.shape[1] == 4)  # Ensure the result has four columns

    def test_sorts_column_order(self):
        result = get_entities_data(
            dataframe=self.df,
            data_columns=["indicator2", "indicator1"],
            entity_column="country",
            time_column="time",
        )
        self.assertEqual(
            list(result.columns.get_level_values(1)), ["indicator1", "indicator2"] * 2
        )


class TestWideToLongDataFrame(TestCase):
    def setUp(self):
        self.data = DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3, 2],
                "category": ["A", "B", "A", "B", "A", "B", "C"],
                "category2": ["A", "D", "A", "B", "A", "B", "C"],
                "year": [2020, 2020, 2021, 2021, 2022, 2022, 2021],
                "value": [10, 20, 30, 40, 50, 60, 5],
                "value2": [100, 200, 300, 400, 500, 600, 30],
            }
        )

    def test_basic_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data, index="id", columns="category", values="value"
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_multiple_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index=["id", "year"],
            columns=["category", "category2"],
            values=["value", "value2"],
            filter_columns={"category": ["A"], "category2": ["A"]},
        )
        self.assertTrue(result.columns.names == [None, "category", "category2"])
        self.assertTrue(result.index.names == ["id", "year"])
        self.assertTrue(result.shape == (3, 2))

    def test_multi_column_transformation(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index=["id", "year"],
            columns="category",
            values="value",
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "year": [2020, 2021, 2022],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index(["id", "year"])
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_filter_index(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            filter_index={"id": [1, 2]},
        )
        expected = DataFrame(
            {"id": [1, 2], "A": [10, 30], "B": [20, 40], "C": [None, 5]}
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_filter_columns(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            filter_columns={"category": ["A"]},
        )
        expected = DataFrame({"id": [1, 2, 3], "A": [10, 30, 50]}).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected)

    def test_aggfunc_sum(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            agg_func="sum",
        )
        expected = DataFrame(
            {
                "id": [1, 2, 3],
                "A": [10, 30, 50],
                "B": [20, 40, 60],
                "C": [None, 5, None],
            }
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected, check_dtype=False)

    def test_fill_value(self):
        result = wide_to_long_dataframe(
            dataframe=self.data,
            index="id",
            columns="category",
            values="value",
            fill_value=0,
        )
        expected = DataFrame(
            {"id": [1, 2, 3], "A": [10, 30, 50], "B": [20, 40, 60], "C": [0, 5, 0]}
        ).set_index("id")
        expected.columns = pd.MultiIndex.from_product([["value"], expected.columns])
        expected.columns.names = [None, "category"]
        assert_frame_equal(result, expected)

    def test_missing_columns_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="nonexistent",
                columns="category",
                values="value",
            )

    def test_invalid_filter_index_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="id",
                columns="category",
                values="value",
                filter_index={"nonexistent": [1]},
            )

    def test_invalid_filter_columns_error(self):
        with self.assertRaises(ArgumentValueError):
            wide_to_long_dataframe(
                dataframe=self.data,
                index="id",
                columns="category",
                values="value",
                filter_columns={"nonexistent": ["A"]},
            )


class TestLongToWideDataFrame(TestCase):
    def setUp(self):
        self.data = DataFrame(
            {
                "id": [1, 2, 3],
                "2020_A": [10, 20, 30],
                "2020_B": [40, 50, 60],
                "2021_A": [70, 80, 90],
                "2021_B": [100, 110, 120],
            }
        )

    def test_basic_transformation(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            var_name="year_category",
            value_name="value",
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "year_category": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "value": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["year_category", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_default_parameters(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "value": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_no_value_vars(self):
        data = DataFrame(
            {"id": [1, 2, 3], "2020_A": [10, 20, 30], "2020_B": [40, 50, 60]}
        )
        result = long_to_wide_dataframe(dataframe=data, id_vars="id")
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 2, 2, 3, 3],
                    "variable": ["2020_A", "2020_B"] * 3,
                    "value": [10, 40, 20, 50, 30, 60],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_filter_id_vars(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            filter_id_vars={"id": [1]},
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"],
                    "value": [10, 40, 70, 100],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_filter_value_vars(self):
        result = long_to_wide_dataframe(
            dataframe=self.data, id_vars="id", value_vars=["2020_A", "2021_A"]
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 2, 2, 3, 3],
                    "variable": ["2020_A", "2021_A"] * 3,
                    "value": [10, 70, 20, 80, 30, 90],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_missing_columns(self):
        with self.assertRaises(ArgumentValueError):
            long_to_wide_dataframe(
                dataframe=self.data,
                id_vars="id",
                value_vars=["2020_A", "missing_column"],
            )

    def test_empty_dataframe(self):
        empty_data = DataFrame(columns=["id", "2020_A", "2020_B"])
        expected = DataFrame(columns=["id", "variable", "value"])
        result = long_to_wide_dataframe(dataframe=empty_data, id_vars="id")
        assert_frame_equal(result, expected)

    def test_custom_var_value_names(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            var_name="attribute",
            value_name="measurement",
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                    "attribute": ["2020_A", "2020_B", "2021_A", "2021_B"] * 3,
                    "measurement": [10, 40, 70, 100, 20, 50, 80, 110, 30, 60, 90, 120],
                }
            )
            .sort_values(by=["attribute", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)

    def test_invalid_id_vars(self):
        with self.assertRaises(ArgumentValueError):
            long_to_wide_dataframe(dataframe=self.data, id_vars="missing_id_var")

    def test_filter_with_scalar(self):
        result = long_to_wide_dataframe(
            dataframe=self.data,
            id_vars="id",
            value_vars=["2020_A", "2020_B", "2021_A", "2021_B"],
            filter_id_vars={"id": 1},
        )
        expected = (
            DataFrame(
                {
                    "id": [1, 1, 1, 1],
                    "variable": ["2020_A", "2020_B", "2021_A", "2021_B"],
                    "value": [10, 40, 70, 100],
                }
            )
            .sort_values(by=["variable", "id"])
            .reset_index(drop=True)
        )
        assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
