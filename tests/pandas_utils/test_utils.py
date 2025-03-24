import shutil
import unittest
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import DataFrame, IndexSlice, MultiIndex
from pandas.testing import assert_frame_equal

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.pandas_utils.utils import (
    check_if_dataframe_sequence,
    dataframe_to_latex,
    idxslice,
    load_dataframe_parquet,
    load_dataframe_sequence,
    remove_dataframe_duplicates,
    save_dataframes_to_excel,
    select_columns,
    select_index,
    store_dataframe_parquet,
    store_dataframe_sequence,
)


class TestIdxSlice(TestCase):
    def setUp(self):
        self.multiindex_index = MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["group", "number"]
        )
        self.multiindex_columns = MultiIndex.from_tuples(
            [("X", "x"), ("X", "y"), ("Y", "z")], names=["level1", "level2"]
        )
        self.multiindex_df = DataFrame(
            np.random.rand(4, 3),
            index=self.multiindex_index,
            columns=self.multiindex_columns,
        )
        self.single_index_df = DataFrame(
            np.random.rand(4, 3), index=["A", "B", "C", "D"], columns=["X", "Y", "Z"]
        )

    def test_multilevel_index_valid_slicing(self):
        slice_obj = idxslice(self.multiindex_df, level="group", values="A", axis=0)
        expected = IndexSlice[(["A"], slice(None))]
        self.assertEqual(slice_obj, expected)
        slice_obj = idxslice(self.multiindex_df, level=1, values=[1, 2], axis=0)
        expected = IndexSlice[(slice(None), [1, 2])]
        self.assertEqual(slice_obj, expected)

    def test_multilevel_columns_valid_slicing(self):
        slice_obj = idxslice(self.multiindex_df, level="level1", values="X", axis=1)
        expected = IndexSlice[(["X"], slice(None))]
        self.assertEqual(slice_obj, expected)
        slice_obj = idxslice(self.multiindex_df, level=1, values="z", axis=1)
        expected = IndexSlice[(slice(None), ["z"])]
        self.assertEqual(slice_obj, expected)

    def test_single_index_valid_slicing(self):
        slice_obj = idxslice(self.single_index_df, level=0, values=["A", "C"], axis=0)
        expected = IndexSlice[["A", "C"]]
        self.assertEqual(slice_obj, expected)

    def test_single_columns_valid_slicing(self):
        slice_obj = idxslice(self.single_index_df, level=0, values=["X", "Z"], axis=1)
        expected = IndexSlice[["X", "Z"]]
        self.assertEqual(slice_obj, expected)

    def test_invalid_axis(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="group", values="A", axis=2)

    def test_invalid_level_in_multilevel_index(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="invalid_level", values="A", axis=0)

    def test_invalid_level_in_single_index(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.single_index_df, level="invalid_level", values="A", axis=0)

    def test_invalid_level_position_single_index(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.single_index_df, level=1, values="A", axis=0)

    def test_invalid_level_position_multilevel_index(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level=10, values="A", axis=0)

    def test_invalid_level_name_single_index(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.single_index_df, level="invalid", values="A", axis=0)

    def test_invalid_level_name_multilevel_columns(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="invalid", values="X", axis=1)

    def test_non_list_values(self):
        slice_obj = idxslice(self.single_index_df, level=0, values="A", axis=0)
        expected = IndexSlice[["A"]]
        self.assertEqual(slice_obj, expected)
        slice_obj = idxslice(self.multiindex_df, level="group", values="B", axis=0)
        expected = IndexSlice[(["B"], slice(None))]
        self.assertEqual(slice_obj, expected)

    def test_multilevel_index_no_matching_level(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="nonexistent", values="X", axis=0)

    def test_multilevel_columns_invalid_axis(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="level1", values="X", axis=2)

    def test_single_index_invalid_axis(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.single_index_df, level=0, values="A", axis=2)

    def test_single_index_invalid_values(self):
        with self.assertRaises(KeyError):
            self.single_index_df.loc[
                idxslice(self.single_index_df, level=0, values="E", axis=0)
            ]

    def test_multilevel_columns_missing_level(self):
        with self.assertRaises(ArgumentValueError):
            idxslice(self.multiindex_df, level="nonexistent", values="X", axis=1)


class TestParquetStorage(TestCase):
    def setUp(self):
        self.test_dir = Path("test_parquet_storage")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_store_and_load_simple_dataframe(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        store_dataframe_parquet(df, self.test_dir, "simple_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "simple_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_multiindex_dataframe(self):
        index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["group", "number"]
        )
        columns = pd.MultiIndex.from_tuples(
            [("X", "x"), ("X", "y"), ("Y", "z")], names=["level1", "level2"]
        )
        df = DataFrame(np.random.rand(4, 3), index=index, columns=columns)
        store_dataframe_parquet(df, self.test_dir, "multiindex_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "multiindex_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_with_default_index(self):
        df = DataFrame(np.random.rand(5, 5), columns=list("ABCDE"))
        store_dataframe_parquet(df, self.test_dir, "default_index_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "default_index_df")
        assert_frame_equal(df, loaded_df)

    def test_store_and_load_with_non_default_index(self):
        df = DataFrame({"A": [10, 20, 30], "B": [40, 50, 60]}, index=[100, 200, 300])
        store_dataframe_parquet(df, self.test_dir, "custom_index_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "custom_index_df")
        assert_frame_equal(df, loaded_df)

    def test_overwrite_protection(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        store_dataframe_parquet(df, self.test_dir, "protected_df", overwrite=True)
        with self.assertRaises(ArgumentValueError):
            store_dataframe_parquet(df, self.test_dir, "protected_df", overwrite=False)

    def test_missing_directory(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with self.assertRaises(ArgumentValueError):
            store_dataframe_parquet(df, "non_existent_dir", "missing_dir_df")

    def test_missing_parquet_file(self):
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        with self.assertRaises(ArgumentValueError):
            load_dataframe_parquet(df, self.test_dir, "nonexistent_df")

    def test_missing_index_file(self):
        df = DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]},
            index=pd.MultiIndex.from_tuples(
                [("A", 1), ("B", 2), ("C", 3)], names=["letter", "number"]
            ),
        )
        store_dataframe_parquet(df, self.test_dir, "missing_index_test", overwrite=True)
        (
            self.test_dir / "missing_index_test_index.parquet"
        ).unlink()  # Remove the index file
        loaded_df = load_dataframe_parquet(df, self.test_dir, "missing_index_test")
        self.assertTrue(
            loaded_df.index.equals(pd.RangeIndex(start=0, stop=3))
        )  # Default index applied

    def test_missing_columns_file(self):
        df = DataFrame(
            np.random.rand(3, 3),
            columns=pd.MultiIndex.from_tuples(
                [("X", "x"), ("Y", "y"), ("Z", "z")], names=["level1", "level2"]
            ),
        )
        store_dataframe_parquet(
            df, self.test_dir, "missing_columns_test", overwrite=True
        )
        (
            self.test_dir / "missing_columns_test_columns.parquet"
        ).unlink()  # Remove the columns file
        loaded_df = load_dataframe_parquet(df, self.test_dir, "missing_columns_test")
        self.assertTrue(
            loaded_df.columns.equals(pd.RangeIndex(start=0, stop=3))
        )  # Default columns applied

    def test_store_and_load_empty_dataframe(self):
        df = DataFrame()
        store_dataframe_parquet(df, self.test_dir, "empty_df", overwrite=True)
        loaded_df = load_dataframe_parquet(df, self.test_dir, "empty_df")
        assert_frame_equal(df, loaded_df)


class TestStoreDataFrameSequence(TestCase):
    def setUp(self):
        self.temp_dir = Path("./tests/.test_assets/.data")
        self.temp_dir.mkdir(exist_ok=True)
        self.dataframes = {
            1: DataFrame({"A": [1, 2, 3]}),
            2: DataFrame({"B": [4, 5, 6]}),
        }

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_valid_storage(self):
        store_dataframe_sequence(self.dataframes, "test_sequence", self.temp_dir)
        for seq_val in self.dataframes:
            filename = f"test_sequence_{seq_val}.parquet"
            filepath = self.temp_dir / "test_sequence" / filename
            self.assertTrue(filepath.exists())

    def test_non_dataframe_value(self):
        invalid_dataframes = {1: DataFrame({"A": [1, 2, 3]}), 2: "not a dataframe"}
        with self.assertRaises(ValueError):
            store_dataframe_sequence(invalid_dataframes, "test_sequence", self.temp_dir)

    def test_empty_dataframes(self):
        store_dataframe_sequence({}, "empty_sequence", self.temp_dir)
        sequence_dir = self.temp_dir / "empty_sequence"
        self.assertTrue(sequence_dir.exists())
        self.assertEqual(len(list(sequence_dir.iterdir())), 0)

    def test_io_error_handling(self):
        self.temp_dir.chmod(0o444)  # Read-only
        with self.assertRaises(IOError):
            store_dataframe_sequence(self.dataframes, "test_sequence", self.temp_dir)
        self.temp_dir.chmod(0o755)

    def test_filename_formatting(self):
        store_dataframe_sequence(self.dataframes, "test sequence", self.temp_dir)
        for seq_val in self.dataframes:
            filename = f"testsequence_{seq_val}.parquet"
            filepath = self.temp_dir / "test sequence" / filename
            self.assertTrue(filepath.exists())


class TestLoadDataFrameSequence(TestCase):
    def setUp(self):
        self.temp_dir = Path("./tests/.test_assets/.data")
        self.sequence_name = "test_sequence"
        self.sequence_dir = self.temp_dir / self.sequence_name
        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes = {
            1: DataFrame({"A": [1, 2, 3]}),
            2: DataFrame({"B": [4, 5, 6]}),
        }
        store_dataframe_sequence(self.dataframes, self.sequence_name, self.temp_dir)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_load_all_dataframes(self):
        result = load_dataframe_sequence(self.temp_dir, self.sequence_name)
        for seq_val, df in self.dataframes.items():
            assert_frame_equal(result[seq_val], df)

    def test_load_specific_sequence_values(self):
        with self.assertRaises(ArgumentValueError):
            load_dataframe_sequence(
                self.temp_dir, self.sequence_name, sequence_values=[1]
            )

    def test_directory_not_found(self):
        with self.assertRaises(ArgumentValueError):
            load_dataframe_sequence(Path("non_existent_dir"), self.sequence_name)

    def test_empty_directory(self):
        empty_dir = self.temp_dir / "empty_sequence"
        empty_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ArgumentValueError):
            load_dataframe_sequence(self.temp_dir, "empty_sequence")


class TestCheckIfDataFrameSequence(TestCase):
    def setUp(self):
        self.temp_dir = Path("./tests/.test_assets/.data")
        self.sequence_name = "test_sequence"
        self.sequence_dir = self.temp_dir / self.sequence_name
        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        self.dataframes = {
            1: DataFrame({"A": [1, 2, 3]}),
            2: DataFrame({"B": [4, 5, 6]}),
        }
        store_dataframe_sequence(self.dataframes, self.sequence_name, self.temp_dir)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_sequence_exists_and_matches(self):
        result = check_if_dataframe_sequence(
            self.temp_dir, self.sequence_name, sequence_values=[1, 2]
        )
        self.assertTrue(result)

    def test_sequence_partial_match(self):
        result = check_if_dataframe_sequence(
            self.temp_dir, self.sequence_name, sequence_values=[1, 3]
        )
        self.assertFalse(result)

    def test_nonexistent_directory(self):
        result = check_if_dataframe_sequence(
            Path("non_existent_dir"), self.sequence_name, sequence_values=[1]
        )
        self.assertFalse(result)

    def test_empty_directory(self):
        empty_dir = self.temp_dir / "empty_sequence"
        empty_dir.mkdir(parents=True, exist_ok=True)
        result = check_if_dataframe_sequence(
            empty_dir, "empty_sequence", sequence_values=[1]
        )
        self.assertFalse(result)


class TestSelectIndex(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}).T
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        self.df_multi = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=columns, index=columns
        ).T

    def test_single_column_single_level(self):
        result = select_index(self.df_single, "A")
        self.assertListEqual(list(result.index), ["A"])
        self.assertEqual(result.shape, (1, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_multiple_columns_single_level(self):
        result = select_index(self.df_single, ["A", "C"])
        self.assertListEqual(list(result.index), ["A", "C"])
        self.assertEqual(result.shape, (2, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_invalid_column_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, "D")
        self.assertIn("Invalid index", str(context.exception))

    def test_mix_columns_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, ["A", "D"])
        self.assertIn("Invalid index", str(context.exception))

    def test_empty_column_selection_single_level(self):
        result = select_index(self.df_single, [])
        self.assertEqual(result.shape, (0, 3))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_select_all_columns_single_level(self):
        result = select_index(self.df_single, list(self.df_single.index))
        self.assertListEqual(list(result.index), list(self.df_single.index))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))

    def test_multiindex_column_selection(self):
        result = select_index(self.df_multi, [("A", "X"), ("B", "Z")])
        self.assertListEqual(list(result.index), [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (2, 3))

    def test_multiindex_with_level_positional(self):
        result = select_index(self.df_multi, ["X", "Y"], level=1)
        self.assertListEqual(list(result.index), [("A", "X"), ("A", "Y")])

    def test_multiindex_with_level_name(self):
        result = select_index(self.df_multi, ["X", "Y"], level="lower")
        self.assertListEqual(list(result.index), [("A", "X"), ("A", "Y")])

    def test_invalid_level_name(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, ["X"], level="invalid")
        self.assertIn("Invalid level name", str(context.exception))

    def test_invalid_level_index(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, ["X"], level=2)
        self.assertIn("Invalid level index", str(context.exception))

    def test_tuple_column_mismatch(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_multi, [("A", "X", "extra")])
        self.assertIn("Invalid index", str(context.exception))

    def test_level_in_single_level_dataframe(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_index(self.df_single, ["A"], level=0)
        self.assertIn("level can only be specified", str(context.exception))

    def test_empty_dataframe_single_level(self):
        empty_df = DataFrame(index=["A", "B", "C"])
        result = select_index(empty_df, ["A", "B"])
        self.assertEqual(result.shape, (2, 0))

    def test_empty_dataframe_multiindex(self):
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        index = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        empty_multi_df = DataFrame(index=index)
        result = select_index(empty_multi_df, [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (2, 0))

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            select_index(self.df_single, {})
        self.assertIn(
            "Provided 'index' must be a string, tuple, int, or list.",
            str(context.exception),
        )


class TestSelectColumns(TestCase):
    def setUp(self):
        self.df_single = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        self.df_multi = DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=columns, index=columns
        )

    def test_single_column_single_level(self):
        result = select_columns(self.df_single, "A")
        self.assertListEqual(list(result.columns), ["A"])
        self.assertEqual(result.shape, (3, 1))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_multiple_columns_single_level(self):
        result = select_columns(self.df_single, ["A", "C"])
        self.assertListEqual(list(result.columns), ["A", "C"])
        self.assertEqual(result.shape, (3, 2))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_invalid_column_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, "D")
        self.assertIn("Invalid columns", str(context.exception))

    def test_mix_columns_single_level(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, ["A", "D"])
        self.assertIn("Invalid columns", str(context.exception))

    def test_empty_column_selection_single_level(self):
        result = select_columns(self.df_single, [])
        self.assertEqual(result.shape, (3, 0))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_select_all_columns_single_level(self):
        result = select_columns(self.df_single, list(self.df_single.columns))
        self.assertListEqual(list(result.columns), list(self.df_single.columns))
        self.assertListEqual(list(result.index), list(self.df_single.index))

    def test_multiindex_column_selection(self):
        result = select_columns(self.df_multi, [("A", "X"), ("B", "Z")])
        self.assertListEqual(list(result.columns), [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (3, 2))

    def test_multiindex_with_level_positional(self):
        result = select_columns(self.df_multi, ["X", "Y"], level=1)
        self.assertListEqual(list(result.columns), [("A", "X"), ("A", "Y")])

    def test_multiindex_with_level_name(self):
        result = select_columns(self.df_multi, ["X", "Y"], level="lower")
        self.assertListEqual(list(result.columns), [("A", "X"), ("A", "Y")])

    def test_invalid_level_name(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, ["X"], level="invalid")
        self.assertIn("Invalid level name", str(context.exception))

    def test_invalid_level_index(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, ["X"], level=2)
        self.assertIn("Invalid level index", str(context.exception))

    def test_tuple_column_mismatch(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_multi, [("A", "X", "extra")])
        self.assertIn("Invalid columns", str(context.exception))

    def test_level_in_single_level_dataframe(self):
        with self.assertRaises(ArgumentValueError) as context:
            select_columns(self.df_single, ["A"], level=0)
        self.assertIn("level can only be specified", str(context.exception))

    def test_empty_dataframe_single_level(self):
        empty_df = DataFrame(columns=["A", "B", "C"])
        result = select_columns(empty_df, ["A", "B"])
        self.assertEqual(result.shape, (0, 2))

    def test_empty_dataframe_multiindex(self):
        arrays = [["A", "A", "B"], ["X", "Y", "Z"]]
        columns = MultiIndex.from_arrays(arrays, names=["upper", "lower"])
        empty_multi_df = DataFrame(columns=columns)
        result = select_columns(empty_multi_df, [("A", "X"), ("B", "Z")])
        self.assertEqual(result.shape, (0, 2))

    def test_invalid_column_type(self):
        with self.assertRaises(ArgumentTypeError) as context:
            select_columns(self.df_single, {})
        self.assertIn(
            "Provided 'columns' must be a string, tuple, int, or list.",
            str(context.exception),
        )


class TestRemoveDataframeDuplicates(TestCase):
    def setUp(self):
        self.df1 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.df2 = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.df3 = DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

    def test_remove_dataframe_duplicates(self):
        dfs = [self.df1, self.df2, self.df3]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 2)
        self.assertTrue(unique_dfs[0].equals(self.df1))
        self.assertTrue(unique_dfs[1].equals(self.df3))

    def test_remove_dataframe_duplicates_no_duplicates(self):
        dfs = [self.df1, self.df3]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 2)
        self.assertTrue(unique_dfs[0].equals(self.df1))
        self.assertTrue(unique_dfs[1].equals(self.df3))

    def test_remove_dataframe_duplicates_all_duplicates(self):
        dfs = [self.df1, self.df1, self.df1]
        unique_dfs = remove_dataframe_duplicates(dfs)
        self.assertEqual(len(unique_dfs), 1)
        self.assertTrue(unique_dfs[0].equals(self.df1))


if __name__ == "__main__":
    unittest.main()
