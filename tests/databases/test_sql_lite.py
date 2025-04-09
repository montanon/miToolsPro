import os
import tempfile
from pathlib import Path
from unittest import TestCase

import pandas as pd
import pytest

from mitoolspro.databases.sql_lite import (
    CustomConnection,
    MainConnection,
    check_if_table,
    check_if_tables,
    connect_to_sql_db,
    get_conn_db_folder,
    read_sql_table,
    read_sql_table_with_types,
    read_sql_tables,
    read_sql_tables_with_types,
    transfer_sql_table,
    validate_table_name,
)
from mitoolspro.exceptions import ArgumentValueError


class TestSQLiteFunctions(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.conn = connect_to_sql_db(self.temp_dir, "test.db")
        self.test_df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": ["a", "b", "c"],
                "col3": ["2021-01-01", "2021-02-01", "2021-03-01"],
                "col4": [1, 0, 1],  # Store as int for SQLite
                "col5": [1.1, 2.2, 3.3],
            }
        )
        self.test_df.to_sql("test_table", self.conn, index=False)

    def tearDown(self):
        self.conn.close()
        os.remove(self.db_path)
        new_db_path = Path(self.temp_dir) / "new_test.db"
        if new_db_path.exists():
            os.remove(new_db_path)
        os.rmdir(self.temp_dir)

    def test_validate_table_name(self):
        self.assertEqual(validate_table_name("valid_table"), "valid_table")
        self.assertEqual(validate_table_name("valid_table_123"), "valid_table_123")
        with pytest.raises(ValueError):
            validate_table_name("invalid-table")
        with pytest.raises(ValueError):
            validate_table_name("invalid@table")

    def test_check_if_table(self):
        self.assertTrue(check_if_table(self.conn, "test_table"))
        self.assertFalse(check_if_table(self.conn, "nonexistent_table"))

    def test_check_if_tables(self):
        results = check_if_tables(self.conn, ["test_table", "nonexistent_table"])
        self.assertEqual(results, [True, False])

    def test_get_conn_db_folder(self):
        db_folder = get_conn_db_folder(self.conn)
        resolved_temp_dir = Path(self.temp_dir).resolve()
        self.assertEqual(str(db_folder), str(resolved_temp_dir))

    def test_connect_to_sql_db(self):
        conn = connect_to_sql_db(self.temp_dir, "test.db")
        self.assertIsInstance(conn, CustomConnection)
        conn.close()

    def test_read_sql_table(self):
        df = read_sql_table(self.conn, "test_table")
        expected_df = self.test_df.copy()
        pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)
        df_with_columns = read_sql_table(self.conn, "test_table", columns=["col1"])
        pd.testing.assert_frame_equal(df_with_columns, self.test_df[["col1"]])
        df_with_index = read_sql_table(self.conn, "test_table", index_col="col1")
        expected_df = self.test_df.set_index("col1")
        pd.testing.assert_frame_equal(df_with_index, expected_df, check_dtype=False)

    def test_read_sql_tables(self):
        dfs = read_sql_tables(self.conn, ["test_table"], index_col=None)
        self.assertEqual(len(dfs), 1)
        expected_df = self.test_df.copy()
        pd.testing.assert_frame_equal(dfs[0], expected_df, check_dtype=False)

    def test_transfer_sql_table(self):
        new_conn = connect_to_sql_db(self.temp_dir, "new_test.db")

        transfer_sql_table(self.conn, new_conn, "test_table", index_col=None)
        transferred_df = read_sql_table(new_conn, "test_table", index_col=None)
        expected_df = self.test_df.copy()
        pd.testing.assert_frame_equal(transferred_df, expected_df, check_dtype=False)

        new_conn.close()

    def test_main_connection_singleton(self):
        conn1 = MainConnection(self.db_path)
        conn2 = MainConnection(self.db_path)
        self.assertIs(conn1, conn2)
        conn1.close()

    def test_read_sql_table_with_types(self):
        column_types = {
            "col1": "int64",
            "col2": "category",
            "col3": "datetime64[ns]",
            "col4": "bool",
            "col5": "float64",
        }

        df = read_sql_table_with_types(self.conn, "test_table", column_types)

        self.assertEqual(df["col1"].dtype, "int64")
        self.assertEqual(df["col2"].dtype.name, "category")
        self.assertEqual(df["col3"].dtype, "datetime64[ns]")
        self.assertEqual(df["col4"].dtype, "bool")
        self.assertEqual(df["col5"].dtype, "float64")

        # Test with specific columns
        df_subset = read_sql_table_with_types(
            self.conn,
            "test_table",
            {"col1": "int64", "col3": "datetime64[ns]"},
            columns=["col1", "col3"],
        )
        self.assertEqual(df_subset["col1"].dtype, "int64")
        self.assertEqual(df_subset["col3"].dtype, "datetime64[ns]")
        self.assertEqual(len(df_subset.columns), 2)

    def test_read_sql_table_with_types_invalid_column(self):
        with self.assertRaises(ArgumentValueError) as context:
            read_sql_table_with_types(
                self.conn, "test_table", {"nonexistent_col": "int64"}
            )
        self.assertIn(
            "Columns ['nonexistent_col'] not found in table test_table",
            str(context.exception),
        )

    def test_read_sql_table_with_types_invalid_dtype(self):
        with self.assertRaises(ArgumentValueError) as context:
            read_sql_table_with_types(
                self.conn, "test_table", {"col1": "invalid_dtype"}
            )
        self.assertIn("Invalid dtypes specified", str(context.exception))

    def test_read_sql_table_with_types_all_valid_dtypes(self):
        # Test all valid dtypes
        valid_dtypes = {
            "col1": "int64",
            "col2": "category",
            "col3": "datetime64[ns]",
            "col4": "bool",
            "col5": "float64",
            "col2": "string",  # Test string type
            "col1": "Int64",  # Test nullable integer
            "col5": "Float64",  # Test nullable float
        }

        # This should not raise an error
        df = read_sql_table_with_types(self.conn, "test_table", valid_dtypes)
        self.assertIsInstance(df, pd.DataFrame)

    def test_read_sql_tables_with_types(self):
        # Create a second table
        test_df2 = pd.DataFrame(
            {
                "col1": [4, 5, 6],
                "col2": ["d", "e", "f"],
                "col3": ["2021-04-01", "2021-05-01", "2021-06-01"],
            }
        )
        test_df2.to_sql("test_table2", self.conn, index=False)

        table_types = {
            "test_table": {
                "col1": "int64",
                "col2": "category",
                "col3": "datetime64[ns]",
                "col4": "bool",
                "col5": "float64",
            },
            "test_table2": {
                "col1": "float64",
                "col2": "category",
                "col3": "datetime64[ns]",
            },
        }

        dfs = read_sql_tables_with_types(
            self.conn, ["test_table", "test_table2"], table_types, index_col=None
        )

        # Check first table types
        self.assertEqual(dfs[0]["col1"].dtype, "int64")
        self.assertEqual(dfs[0]["col2"].dtype.name, "category")
        self.assertEqual(dfs[0]["col3"].dtype, "datetime64[ns]")
        self.assertEqual(dfs[0]["col4"].dtype, "bool")
        self.assertEqual(dfs[0]["col5"].dtype, "float64")

        # Check second table types
        self.assertEqual(dfs[1]["col1"].dtype, "float64")
        self.assertEqual(dfs[1]["col2"].dtype.name, "category")
        self.assertEqual(dfs[1]["col3"].dtype, "datetime64[ns]")
