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
    read_sql_tables,
    transfer_sql_table,
    validate_table_name,
)


class TestSQLiteFunctions(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.conn = connect_to_sql_db(self.temp_dir, "test.db")
        self.test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
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
        pd.testing.assert_frame_equal(df, self.test_df)

        df_with_columns = read_sql_table(self.conn, "test_table", columns=["col1"])
        pd.testing.assert_frame_equal(df_with_columns, self.test_df[["col1"]])

        df_with_index = read_sql_table(self.conn, "test_table", index_col="col1")
        expected_df = self.test_df.set_index("col1")
        pd.testing.assert_frame_equal(df_with_index, expected_df)

    def test_read_sql_tables(self):
        dfs = read_sql_tables(self.conn, ["test_table"], index_col=None)
        self.assertEqual(len(dfs), 1)
        pd.testing.assert_frame_equal(dfs[0], self.test_df)

    def test_transfer_sql_table(self):
        new_conn = connect_to_sql_db(self.temp_dir, "new_test.db")

        transfer_sql_table(self.conn, new_conn, "test_table", index_col=None)
        transferred_df = read_sql_table(new_conn, "test_table", index_col=None)
        pd.testing.assert_frame_equal(transferred_df, self.test_df)

        new_conn.close()

    def test_main_connection_singleton(self):
        conn1 = MainConnection(self.db_path)
        conn2 = MainConnection(self.db_path)
        self.assertIs(conn1, conn2)
        conn1.close()
