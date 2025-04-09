from os import PathLike
from pathlib import Path
from sqlite3 import Connection, OperationalError
from typing import Dict, Iterable, List, Literal, Optional, Union

import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.utils.decorators import suppress_user_warning


def validate_table_name(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Unsafe table name: {name}")
    return name


class CustomConnection(Connection):
    def __init__(self, path: PathLike, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self.path = Path(path).absolute()

    @property
    def __class__(self):
        return Connection


class MainConnection(CustomConnection):
    _instances = {}

    def __new__(cls, path: PathLike):
        path = Path(path).absolute()
        if path not in cls._instances:
            cls._instances[path] = super(MainConnection, cls).__new__(cls)
            cls._instances[path]._initialized = False
        return cls._instances[path]

    def __init__(self, path: PathLike, *args, **kwargs):
        if not self._initialized:
            super().__init__(path, *args, **kwargs)
            self._initialized = True


def check_if_table(conn: Connection, table_name: str) -> bool:
    validate_table_name(table_name)
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
    params = (table_name,)
    cursor = conn.cursor()
    try:
        return cursor.execute(query, params).fetchone() is not None
    except OperationalError:
        try:
            parquet_folder = get_conn_db_folder(conn) / "parquet"
            return (parquet_folder / f"{table_name}.parquet").exists()
        except Exception:
            return False


def check_if_tables(conn: Connection, tables_names: Iterable[str]) -> List[bool]:
    return [check_if_table(conn, table_name) for table_name in tables_names]


def get_conn_db_folder(conn: Connection) -> Path:
    cursor = conn.cursor()
    cursor.execute("PRAGMA database_list;")
    db_path = Path(cursor.fetchone()[2])
    return db_path.parent.absolute()


def connect_to_sql_db(db_path: Union[str, PathLike], db_name: str) -> CustomConnection:
    db_path = Path(db_path) / db_name
    return CustomConnection(db_path)


@suppress_user_warning
def read_sql_table(
    conn: Connection,
    table_name: str,
    columns: Union[str, List[str], ndarray] = None,
    index_col: str = None,
) -> DataFrame:
    validate_table_name(table_name)
    if columns is None:
        query = f'SELECT * FROM "{table_name}";'
    elif isinstance(columns, (list, ndarray)):
        query = f'SELECT {", ".join(columns)} FROM "{table_name}";'
    elif isinstance(columns, str):
        query = f'SELECT {columns} FROM "{table_name}";'
    else:
        raise ValueError("Invalid column specification")

    return pd.read_sql(query, conn, index_col=index_col if index_col else None)


def read_sql_tables(
    conn: Connection,
    table_names: Iterable[str],
    columns: Union[str, List[str], ndarray] = None,
    index_col: str = "index",
) -> List[DataFrame]:
    return [read_sql_table(conn, name, columns, index_col) for name in table_names]


@suppress_user_warning
def transfer_sql_table(
    src_conn: Connection,
    dst_conn: Connection,
    table_name: str,
    if_exists: str = "fail",
    index_col: str = "index",
) -> None:
    validate_table_name(table_name)
    query = f'SELECT * FROM "{table_name}";'
    table = pd.read_sql(query, src_conn, index_col=index_col)
    table.to_sql(table_name, dst_conn, if_exists=if_exists, index=False)


@suppress_user_warning
def read_sql_table_with_types(
    conn: Connection,
    table_name: str,
    column_types: Optional[Dict[str, str]] = None,
    columns: Optional[Union[str, Iterable[str]]] = None,
    index_col: Optional[str] = None,
) -> DataFrame:
    validate_table_name(table_name)
    df_info = pd.read_sql(f'SELECT * FROM "{table_name}" LIMIT 0', conn)
    table_columns = df_info.columns.tolist()

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        missing_columns = [col for col in columns if col not in table_columns]
        if missing_columns:
            raise ArgumentValueError(
                f"Columns {missing_columns} not found in table {table_name}"
            )
        query = f'SELECT {", ".join(columns)} FROM "{table_name}";'
    else:
        query = f'SELECT * FROM "{table_name}";'

    if column_types:
        invalid_columns = [col for col in column_types if col not in table_columns]
        if invalid_columns:
            raise ArgumentValueError(
                f"Columns {invalid_columns} not found in table {table_name}"
            )
        column_types = {
            col: dtype if dtype != "date" else "datetime64[ns]"
            for col, dtype in column_types.items()
        }

    df = pd.read_sql(
        query, conn, index_col=index_col if index_col else None, coerce_float=True
    )

    if column_types:
        for col, dtype in column_types.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype, errors="ignore")
                except TypeError:
                    raise ArgumentValueError(
                        f"Invalid dtype specification={dtype} for column={col}"
                    )

    return df


def read_sql_tables_with_types(
    conn: Connection,
    table_names: Iterable[str],
    column_types: Dict[str, Dict[str, str]] = None,
    columns: Union[str, List[str], ndarray] = None,
    index_col: str = "index",
) -> List[DataFrame]:
    return [
        read_sql_table_with_types(
            conn,
            name,
            column_types.get(name, {}) if column_types else None,
            columns,
            index_col,
        )
        for name in table_names
    ]
