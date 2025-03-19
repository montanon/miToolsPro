import re
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pandas import DataFrame, IndexSlice, MultiIndex

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError


def idxslice(
    df: DataFrame, level: Union[int, str], values: Union[List[Any], Any], axis: int
) -> slice:
    if axis not in {0, 1}:
        raise ArgumentValueError(
            f"Invalid 'axis'={axis}, must be 0 for index or 1 for columns"
        )
    values = [values] if not isinstance(values, list) else values
    idx = df.index if axis == 0 else df.columns
    if isinstance(idx, MultiIndex):
        if isinstance(level, str):
            if level not in idx.names:
                raise ArgumentValueError(
                    f"'level'={level} is not in the MultiIndex names: {idx.names}"
                )
            level = idx.names.index(level)
        elif not isinstance(level, int) or level < 0 or level >= idx.nlevels:
            raise ArgumentValueError(
                f"Provided 'level'={level} is out of bounds for the MultiIndex with {idx.nlevels} levels."
            )
        slices = [slice(None)] * idx.nlevels
        slices[level] = values
        return IndexSlice[tuple(slices)]
    if not isinstance(idx, MultiIndex):
        if isinstance(level, int) and level != 0:
            raise ArgumentValueError(
                "For single-level Index or Columns, level must be 0."
            )
        if isinstance(level, str) and level != idx.name:
            raise ArgumentValueError(
                f"Level '{level}' does not match the Index or Columns name."
            )
        return IndexSlice[values]


def store_dataframe_parquet(
    dataframe: DataFrame,
    base_path: Union[str, PathLike],
    dataframe_name: str,
    overwrite: bool = False,
) -> None:
    base_path = Path(base_path).absolute()
    if not base_path.is_dir():
        raise ArgumentValueError(
            f"'base_path'={base_path} directory not found. It must be a directory."
        )
    index_path = base_path / f"{dataframe_name}_index.parquet"
    columns_path = base_path / f"{dataframe_name}_columns.parquet"
    data_path = base_path / f"{dataframe_name}.parquet"
    if data_path.exists() and not overwrite:
        raise ArgumentValueError(
            f"File {data_path} already exists. Set 'overwrite=True' to overwrite."
        )
    if not data_path.exists() or overwrite:
        if isinstance(dataframe.index, MultiIndex):
            indexes = dataframe.index.to_frame()
            indexes.to_parquet(index_path)
            dataframe = dataframe.reset_index(drop=True)
        if isinstance(dataframe.columns, MultiIndex):
            columns = dataframe.columns.to_frame(index=False)
            columns.to_parquet(columns_path)
            dataframe.columns = range(len(dataframe.columns))
        dataframe.to_parquet(data_path)


def load_dataframe_parquet(
    dataframe: DataFrame, base_path: Union[str, PathLike], dataframe_name: str
) -> DataFrame:
    base_path = Path(base_path).absolute()
    if not base_path.is_dir():
        raise ArgumentValueError(
            f"'base_path'={base_path} directory not found. It must be a directory."
        )
    index_path = base_path / f"{dataframe_name}_index.parquet"
    columns_path = base_path / f"{dataframe_name}_columns.parquet"
    data_path = base_path / f"{dataframe_name}.parquet"
    if not data_path.exists():
        raise ArgumentValueError(f"File {data_path} not found.")
    dataframe = pd.read_parquet(data_path)
    if index_path.exists():
        indexes = pd.read_parquet(index_path)
        dataframe.index = pd.MultiIndex.from_frame(indexes)
    if columns_path.exists():
        columns = pd.read_parquet(columns_path)
        dataframe.columns = pd.MultiIndex.from_frame(columns)
    return dataframe


def dataframe_to_latex(dataframe: DataFrame):
    def regex_symbol_replacement(match):
        return rf"\{match.group(0)}"

    symbols_pattern = r"([\ \_\-\&\%\$\#])"
    table = dataframe.rename(
        columns=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x)
        if isinstance(x, str)
        else str(round(x, 1)),
        index=lambda x: re.sub(symbols_pattern, regex_symbol_replacement, x)
        if isinstance(x, str)
        else str(round(x, 1)),
    ).to_latex(multirow=True, multicolumn=True, multicolumn_format="c")
    table = (
        "\\begin{adjustbox}{width=\\textwidth,center}\n"
        + f"{table}"
        + "\end{adjustbox}\n"
    )
    return table


def check_if_dataframe_sequence(
    data_dir: PathLike,
    name: str,
    sequence_values: Optional[List[Union[str, int]]] = None,
) -> bool:
    sequence_dir = data_dir / name
    if not sequence_dir.exists():
        return False
    if sequence_values is not None:
        try:
            sequence_files = [
                int(file.stem.split("_")[-1])
                if file.stem.split("_")[-1].isdigit()
                else file.stem.split("_")[-1]
                for file in sequence_dir.glob("*.parquet")
            ]
        except (ValueError, TypeError, IndexError) as e:
            raise ArgumentValueError(f"Invalid sequence value in filenames: {e}")
        sequence_files = sequence_dir.glob("*.parquet")
        sequence_files = [int(file.stem.split("_")[-1]) for file in sequence_files]
        return set(sequence_values) == set(sequence_files)
    return False


def store_dataframe_sequence(
    dataframes: Dict[Union[str, int], DataFrame], name: str, data_dir: PathLike
) -> None:
    sequence_dir = data_dir / name
    if not all(isinstance(df, DataFrame) for df in dataframes.values()):
        raise ValueError("All values in 'dataframes' must be pandas DataFrames")
    try:
        sequence_dir.mkdir(exist_ok=True, parents=True)
        for seq_val, dataframe in dataframes.items():
            seq_val_name = f"{name}_{seq_val}".replace(" ", "")
            filepath = sequence_dir / f"{seq_val_name}.parquet"
            dataframe.to_parquet(filepath)
        if not check_if_dataframe_sequence(data_dir, name, list(dataframes.keys())):
            raise IOError(f"Failed to store all DataFrames for '{name}' sequence")
    except (IOError, OSError) as e:
        raise IOError(f"Error storing DataFrame sequence: {e}")


def load_dataframe_sequence(
    data_dir: PathLike,
    name: str,
    sequence_values: Optional[List[Union[str, int]]] = None,
) -> Dict[Union[str, int], DataFrame]:
    sequence_dir = data_dir / name
    if sequence_values and not check_if_dataframe_sequence(
        data_dir, name, sequence_values
    ):
        raise ArgumentValueError(
            f"Sequence '{name}' is missing required values: {sequence_values}"
        )
    sequence_files = sequence_dir.glob("*.parquet")
    dataframes = {}
    for file in sequence_files:
        try:
            seq_value = file.stem.split("_")[-1]
            seq_value = int(seq_value) if seq_value.isdigit() else seq_value
            if sequence_values is None or seq_value in sequence_values:
                dataframes[seq_value] = pd.read_parquet(file)
        except (ValueError, TypeError, IndexError) as e:
            raise ArgumentValueError(
                f"Invalid sequence value in file: {file.name}"
            ) from e
    if not dataframes:
        raise ArgumentValueError(
            f"No dataframes were loaded from the provided 'sequence_values={sequence_values}'"
        )
    return dataframes


def select_index(
    dataframe: DataFrame,
    index: Union[str, Tuple, int, List[Union[str, Tuple, int]]],
    level: Optional[Union[str, int]] = None,
) -> DataFrame:
    if level is not None and not isinstance(level, (str, int)):
        raise ArgumentTypeError(
            f"The 'level' must be a string (for named levels) or an integer (for positional levels), not {type(level)}."
        )
    has_multi_index = isinstance(dataframe.index, MultiIndex)
    if level is not None:
        if not has_multi_index:
            raise ArgumentValueError(
                "level can only be specified for DataFrames with multi-index index"
            )
        if isinstance(level, str) and level not in dataframe.index.names:
            raise ArgumentValueError(f"Invalid level name: {level}")
        if isinstance(level, int) and (abs(level) >= dataframe.index.nlevels):
            raise ArgumentValueError(f"Invalid level index: {level}")
        if any(isinstance(col, tuple) for col in index):
            raise ArgumentValueError(
                "Cannot use tuples in index when level is specified"
            )
    index = [index] if isinstance(index, (str, tuple, int)) else index
    if not isinstance(index, list):
        raise ArgumentTypeError(
            "Provided 'index' must be a string, tuple, int, or list."
        )
    if level is not None:
        level_index = (
            level if isinstance(level, int) else dataframe.index.names.index(level)
        )
        index = [idx for idx in dataframe.index if idx[level_index] in index]
        if not index:
            raise ArgumentValueError(f"No 'index' provided are in 'level={level}'!")
    invalid_index = set(index) - set(dataframe.index)
    if invalid_index:
        raise ArgumentValueError(f"Invalid index: {invalid_index}")
    return dataframe.loc[index, :]


def select_columns(
    dataframe: DataFrame,
    columns: Union[str, Tuple, int, List[Union[str, Tuple, int]]],
    level: Optional[Union[str, int]] = None,
) -> DataFrame:
    if level is not None and not isinstance(level, (str, int)):
        raise ArgumentTypeError(
            f"The 'level' must be a string (for named levels) or an integer (for positional levels), not {type(level)}."
        )
    has_multi_index = isinstance(dataframe.columns, MultiIndex)
    if level is not None:
        if not has_multi_index:
            raise ArgumentValueError(
                "level can only be specified for DataFrames with multi-index columns"
            )
        if isinstance(level, str) and level not in dataframe.columns.names:
            raise ArgumentValueError(f"Invalid level name: {level}")
        if isinstance(level, int) and (abs(level) >= dataframe.columns.nlevels):
            raise ArgumentValueError(f"Invalid level index: {level}")
        if any(isinstance(col, tuple) for col in columns):
            raise ArgumentValueError(
                "Cannot use tuples in columns when level is specified"
            )
    columns = [columns] if isinstance(columns, (str, tuple, int)) else columns
    if not isinstance(columns, list):
        raise ArgumentTypeError(
            "Provided 'columns' must be a string, tuple, int, or list."
        )
    if level is not None:
        level_index = (
            level if isinstance(level, int) else dataframe.columns.names.index(level)
        )
        columns = [col for col in dataframe.columns if col[level_index] in columns]
        if not columns:
            raise ArgumentValueError(f"No 'columns' provided are in 'level={level}'!")
    invalid_columns = set(columns) - set(dataframe.columns)
    if invalid_columns:
        raise ArgumentValueError(f"Invalid columns: {invalid_columns}")
    return dataframe.loc[:, columns]
