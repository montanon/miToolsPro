from typing import Any, Iterable, List, Literal, Union

import pandas as pd
from pandas import DataFrame
from pandas._libs.tslibs.parsing import DateParseError

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError

INT_COL_ERROR = "Value or values in any of columns={} cannnot be converted into int."
BOOL_COL_ERROR = "Value or values in any of columns={} cannnot be converted into bool."
NON_DATE_COL_ERROR = (
    "Column {} has values that cannot be converted to datetime objects."
)


def validate_columns(
    dataframe: DataFrame, columns: Union[Iterable[str], str]
) -> Iterable[str]:
    columns = [columns] if isinstance(columns, str) else columns
    if not isinstance(columns, Iterable) or not all(
        isinstance(c, str) for c in columns
    ):
        raise ArgumentTypeError(
            "Argument 'cols' must be a string or an iterable of strings."
        )
    missing_cols = [col for col in columns if col not in dataframe.columns]
    if missing_cols:
        raise ArgumentValueError(f"Columns {missing_cols} not found in DataFrame.")
    return columns


def prepare_int_columns(
    dataframe: DataFrame,
    columns: Union[Iterable[str], str],
    nan_placeholder: int,
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
) -> DataFrame:
    if errors not in ["raise", "coerce", "ignore"]:
        raise ArgumentValueError(
            "Argument 'errors' must be one of ['raise', 'coerce', 'ignore']."
        )
    columns = validate_columns(dataframe, columns)
    try:
        for col in columns:
            dataframe[col] = pd.to_numeric(
                dataframe[col], errors=errors, downcast="integer"
            )
            if errors != "ignore":
                dataframe[col] = dataframe[col].fillna(nan_placeholder)
                dataframe[col] = dataframe[col].astype(int)
    except (ValueError, KeyError) as e:
        raise ArgumentTypeError(f"{INT_COL_ERROR.format(col)}, Details: {str(e)}")
    return dataframe


def prepare_categorical_columns(
    dataframe: DataFrame,
    columns: Union[Iterable[str], str],
    categories: List[str] = None,
    ordered: bool = False,
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    for col in columns:
        dataframe[col] = pd.Categorical(
            dataframe[col], categories=categories, ordered=ordered
        )
    return dataframe


def prepare_rank_columns(
    dataframe: DataFrame,
    columns: Union[str, List[str]],
    method: Literal["average", "min", "max", "first", "dense"] = "average",
    ascending: bool = True,
) -> DataFrame:
    if method not in ["average", "min", "max", "first", "dense"]:
        raise ArgumentValueError(
            f"Argument 'method'={method} must be one of ['average', 'min', 'max', 'first', 'dense']."
        )
    columns = validate_columns(dataframe, columns)
    for col in columns:
        dataframe[col] = dataframe[col].rank(method=method, ascending=ascending)
    return dataframe


def prepare_standardized_columns(
    dataframe: DataFrame, columns: Union[str, List[str]]
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    for col in columns:
        dataframe[col] = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()
    return dataframe


def prepare_normalized_columns(
    dataframe: DataFrame,
    columns: Union[str, List[str]],
    range_min: float = 0.0,
    range_max: float = 1.0,
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    for col in columns:
        min_val = dataframe[col].min()
        max_val = dataframe[col].max()
        values_range = max_val - min_val if max_val != min_val else 1
        dataframe[col] = (dataframe[col] - min_val) / values_range * (
            range_max - range_min
        ) + range_min
    return dataframe


def prepare_bin_columns(
    dataframe: DataFrame,
    columns: Union[str, List[str]],
    bins: Union[int, List[float]] = 10,
    labels: List[Any] = None,
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    if labels is not None:
        n_bins = bins if isinstance(bins, int) else len(bins) - 1
        if len(labels) != n_bins:
            raise ArgumentValueError(
                f"Length of 'labels': {len(labels)} must be equal to amount of 'bins': {n_bins}."
            )
    try:
        for col in columns:
            dataframe[col] = pd.cut(dataframe[col], bins=bins, labels=labels)
    except TypeError:
        raise ArgumentTypeError(f"'column'={col} must be of numeric type.")
    return dataframe


def prepare_quantile_columns(
    dataframe: DataFrame,
    columns: Union[str, List[str]],
    quantiles: int = 10,
    labels: List[Any] = None,
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    if not isinstance(quantiles, int) or quantiles < 2:
        raise ArgumentValueError(
            f"Argument 'quantiles'={quantiles} must be an int greater than 1."
        )
    if labels is not None and len(labels) != quantiles:
        raise ArgumentValueError(
            f"Length of 'labels': {len(labels)} must be equal to 'quantiles'={quantiles}."
        )
    for col in columns:
        dataframe[col] = pd.qcut(dataframe[col], q=quantiles, labels=labels)
    return dataframe


def prepare_str_columns(
    dataframe: DataFrame, columns: Union[Iterable[str], str]
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    dataframe[columns] = dataframe[columns].astype(str)
    return dataframe


def prepare_date_columns(
    dataframe: DataFrame,
    columns: Union[Iterable[str], str],
    nan_placeholder: Union[str, pd.Timestamp],
    errors: Literal["raise", "coerce", "ignore"] = "coerce",
    date_format: str = None,
) -> DataFrame:
    if errors not in ["raise", "coerce", "ignore"]:
        raise ArgumentValueError(
            "Argument 'errors' must be one of ['raise', 'coerce', 'ignore']."
        )
    columns = validate_columns(dataframe, columns)
    try:
        for col in columns:
            dataframe[col] = pd.to_datetime(
                dataframe[col], errors=errors, format=date_format
            )
            if errors != "ignore" and nan_placeholder is not None:
                dataframe[col] = dataframe[col].fillna(pd.to_datetime(nan_placeholder))
    except (ValueError, DateParseError) as e:
        raise ArgumentTypeError(f"{NON_DATE_COL_ERROR.format(col)}, Details: {str(e)}")
    return dataframe


def prepare_bool_columns(
    dataframe: DataFrame,
    columns: Union[Iterable[str], str],
    nan_placeholder: bool = False,
) -> DataFrame:
    columns = validate_columns(dataframe, columns)
    try:
        for col in columns:
            dataframe[col] = dataframe[col].fillna(nan_placeholder)
            dataframe[col] = dataframe[col].astype(bool)
    except Exception as e:
        raise ArgumentTypeError(
            f"{BOOL_COL_ERROR.format(col)}: {columns}. Details: {str(e)}"
        )
    return dataframe
