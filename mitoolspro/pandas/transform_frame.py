from typing import Any, Dict, List, Union

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from mitoolspro.exceptions import ArgumentValueError


def reshape_country_indicators(
    data: DataFrame,
    country: str,
    indicator_column: str,
    country_column: str,
    region_column: str,
    year_column: str,
    aggregation_function: str = "first",
) -> DataFrame:
    """
    Reshapes data for a specific country by aggregating and pivoting regional indicators over years.

    Args:
        data (DataFrame): The input DataFrame containing country data.
        country (str): The name of the country to filter data for.
        indicator_column (str): The column containing indicator values (e.g., GDP, population).
        country_column (str): The column identifying countries.
        region_column (str): The column identifying regions or sub-regions within countries.
        year_column (str): The column representing years.
        agg_func (str): The aggregation function to apply to the indicators (default is 'first').

    Returns:
        DataFrame: A pivoted DataFrame with `year_column` as the index,
                   `region_column` values as columns, and aggregated indicators.
    """
    return reshape_group_data(
        dataframe=data,
        filter_value=country,
        value_column=indicator_column,
        group_column=country_column,
        subgroup_column=region_column,
        time_column=year_column,
        agg_func=aggregation_function,
    )


def reshape_group_data(
    dataframe: DataFrame,
    filter_value: str,
    value_column: str,
    group_column: str,
    subgroup_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {value_column, group_column, subgroup_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    filtered_data = dataframe.query(f"{group_column} == @filter_value")
    if filtered_data.empty:
        raise ArgumentValueError(
            f"No data found for group '{filter_value}' in column '{group_column}'."
        )
    grouped_data = (
        filtered_data.groupby(by=[time_column, group_column, subgroup_column])[
            [value_column]
        ]
        .agg(agg_func)
        .reset_index()
    )
    pivoted_data = grouped_data.pivot(
        index=time_column, columns=subgroup_column, values=value_column
    )
    all_times = dataframe[time_column].unique()
    pivoted_data = pivoted_data.reindex(all_times, fill_value=None)
    pivoted_data = pivoted_data.sort_index()
    pivoted_data = pivoted_data.sort_index(axis=1)
    pivoted_data.index.name = filter_value
    return pivoted_data


def reshape_countries_indicators(
    data: DataFrame,
    country_column: str,
    indicator_column: str,
    region_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    """
    Reshapes data for countries by aggregating and pivoting regional indicators over years.

    Args:
        data (DataFrame): The input DataFrame.
        country_column (str): The column identifying the country.
        indicator_column (str): The column containing the indicator values to aggregate.
        region_column (str): The column identifying regions within each country.
        time_column (str): The column representing time.
        agg_func (str): Aggregation function to apply (default is "first").

    Returns:
        DataFrame: A multi-index DataFrame with countries as primary columns
                   and industries as secondary columns.
    """
    return reshape_groups_subgroups(
        dataframe=data,
        group_column=country_column,
        value_column=indicator_column,
        subgroup_column=region_column,
        time_column=time_column,
        agg_func=agg_func,
    )


def reshape_groups_subgroups(
    dataframe: DataFrame,
    group_column: str,
    value_column: str,
    subgroup_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {value_column, group_column, subgroup_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    groups_subgroups: Dict[str, DataFrame] = {}
    for group in tqdm(dataframe[group_column].unique(), desc="Processing groups"):
        try:
            groups_subgroups[group] = reshape_group_data(
                dataframe=dataframe,
                filter_value=group,
                value_column=value_column,
                group_column=group_column,
                subgroup_column=subgroup_column,
                time_column=time_column,
                agg_func=agg_func,
            )
        except ArgumentValueError as e:
            raise ArgumentValueError(f"Error processing group '{group}': {str(e)}")
    for group, subgroups in groups_subgroups.items():
        subgroups.columns = pd.MultiIndex.from_product([[group], subgroups.columns])
    try:
        combined_groups = pd.concat(groups_subgroups.values(), axis=1)
    except ValueError as e:
        raise ArgumentValueError(f"Error concatenating groups: {str(e)}")
    combined_groups = combined_groups.sort_index()
    combined_groups = combined_groups.sort_index(axis=1)
    combined_groups.columns.names = [group_column, subgroup_column]
    return combined_groups


def get_entity_data(
    dataframe: DataFrame,
    data_columns: List[str],
    entity: str,
    entity_column: str,
    time_column: str,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {*data_columns, entity_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    filtered_data = dataframe.query(f"{entity_column} == @entity")
    if filtered_data.empty:
        raise ArgumentValueError(
            f"No data found for entity '{entity}' in column '{entity_column}'."
        )
    grouped_data = (
        filtered_data.groupby(by=[time_column])[data_columns]
        .agg(agg_func)
        .reset_index()
    )
    grouped_data = grouped_data.set_index(time_column)
    all_times = dataframe[time_column].unique()
    grouped_data = grouped_data.reindex(all_times, fill_value=None)
    grouped_data = grouped_data.sort_index()
    grouped_data = grouped_data.sort_index(axis=1)
    grouped_data.index.name = entity
    return grouped_data


def get_entities_data(
    dataframe: DataFrame,
    data_columns: List[str],
    entity_column: str,
    time_column: str,
    entities: List[str] = None,
    agg_func: str = "first",
) -> DataFrame:
    required_columns = {*data_columns, entity_column, time_column}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    entities = entities or dataframe[entity_column].unique()
    entities_data = {}
    for entity in tqdm(entities, desc="Processing entities"):
        try:
            entities_data[entity] = get_entity_data(
                dataframe=dataframe,
                data_columns=data_columns,
                entity=entity,
                entity_column=entity_column,
                time_column=time_column,
                agg_func=agg_func,
            )
            entities_data[entity].columns = pd.MultiIndex.from_product(
                [[entity], entities_data[entity].columns]
            )
        except ArgumentValueError as e:
            raise ArgumentValueError(f"Error processing entity '{entity}': {str(e)}")
    try:
        combined_data = pd.concat(entities_data.values(), axis=1)
    except ValueError as e:
        raise ArgumentValueError(f"Error concatenating entities: {str(e)}")
    combined_data = combined_data.sort_index()
    combined_data = combined_data.sort_index(axis=1)
    combined_data.columns.names = [entity_column, "indicator"]
    combined_data.index.names = [time_column]
    return combined_data


def wide_to_long_dataframe(
    dataframe: DataFrame,
    index: Union[str, List[str]],
    columns: Union[str, List[str]],
    values: Union[str, List[str]] = None,
    filter_index: Dict = None,
    filter_columns: Dict = None,
    agg_func: str = "first",
    fill_value: Any = None,
) -> DataFrame:
    index = [index] if isinstance(index, str) else index
    columns = [columns] if isinstance(columns, str) else columns
    values = [values] if isinstance(values, str) else values
    required_columns = (
        {*index, *columns, *values} if values is not None else {*index, *columns}
    )
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    if filter_index is not None and any(
        key not in dataframe.columns for key in filter_index.keys()
    ):
        missing_columns = [
            key for key in filter_index.keys() if key not in dataframe.columns
        ]
        raise ArgumentValueError(
            f"Columns to filter {missing_columns} not found in the DataFrame."
        )
    if filter_columns is not None and any(
        key not in dataframe.columns for key in filter_columns.keys()
    ):
        missing_columns = [
            key for key in filter_columns.keys() if key not in dataframe.columns
        ]
        raise ArgumentValueError(
            f"Columns to filter {missing_columns} not found in the DataFrame."
        )
    if filter_index:
        for key, value in filter_index.items():
            if key in dataframe.columns:
                filter_values = value if isinstance(value, list) else [value]
                dataframe = dataframe[dataframe[key].isin(filter_values)]
    if filter_columns:
        for key, value in filter_columns.items():
            if key in dataframe.columns:
                filter_values = value if isinstance(value, list) else [value]
                dataframe = dataframe[dataframe[key].isin(filter_values)]
    try:
        wide_dataframe = dataframe.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=agg_func,
            fill_value=fill_value,
        )
    except ValueError as e:
        raise ArgumentValueError(f"Error pivoting DataFrame: {str(e)}")
    return wide_dataframe


def long_to_wide_dataframe(
    dataframe: DataFrame,
    id_vars: Union[str, List[str]],
    value_vars: Union[str, List[str]] = None,
    var_name: str = "variable",
    value_name: str = "value",
    filter_id_vars: Dict = None,
    filter_value_vars: Dict = None,
) -> DataFrame:
    id_vars = [id_vars] if isinstance(id_vars, str) else id_vars
    if value_vars is not None:
        value_vars = [value_vars] if isinstance(value_vars, str) else value_vars
    required_columns = {
        *id_vars,
        *(value_vars or dataframe.columns.difference(id_vars)),
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ArgumentValueError(
            f"Columns {missing_columns} not found in the DataFrame."
        )
    if filter_id_vars:
        for key, value in filter_id_vars.items():
            if key in dataframe.columns:
                filter_values = value if isinstance(value, list) else [value]
                dataframe = dataframe[dataframe[key].isin(filter_values)]
    if filter_value_vars and value_vars:
        for key, value in filter_value_vars.items():
            if key in value_vars:
                filter_values = value if isinstance(value, list) else [value]
                dataframe = dataframe[dataframe[key].isin(filter_values)]
    long_dataframe = pd.melt(
        dataframe,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    return long_dataframe
