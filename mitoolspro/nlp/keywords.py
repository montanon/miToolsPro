from typing import List, Optional

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objects import Sankey

from mitoolspro.pandas_utils import idxslice
from mitoolspro.utils.decorators import validate_dataframe_structure
from mitoolspro.utils.validation_templates.sankey import sankey_plot_validation


def get_yearly_ranges_ngram(
    yearly_ranges_ngrams: DataFrame, n_gram: str, max_ngram: int
) -> DataFrame:
    yearly_ranges_ngram = yearly_ranges_ngrams.loc[
        :, idxslice(yearly_ranges_ngrams, "n-gram", n_gram, axis=1)
    ]
    yearly_ranges_ngram = yearly_ranges_ngram.iloc[:max_ngram, :]
    return yearly_ranges_ngram


def create_grams_data(
    yearly_ranges_ngram: DataFrame, n_periods: int, max_ngram: int
) -> DataFrame:
    grams_data = []
    for time_range, time_ngrams in yearly_ranges_ngram.groupby("year_range", axis=1):
        range_grams = time_ngrams.iloc[:, [0]]
        range_grams.columns = range_grams.columns.droplevel([0, 1])
        range_grams["period"] = time_range
        grams_data.append(range_grams)
    grams_data = pd.concat(grams_data, axis=0).reset_index(drop=True)
    grams_data["x_pos"] = [pos for pos in range(n_periods) for _ in range(max_ngram)]
    grams_data.loc[grams_data["x_pos"] == n_periods - 1, "x_pos"] += 0.25 * (
        len(grams_data.iloc[0, 0].split(" ")) - 1
    )  # Heuristic for wider last period
    grams_data["y_pos"] = [pos for _ in range(n_periods) for pos in range(max_ngram)]
    return grams_data


def update_out_sources(
    grams_data: DataFrame, periods: List, max_ngram: int
) -> DataFrame:
    out_sources = {period: False for period in periods[:-1]}
    for n, period in enumerate(periods[:-1]):
        next_period = periods[n + 1] if n != len(periods) - 1 else None
        if not out_sources[period]:
            period_grams = grams_data.loc[grams_data["period"] == period, "Gram"]
            next_comparison = (
                next_period
                and (
                    ~period_grams.isin(
                        grams_data.loc[grams_data["period"] == next_period, "Gram"]
                    )
                ).any()
            )
            current_comparison = (
                period
                and (
                    ~period_grams.isin(
                        grams_data.loc[grams_data["period"] == period, "Gram"]
                    )
                ).any()
            )
            out_sources[period] = next_comparison or current_comparison
    out_x_pos = [
        (n + 1 - 0.5) if source else None
        for n, source in enumerate(out_sources.values())
    ]
    out_y_pos = [
        max(grams_data["y_pos"]) + 3 if source else None
        for source in out_sources.values()
    ]
    out_data = pd.DataFrame(
        {
            "Gram": ["" for _ in out_x_pos],
            "period": list(out_sources.keys()),
            "x_pos": out_x_pos,
            "y_pos": out_y_pos,
        }
    ).dropna()
    grams_data = pd.concat([grams_data, out_data], axis=0).reset_index(drop=True)
    return grams_data


def update_periods_links(
    yearly_ranges_ngram: DataFrame, grams_data: DataFrame, periods: List, n_gram: str
) -> DataFrame:
    sources, targets, values = (
        {k: [] for k in periods},
        {k: [] for k in periods},
        {k: [] for k in periods},
    )
    for n, period in enumerate(periods):
        period_grams = yearly_ranges_ngram[period]
        if n != len(periods) - 1:
            next_period = periods[n + 1]
            next_grams = yearly_ranges_ngram[next_period]
            for _, (gram, value) in period_grams.iterrows():
                sources[period].append(gram)
                values[period].append(value)
                if gram in next_grams[(n_gram, "Gram")].values:
                    targets[period].append(gram)
                else:
                    targets[period].append("")
        if n != 0:
            previous_period = periods[n - 1]
            previous_grams = yearly_ranges_ngram[previous_period]
            for _, (gram, value) in period_grams.iterrows():
                if gram not in previous_grams[(n_gram, "Gram")].values:
                    sources[previous_period].append("")
                    targets[previous_period].append(gram)
                    values[previous_period].append(value)
    periods_links = []
    for period in sources:
        period_links = pd.DataFrame(
            {
                "sources": sources[period],
                "targets": targets[period],
                "values": values[period],
            }
        )
        period_links["period"] = period
        periods_links.append(period_links)
    periods_links = pd.concat(periods_links).reset_index(drop=True)
    periods_links["sources_id"] = np.nan
    periods_links["targets_id"] = np.nan
    for n, (
        source,
        target,
        value,
        period,
        source_id,
        targets_id,
    ) in periods_links.iterrows():
        source_index = grams_data.loc[
            (grams_data["Gram"] == source) & (grams_data["period"] == period)
        ].index.values[0]
        if target != "":
            next_period = periods[periods.get_loc(period) + 1]
            gram_is_target = grams_data["Gram"] == target
            target_index = grams_data.loc[
                gram_is_target & (grams_data["period"] == next_period)
            ].index.values[0]
        elif target != " ":
            gram_is_target = grams_data["Gram"] == target
            target_index = grams_data.loc[
                gram_is_target & (grams_data["period"] == period)
            ].index.values[0]
        else:
            pass
        periods_links.at[n, "sources_id"] = source_index
        periods_links.at[n, "targets_id"] = target_index
    return periods_links


def update_grams_data(grams_data: DataFrame) -> DataFrame:
    grams_data["x_pos"] = grams_data["x_pos"] / grams_data["x_pos"].max()
    grams_data["x_pos"] = grams_data["x_pos"].clip(0.001, 0.999)
    grams_data["y_pos"] = grams_data["y_pos"] / grams_data["y_pos"].max()
    grams_data["y_pos"] = grams_data["y_pos"].clip(0.001, 0.999)
    return grams_data


def create_sankey_data(
    periods_links: DataFrame,
    grams_data: DataFrame,
    periods: List,
    width: Optional[int] = 1500,
    height: Optional[int] = 500,
) -> Sankey:
    sankey_nodes = {
        "label": grams_data["Gram"].values.tolist(),
        "x": grams_data["x_pos"].values.tolist(),
        "y": grams_data["y_pos"].values.tolist(),
        "pad": 20,
        "thickness": 20,
    }
    sankey_links = {
        "source": periods_links["sources_id"].values.tolist(),
        "target": periods_links["targets_id"].values.tolist(),
        "value": periods_links["values"].values.tolist(),
    }
    label_names = sorted(list(set(grams_data["Gram"].values.tolist())))
    colors = mpl.colormaps["Spectral_r"](np.linspace(0, 1, len(label_names)))
    labels_colors = {w: c for w, c in zip(label_names, colors)}
    PLAIN_GRAY_COLOR = [193 / 255.0, 193 / 255.0, 193 / 255.0, 1.0]
    labels_colors[""] = np.array(PLAIN_GRAY_COLOR)
    nodes_colors = [labels_colors[l] for l in grams_data["Gram"]]
    nodes_colors = [f"rgba({c[0]},{c[1]},{c[2]},{c[3]})" for c in nodes_colors]
    color_sources = periods_links.copy(True)
    color_sources["color_labels"] = color_sources.apply(
        lambda x: x["sources"] if x["sources"] != "" else x["targets"], axis=1
    )
    links_colors = [labels_colors[l] for l in color_sources["color_labels"]]
    links_colors = [f"rgba({c[0]},{c[1]},{c[2]},{0.5})" for c in links_colors]
    sankey_data = go.Sankey(link=sankey_links, node=sankey_nodes, arrangement="fixed")
    fig = go.Figure(sankey_data)
    fig.update_traces(node_color=nodes_colors, link_color=links_colors)
    period_labels = [f"{'-'.join(period[1:-1].split(', '))}" for period in periods]
    for i, label in enumerate(period_labels):
        x = i / (len(period_labels) - 1)
        fig.add_annotation(
            dict(
                font=dict(color="black", size=14, family="Helvetica, sans-serif"),
                x=x,
                y=1.2,
                showarrow=False,
                text=f"<b>{label}</b>",
            )
        )
    fig.update_layout(width=width, height=height, font_size=12)
    return fig


@validate_dataframe_structure(
    dataframe_name="yearly_ranges_ngrams", validation=sankey_plot_validation
)
def evolution_sankey_plot_clusters_ngrams(
    yearly_ranges_ngrams: DataFrame,
    n_gram: int,
    max_ngram: int,
    year_range_level: str,
    width: Optional[int] = 1500,
    height: Optional[int] = 500,
) -> Sankey:
    periods = yearly_ranges_ngrams.columns.get_level_values(year_range_level).unique()
    n_periods = len(periods)
    n_gram = yearly_ranges_ngrams.columns.get_level_values("n-gram").unique()[
        n_gram - 1
    ]
    yearly_ranges_ngram = (
        get_yearly_ranges_ngram(yearly_ranges_ngrams, n_gram, max_ngram)
        .fillna(0.0)
        .replace(0.0, 1e-5)
    )
    count_columns = [col for col in yearly_ranges_ngram.columns if col[-1] == "Count"]
    for col in count_columns:
        yearly_ranges_ngram[col] = (
            yearly_ranges_ngram[col] / yearly_ranges_ngram[col].sum()
        )
    grams_data = create_grams_data(yearly_ranges_ngram, n_periods, max_ngram)
    grams_data = update_out_sources(grams_data, periods, max_ngram)
    periods_links = update_periods_links(
        yearly_ranges_ngram, grams_data, periods, n_gram
    )
    grams_data = update_grams_data(grams_data)
    fig = create_sankey_data(
        periods_links, grams_data, periods, width=width, height=height
    )
    return fig
