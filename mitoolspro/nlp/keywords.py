from typing import List, Optional

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objects import Sankey

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.pandas_utils import idxslice
from mitoolspro.utils.decorators import validate_dataframe_structure
from mitoolspro.utils.validation_templates.sankey import sankey_plot_validation

PLAIN_GRAY_COLOR = [193 / 255.0, 193 / 255.0, 193 / 255.0, 1.0]


class SankeyNode:
    def __init__(self, name: str, count: float, period: int, rank: int):
        self.name = name
        self.count = count
        self.period = period
        self.rank = rank
        self.id: Optional[int] = None
        self.x_pos: Optional[float] = None
        self.y_pos: Optional[float] = None
        self.color: Optional[str] = None

    def __str__(self):
        return f"SankeyNode: {self.name} ({self.count})"


class SankeySinkNode:
    def __init__(self, from_period: int):
        self.name = ""
        self.count = 1e-5
        self.period = from_period
        self.id = Optional[int] = None
        self.x_pos: Optional[float] = None
        self.y_pos: Optional[float] = None
        self.color: list[float] = PLAIN_GRAY_COLOR

    def __str__(self):
        return f"SankeySinkNode: {self.name} ({self.count})"


class SankeyColumn:
    def __init__(
        self, name: str, period: int, nodes: Optional[List[SankeyNode]] = None
    ):
        self.name = name
        self.period = period
        self.nodes: List[SankeyNode] = nodes if nodes else []

    def __str__(self):
        return f"SankeyColumn: {self.name} ({self.period})"

    def add_node(self, gram: str, count: float, period: int, rank: int):
        self.nodes.append(SankeyNode(gram, count, period, rank))

    def get_node(self, name: str) -> Optional[SankeyNode]:
        return next((n for n in self.nodes if n.name == name), None)

    def normalize_y_positions(self, ascending: bool = True):
        ranks = np.asarray([node.rank for node in self.nodes])
        max_rank = np.max(ranks)
        min_rank = np.min(ranks)
        if min_rank == max_rank:
            positions = np.zeros_like(ranks) if ascending else np.ones_like(ranks)
        else:
            positions = (ranks - min_rank) / (max_rank - min_rank)
            positions = positions if ascending else 1 - positions
        self.set_y_positions(positions)

    def set_x_positions(self, x_position: float):
        for node in self.nodes:
            node.x_pos = x_position

    def set_y_positions(self, y_positions: List[float]):
        for node, y_pos in zip(self.nodes, y_positions):
            node.y_pos = y_pos
   



class SankeyLink:
    def __init__(self, source: SankeyNode, target: SankeyNode, value: float):
        if source.period == target.period:
            raise ArgumentValueError("Source and target cannot be in the same period")
        if value <= 0:
            raise ArgumentValueError("Value must be greater than 0")
        self.source = source
        self.target = target
        self.value = value
        self.color: Optional[str] = None


class SankeyDiagram:
    def __init__(
        self,
        columns: Optional[List[SankeyColumn]] = None,
        links: Optional[List[SankeyLink]] = None,
    ):
        self.columns = {}
        self.column_order = []
        self.links = []
        if columns:
            self.add_columns(columns)
        if links:
            self.add_links(links)
        self.sink_nodes = {}
        self.sink_links = []

    def add_column(self, column: SankeyColumn):
        self.columns[column.period] = column
        if column.period not in self.column_order:
            self.column_order.append(column.period)
            self.column_order.sort()  # Keep periods in order

    def get_column_by_position(self, position: int) -> SankeyColumn:
        if position < 0 or position >= len(self.column_order):
            raise ArgumentValueError(f"Position {position} out of range")
        period = self.column_order[position]
        return self.columns[period]

    def get_column_position(self, period: int) -> int:
        if period not in self.column_order:
            raise ArgumentValueError(f"Period {period} not found")
        return self.column_order.index(period)

    def add_link(self, link: SankeyLink):
        if link.source.period not in self.columns:
            raise ArgumentValueError(f"Source {link.source.name} not in columns")
        if link.target.period not in self.columns:
            raise ArgumentValueError(f"Target {link.target.name} not in columns")
        if link.source not in self.columns[link.source.period].nodes:
            raise ArgumentValueError(f"Source {link.source.name} not in nodes")
        if link.target not in self.columns[link.target.period].nodes:
            raise ArgumentValueError(f"Target {link.target.name} not in nodes")
        self.links.append(link)

    def add_columns(self, columns: List[SankeyColumn]):
        for column in columns:
            self.add_column(column)

    def add_links(self, links: List[SankeyLink]):
        for link in links:
            self.add_link(link)

    def connect_columns(self):
        periods = list(self.columns.keys())
        for i in range(len(periods) - 1):
            self._connect_column_pair(
                self.columns[periods[i]], self.columns[periods[i + 1]]
            )

    def _connect_column_pair(self, col1: SankeyColumn, col2: SankeyColumn):
        target_names = {node.name for node in col2.nodes}
        for node in col1.nodes:
            if node.name in target_names:
                match = col2.get_node(node.name)
                if match:
                    self.links.append(
                        SankeyLink(source=node, target=match, value=node.count)
                    )
        if self._columns_require_sink(col1, col2):
            between_period = (col1.period + col2.period) / 2
            self.sink_nodes[between_period] = SankeySinkNode(
                from_period=col1.period, to_period=col2.period
            )
            for node in col1.nodes:
                if node.name not in target_names:
                    self.sink_links.append(
                        SankeyLink(
                            source=node,
                            target=self.sink_nodes[between_period],
                            value=node.count,
                        )
                    )
            for node in col2.nodes:
                if node.name not in target_names:
                    self.sink_links.append(
                        SankeyLink(
                            source=node,
                            target=self.sink_nodes[between_period],
                            value=node.count,
                        )
                    )

    def _columns_require_sink(self, col1: SankeyColumn, col2: SankeyColumn) -> bool:
        case1 = any(
            node.name not in {n.name for n in col2.nodes} for node in col1.nodes
        )
        case2 = any(
            node.name not in {n.name for n in col1.nodes} for node in col2.nodes
        )
        return case1 or case2

    def assign_node_ids(self):
        all_nodes = [node for col in self.columns.values() for node in col.nodes]
        all_nodes.extend([node for node in self.sink_nodes.values()])
        for idx, node in enumerate(all_nodes):
            node.id = idx

    def normalize_positions(self):
        # TODO: Allow for different length columns
        periods = list(self.columns.keys())
        for i, col in enumerate(periods):
            max_rank = max((node.rank for node in self.columns[col].nodes), default=1)
            self.columns[col].normalize_positions(
                max_rank=max_rank, x_pos=i / (len(periods) - 1)
            )
        sink_periods = list(self.sink_nodes.keys())
        for i, period in enumerate(sink_periods):
                self.sink_nodes[col].x_pos = 

    def normalize_columns_counts(self):
        pass 

    def update(self):
        # TODO: Keeps all data structures up to date
        pass

    def render(self, width: int = 1500, height: int = 500) -> go.Figure:
        self.update()
        self.assign_node_ids()
        self.normalize_positions()

        all_nodes = [node for col in self.columns.values() for node in col.nodes]
        all_nodes.extend([node for node in self.sink_nodes.values()])
        label = [node.name for node in all_nodes]
        label.extend([node.name for node in self.sink_nodes.values()])
        x = [node.x_pos for node in all_nodes]
        x.extend([node.x_pos for node in self.sink_nodes.values()])
        y = [node.y_pos for node in all_nodes]
        y.extend([node.y_pos for node in self.sink_nodes.values()])

        source = [link.source.id for link in self.links]
        target = [link.target.id for link in self.links]
        value = [link.value for link in self.links]

        sankey_data = go.Sankey(
            node=dict(label=label, x=x, y=y, pad=20, thickness=20),
            link=dict(source=source, target=target, value=value),
            arrangement="fixed",
        )
        fig = go.Figure(sankey_data)
        fig.update_layout(width=width, height=height, font_size=12)
        return fig


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
