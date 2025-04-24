import json
from typing import Any, List, Optional, Union

import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mitoolspro.exceptions import ArgumentValueError

PLAIN_GRAY_COLOR = [193 / 255.0, 193 / 255.0, 193 / 255.0, 1.0]


def _scale_array(array: np.ndarray, ascending: bool = True) -> np.ndarray:
    array = array.astype(np.float64)
    array_max = np.max(array)
    array_min = np.min(array)
    if array_max == array_min:
        array = np.full_like(array, 0.001 if ascending else 0.999)
    else:
        array = (array - array_min) / (array_max - array_min)
        array = array * 0.998 + 0.001 if ascending else 1 - array * 0.998 - 0.001
    return array


class SankeyNode:
    def __init__(
        self,
        name: str,
        count: float,
        period: int,
        rank: int,
        color: Optional[str] = None,
    ):
        self.name = name
        self.count = count
        self.period = period
        self.rank = rank
        self.id: Optional[int] = None
        self.x_pos: Optional[float] = None
        self.y_pos: Optional[float] = None
        self.color: str = color

    def __str__(self):
        return f"SankeyNode: {self.name} ({self.count})"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SankeyNode":
        node = SankeyNode(data["name"], data["count"], data["period"], data["rank"])
        node.__dict__.update(data)
        return node


class SankeySinkNode(SankeyNode):
    def __init__(self, period: int):
        super().__init__(name="", count=1e-5, period=period, rank=-1)
        self.color = f"rgba({PLAIN_GRAY_COLOR[0]},{PLAIN_GRAY_COLOR[1]},{PLAIN_GRAY_COLOR[2]},{PLAIN_GRAY_COLOR[3]})"

    def __str__(self):
        return f"SankeySinkNode: {self.name} ({self.count})"


class SankeyLink:
    def __init__(
        self,
        source: SankeyNode,
        target: SankeyNode,
        value: float,
        color: Optional[str] = None,
    ):
        if source.period == target.period:
            raise ArgumentValueError("Source and target cannot be in the same period")
        if value <= 0:
            raise ArgumentValueError("Value must be greater than 0")
        self.source = source
        self.target = target
        self.value = value
        self.color: Optional[str] = color

    def __str__(self):
        return f"SankeyLink: {self.source.period}:{self.source.name} -> {self.target.period}:{self.target.name} ({self.value})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.name,
            "target": self.target.name,
            "source_period": self.source.period,
            "target_period": self.target.period,
            "value": self.value,
            "color": self.color,
        }


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
        positions = self.scale_array(ranks, ascending)
        self.set_y_positions(positions)

    def set_x_positions(self, x_position: float):
        for node in self.nodes:
            node.x_pos = x_position

    def set_y_positions(self, y_positions: List[float]):
        for node, y_pos in zip(self.nodes, y_positions):
            node.y_pos = y_pos

    def x_positions(self) -> List[float]:
        return [node.x_pos for node in self.nodes]

    def x_position(self) -> float:
        return self.x_positions()[0]

    def y_positions(self) -> List[float]:
        return [node.y_pos for node in self.nodes]

    def names(self) -> List[str]:
        return [node.name for node in self.nodes]

    def scale_array(self, array: np.ndarray, ascending: bool = True) -> np.ndarray:
        return _scale_array(array, ascending)


class SankeyPlotter:
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
            self.column_order.sort()

    def get_column_by_index(self, index: int) -> SankeyColumn:
        if abs(index) >= len(self.column_order):
            raise ArgumentValueError(f"Position {index} out of range")
        period = self.column_order[index]
        return self.columns[period]

    def get_column_index(self, period: int) -> int:
        if period not in self.column_order:
            raise ArgumentValueError(f"Period {period} not found")
        return self.column_order.index(period)

    def _is_sink_node(self, node: SankeyNode) -> bool:
        return isinstance(node, SankeySinkNode) or (
            node.name == "" and node.period in self.sink_nodes
        )

    def add_link(self, link: SankeyLink):
        if not self._is_sink_node(link.source):
            if link.source.period not in self.columns:
                raise ArgumentValueError(f"Source {link.source.name} not in columns")
            if link.source not in self.columns[link.source.period].nodes:
                raise ArgumentValueError(f"Source {link.source.name} not in nodes")
        if not self._is_sink_node(link.target):
            if link.target.period not in self.columns:
                raise ArgumentValueError(f"Target {link.target.name} not in columns")
            if link.target not in self.columns[link.target.period].nodes:
                raise ArgumentValueError(f"Target {link.target.name} not in nodes")
        if self._is_sink_node(link.source) or self._is_sink_node(link.target):
            self.sink_links.append(link)
        else:
            self.links.append(link)

    def add_columns(self, columns: List[SankeyColumn]):
        for column in columns:
            self.add_column(column)

    def add_links(self, links: List[SankeyLink]):
        for link in links:
            self.add_link(link)

    def connect_columns(self):
        periods = sorted(self.columns.keys())
        for i in range(len(periods) - 1):
            self._connect_column_pair_no_sink(
                self.columns[periods[i]], self.columns[periods[i + 1]]
            )
        self.create_sink_links()

    def _connect_column_pair_no_sink(self, col1: SankeyColumn, col2: SankeyColumn):
        for node in col1.nodes:
            if node.name in col2.names():
                match = col2.get_node(node.name)
                if match:
                    self.links.append(
                        SankeyLink(source=node, target=match, value=node.count)
                    )

    def create_sink_links(self):
        period_nodes: dict[int, List[SankeyNode]] = {
            col.period: col.nodes for col in self.columns.values()
        }
        sorted_periods = sorted(period_nodes.keys())

        links_by_period = {}
        for link in self.links:
            key = (link.source.period, link.target.period)
            links_by_period.setdefault(key, []).append(link)

        for i in range(len(sorted_periods) - 1):
            period, next_period = sorted_periods[i], sorted_periods[i + 1]
            col1_nodes = period_nodes[period]
            col2_nodes = period_nodes[next_period]

            sources = {
                link.source for link in links_by_period.get((period, next_period), [])
            }
            targets = {
                link.target for link in links_by_period.get((period, next_period), [])
            }

            unlinked_sources = [n for n in col1_nodes if n not in sources]
            unlinked_targets = [n for n in col2_nodes if n not in targets]

            if not unlinked_sources and not unlinked_targets:
                continue

            between = (period + next_period) / 2
            if between not in self.sink_nodes:
                self.sink_nodes[between] = SankeySinkNode(period=between)

            sink = self.sink_nodes[between]

            for node in unlinked_sources:
                self.sink_links.append(
                    SankeyLink(source=node, target=sink, value=node.count)
                )

            for node in unlinked_targets:
                self.sink_links.append(
                    SankeyLink(source=sink, target=node, value=node.count)
                )

    def _connect_column_pair(self, col1: SankeyColumn, col2: SankeyColumn):
        self._connect_column_pair_no_sink(col1, col2)
        if self._columns_require_sink(col1, col2):
            between_period = (col1.period + col2.period) / 2
            self.sink_nodes[between_period] = SankeySinkNode(period=between_period)
            for node in col1.nodes:
                if node.name not in col2.names():
                    self.sink_links.append(
                        SankeyLink(
                            source=node,
                            target=self.sink_nodes[between_period],
                            value=node.count,
                        )
                    )
            for node in col2.nodes:
                if node.name not in col1.names():
                    self.sink_links.append(
                        SankeyLink(
                            source=self.sink_nodes[between_period],
                            target=node,
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

    def normalize_x_positions(self, ascending: bool = True):
        periods = list(self.columns.keys())
        periods.extend(list(self.sink_nodes.keys()))
        max_period = max(periods)
        max_name_length = max(
            len(name.split(" ")) for name in self.get_column_by_index(-1).names()
        )
        last_period_extra = 0.25 * max_name_length
        # Heuristic for wider last period with names to the left
        periods = [
            period + last_period_extra if period == max_period else period
            for period in periods
        ]
        positions = self.scale_array(np.array(periods), ascending)
        for period, x_pos in zip(periods, positions):
            if period in self.columns:
                self.columns[period].set_x_positions(x_pos)
            elif period in self.sink_nodes:
                self.sink_nodes[period].x_pos = x_pos
            # Handle wider last period with names to the left
            elif period - last_period_extra in self.columns:
                self.columns[period - last_period_extra].set_x_positions(x_pos)

    def normalize_positions(self):
        for col in self.columns.values():
            col.normalize_y_positions()
        self.normalize_x_positions()
        if self.sink_nodes:
            for node in self.sink_nodes.values():
                node.y_pos = 0.999 * 1.5

    def scale_array(self, array: np.ndarray, ascending: bool = True) -> np.ndarray:
        return _scale_array(array, ascending)

    def assign_colors(self, color_map: str = "Spectral_r"):
        all_names = sorted(
            {node.name for col in self.columns.values() for node in col.nodes}
        )
        cmap = mpl.colormaps[color_map]
        colors = cmap(np.linspace(0, 1, len(all_names)))
        label_to_color = {name: color for name, color in zip(all_names, colors)}
        label_to_color[""] = np.array(PLAIN_GRAY_COLOR)

        for col in self.columns.values():
            for node in col.nodes:
                if node.color:
                    continue
                rgba = label_to_color.get(node.name, PLAIN_GRAY_COLOR)
                node.color = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]})"
        for node in self.sink_nodes.values():
            if node.color:
                continue
            rgba = label_to_color.get(node.name, PLAIN_GRAY_COLOR)
            node.color = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{rgba[3]})"

        for link in self.links + self.sink_links:
            if link.color:
                continue
            name = link.source.name if link.source.name != "" else link.target.name
            rgba = label_to_color.get(name, PLAIN_GRAY_COLOR)
            link.color = f"rgba({rgba[0]},{rgba[1]},{rgba[2]},{0.5})"

    def update(self):
        self.assign_node_ids()
        self.normalize_positions()
        self.assign_colors()

    def render(
        self, width: int = 1500, height: int = 500, pad: int = 20, thickness: int = 20
    ) -> go.Figure:
        self.update()

        all_nodes = [node for col in self.columns.values() for node in col.nodes]
        all_nodes.extend([node for node in self.sink_nodes.values()])
        label = [node.name for node in all_nodes]

        x = [node.x_pos for node in all_nodes]
        x = self.scale_array(np.array(x))

        y = [node.y_pos for node in all_nodes]
        y = self.scale_array(np.array(y))

        all_links = self.links + self.sink_links

        node_colors = [node.color for node in all_nodes]
        link_colors = [link.color for link in all_links]

        source = [link.source.id for link in all_links]
        target = [link.target.id for link in all_links]
        value = [link.value for link in all_links]

        sankey_data = go.Sankey(
            node=dict(
                label=label, x=x, y=y, pad=pad, thickness=thickness, color=node_colors
            ),
            link=dict(source=source, target=target, value=value, color=link_colors),
            arrangement="fixed",
        )
        fig = go.Figure(sankey_data)
        for col in self.columns.values():
            if len(col.name) > 20:
                column_label = col.name[:17] + "..."
            else:
                column_label = col.name
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=14, family="Helvetica, sans-serif"),
                    x=col.x_position(),
                    y=1.25,
                    showarrow=False,
                    xanchor="center",
                    text=f"<b>{column_label}</b>",
                )
            )
        fig.update_layout(width=width, height=height, font_size=12)
        return fig

    def to_json(self) -> str:
        data = {
            "columns": [
                {
                    "name": col.name,
                    "period": col.period,
                    "nodes": [node.to_dict() for node in col.nodes],
                }
                for col in self.columns.values()
            ],
            "links": [link.to_dict() for link in self.links + self.sink_links],
        }
        return json.dumps(data, indent=2)

    @staticmethod
    def from_json(json_str: str) -> "SankeyPlotter":
        data = json.loads(json_str)
        columns = [
            SankeyColumn(
                name=col_data["name"],
                period=col_data["period"],
                nodes=[SankeyNode.from_dict(n) for n in col_data["nodes"]],
            )
            for col_data in data["columns"]
        ]
        diagram = SankeyPlotter(columns)
        all_nodes = {
            (node.name, node.period): node
            for col in diagram.columns.values()
            for node in col.nodes
        }

        def get_or_create_node(name: str, period: float) -> SankeyNode:
            key = (name, period)
            if key in all_nodes:
                return all_nodes[key]
            if name == "":
                if period not in diagram.sink_nodes:
                    diagram.sink_nodes[period] = SankeySinkNode(period)
                return diagram.sink_nodes[period]
            raise KeyError(f"Node {key} not found in loaded data.")

        for link_data in data["links"]:
            src = get_or_create_node(link_data["source"], link_data["source_period"])
            tgt = get_or_create_node(link_data["target"], link_data["target_period"])
            link = SankeyLink(source=src, target=tgt, value=link_data["value"])
            link.color = link_data.get("color")
            diagram.add_link(link)
        return diagram

    def to_dataframe(
        self, include_links: bool = True
    ) -> Union[pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
        node_rows = []
        for col in self.columns.values():
            for node in col.nodes:
                node_rows.append(
                    {
                        "column_name": col.name,
                        "name": node.name,
                        "count": node.count,
                        "period": node.period,
                        "rank": node.rank,
                        "x_pos": node.x_pos,
                        "y_pos": node.y_pos,
                        "color": node.color,
                        "is_sink": False,
                    }
                )
        for node in self.sink_nodes.values():
            node_rows.append(
                {
                    "name": node.name,
                    "count": node.count,
                    "period": node.period,
                    "rank": node.rank,
                    "x_pos": node.x_pos,
                    "y_pos": node.y_pos,
                    "color": node.color,
                    "is_sink": True,
                }
            )
        if include_links:
            link_rows = [
                {
                    "source_name": link.source.name,
                    "source_period": link.source.period,
                    "target_name": link.target.name,
                    "target_period": link.target.period,
                    "value": link.value,
                    "color": link.color,
                }
                for link in self.links + self.sink_links
            ]
            return pd.DataFrame(node_rows), pd.DataFrame(link_rows)
        else:
            return pd.DataFrame(node_rows)

    @staticmethod
    def from_dataframe(
        node_df: pd.DataFrame,
        link_df: Optional[pd.DataFrame] = None,
        auto_link: bool = True,
    ) -> "SankeyPlotter":
        columns: dict[int, SankeyColumn] = {}
        node_map: dict[tuple[str, float], SankeyNode] = {}
        sink_nodes: dict[float, SankeySinkNode] = {}

        for _, row in node_df.iterrows():
            name, period = row["name"], row["period"]
            if row.get("is_sink", False):
                node = SankeySinkNode(period=period)
            else:
                node = SankeyNode(name, row["count"], period, row["rank"])

            node.x_pos = row.get("x_pos")
            node.y_pos = row.get("y_pos")
            node.color = row.get("color")

            node_map[(name, period)] = node

            if isinstance(node, SankeySinkNode):
                sink_nodes[period] = node
            else:
                col_name = row.get("column_name", f"Period {period}")
                if period not in columns:
                    columns[period] = SankeyColumn(name=col_name, period=period)
                columns[period].nodes.append(node)

        diagram = SankeyPlotter(columns=list(columns.values()))
        diagram.sink_nodes = sink_nodes

        if link_df is not None:
            for _, row in link_df.iterrows():
                src = node_map[(row["source_name"], row["source_period"])]
                tgt = node_map[(row["target_name"], row["target_period"])]
                link = SankeyLink(src, tgt, row["value"])
                link.color = row.get("color")
                diagram.add_link(link)
        elif auto_link:
            diagram.connect_columns()

        return diagram
