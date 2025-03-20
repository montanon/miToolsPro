from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
from networkx import DiGraph, Graph
from pandas import DataFrame, Interval
from pyvis.network import Network as VisNetwork

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError

NodeID = Union[str, int]
NodeColor = Union[Tuple[float, ...], Tuple[int, ...]]
NodesColors = Dict[NodeID, NodeColor]
NodesLabels = Dict[NodeID, str]
NodesSizes = Dict[NodeID, Union[int, float]]
EdgesWidthsBins = Dict[Interval, float]


def build_nx_graph(
    proximity_vectors: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
) -> Graph:
    required_columns = {orig_product, dest_product}
    if not required_columns.issubset(proximity_vectors.columns):
        missing_cols = required_columns - set(proximity_vectors.columns)
        raise ArgumentValueError(f"Missing columns in DataFrame: {missing_cols}")
    G = nx.from_pandas_edgelist(
        proximity_vectors, source=orig_product, target=dest_product, edge_attr=True
    )

    return G


def build_nx_graphs(
    proximity_vectors: Dict[Union[str, int], DataFrame],
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    networks_folder: Optional[PathLike] = None,
    recalculate: Optional[bool] = False,
) -> Tuple[Dict[Union[str, int], Graph], Dict[Union[str, int], Optional[Path]]]:
    graphs = {}
    graph_files = {}
    for key, vectors in proximity_vectors.items():
        if networks_folder is not None:
            networks_folder = Path(networks_folder)
            if not networks_folder.exists():
                raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")
            gml_name = f"{key}_G_graph.gml".replace(" ", "_")
            gml_path = networks_folder / gml_name

            if not gml_path.exists() or recalculate:
                G = build_nx_graph(
                    vectors, orig_product=orig_product, dest_product=dest_product
                )
                nx.write_gml(G, gml_path)
            else:
                G = nx.read_gml(gml_path)
            graph_files[key] = gml_path
        else:
            G = build_nx_graph(
                vectors, orig_product=orig_product, dest_product=dest_product
            )
            graph_files[key] = None

        graphs[key] = G

    return graphs, graph_files


def build_mst_graph(
    proximity_vectors: DataFrame,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    attribute: str = "weight",
    attribute_th: Optional[float] = None,
    n_extra_edges: Optional[int] = None,
    pct_extra_edges: Optional[float] = None,
) -> Graph:
    required_columns = {orig_product, dest_product, attribute}
    if not required_columns.issubset(proximity_vectors.columns):
        missing_cols = required_columns - set(proximity_vectors.columns)
        raise ArgumentValueError(f"Missing columns in DataFrame: {missing_cols}")
    sorted_vectors = proximity_vectors.sort_values(by=attribute, ascending=False)
    G = build_nx_graph(
        sorted_vectors, orig_product=orig_product, dest_product=dest_product
    )
    MST = nx.maximum_spanning_tree(G, weight=attribute)
    extra_edges = None
    if attribute_th is not None:
        extra_edges = sorted_vectors.query(f"{attribute} >= @attribute_th")
    elif n_extra_edges is not None:
        n_total_edges = len(MST.edges) + n_extra_edges
        extra_edges = sorted_vectors.iloc[:n_total_edges]
    elif pct_extra_edges is not None:
        n_total_edges = int(
            (sorted_vectors.shape[0] - len(MST.edges)) * pct_extra_edges
        )
        extra_edges = sorted_vectors.iloc[
            len(MST.edges) : len(MST.edges) + n_total_edges
        ]
    if extra_edges is not None:
        extra_graph = build_nx_graph(
            extra_edges, orig_product=orig_product, dest_product=dest_product
        )
        combined_graph = nx.compose(MST, extra_graph)
        for u, v, data in G.edges(data=True):
            if combined_graph.has_edge(u, v):
                combined_graph[u][v][attribute] = data[attribute]
        MST = combined_graph
    return MST


def build_mst_graphs(
    proximity_vectors: Dict[Union[str, int], DataFrame],
    networks_folder: PathLike,
    orig_product: str = "product_i",
    dest_product: str = "product_j",
    attribute: str = "weight",
    attribute_th: Optional[float] = None,
    n_extra_edges: Optional[int] = None,
    pct_extra_edges: Optional[float] = None,
    recalculate: bool = False,
) -> Tuple[Dict[Union[str, int], Graph], Dict[Union[str, int], Path]]:
    networks_folder = Path(networks_folder)
    if not networks_folder.exists():
        raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")
    graphs = {}
    graph_files = {}
    for key, vectors in proximity_vectors.items():
        gml_name = f"{key}_MST_graph.gml".replace(" ", "_")
        gml_path = networks_folder / gml_name
        if not gml_path.exists() or recalculate:
            MST = build_mst_graph(
                vectors,
                orig_product=orig_product,
                dest_product=dest_product,
                attribute=attribute,
                attribute_th=attribute_th,
                n_extra_edges=n_extra_edges,
                pct_extra_edges=pct_extra_edges,
            )
            nx.write_gml(MST, gml_path)
        else:
            MST = nx.read_gml(gml_path)
        graphs[key] = MST
        graph_files[key] = str(gml_path)

    return graphs, graph_files


def build_vis_graph(
    graph: Graph,
    nodes_sizes: Optional[Union[NodesSizes, int, float]] = None,
    nodes_colors: Optional[Union[NodesColors, NodeColor]] = None,
    nodes_labels: Optional[Union[NodesLabels, str]] = None,
    node_label_size: Optional[Union[Dict[NodeID, int], int]] = None,
    edges_widths: Optional[EdgesWidthsBins] = None,
    net_height: int = 700,
    notebook: bool = True,
    physics: bool = False,
    physics_kwargs: Optional[Dict[str, Any]] = None,
) -> VisNetwork:
    if physics_kwargs is None:
        physics_kwargs = {
            "gravity": -1000000,
            "central_gravity": 0.0,
            "spring_length": 500,
            "spring_strength": 2,
            "damping": 0.1,
            "overlap": 1,
        }
    net = VisNetwork(height=f"{net_height}px", notebook=notebook)
    net.from_nx(graph)

    assign_net_nodes_attributes(
        net=net,
        sizes=nodes_sizes,
        colors=nodes_colors,
        labels=nodes_labels,
        label_sizes=node_label_size,
    )

    assign_net_edges_attributes(net=net, edges_widths=edges_widths)

    net.barnes_hut(**physics_kwargs)
    if physics:
        net.show_buttons(filter_=["physics"])
    return net


def build_vis_graphs(
    graphs_data: Dict[Union[str, int], Graph],
    networks_folder: PathLike,
    nodes_sizes: Optional[Union[NodesSizes, int, float]] = None,
    nodes_colors: Optional[Union[NodesColors, NodeColor]] = None,
    nodes_labels: Optional[Union[NodesLabels, str]] = None,
    node_label_size: Optional[Union[Dict[NodeID, int], int]] = None,
    edges_widths: Optional[EdgesWidthsBins] = None,
    net_height: int = 700,
    notebook: bool = True,
    physics: bool = False,
    physics_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[Union[str, int], VisNetwork], Dict[Union[str, int], str]]:
    networks_folder = Path(networks_folder)
    if not networks_folder.exists():
        raise ArgumentValueError(f"Folder '{networks_folder}' does not exist.")

    vis_graphs = {}
    graph_files = {}

    for key, graph in graphs_data.items():
        gml_name = f"{key}_vis_graph.html".replace(" ", "_")
        gml_path = networks_folder / gml_name

        if not gml_path.exists():
            net = build_vis_graph(
                graph=graph,
                nodes_sizes=nodes_sizes,
                nodes_colors=nodes_colors,
                nodes_labels=nodes_labels,
                node_label_size=node_label_size,
                edges_widths=edges_widths,
                net_height=net_height,
                notebook=notebook,
                physics=physics,
                physics_kwargs=physics_kwargs,
            )
            net.save_graph(str(gml_path))  # Save the graph as an HTML file

        vis_graphs[key] = net
        graph_files[key] = str(gml_path)

    return vis_graphs, graph_files


def assign_net_edges_attributes(net: VisNetwork, edges_widths: EdgesWidthsBins):
    if edges_widths is not None:
        for edge in net.edges:
            try:
                edge["width"] = next(
                    w for b, w in edges_widths.items() if edge["width"] in b
                )
            except StopIteration:
                raise ArgumentValueError(
                    "Some edge width values are not present in the corresponding 'edges_widths' argument."
                )


def assign_net_nodes_attributes(
    net: VisNetwork,
    sizes: Optional[Union[NodesSizes, int, float]] = None,
    colors: Optional[Union[NodesColors, NodeColor]] = None,
    labels: Optional[Union[NodesLabels, str]] = None,
    label_sizes: Optional[Union[Dict[NodeID, int], int]] = None,
):
    if sizes is not None and not isinstance(sizes, (int, float, dict)):
        raise ArgumentTypeError("Nodes 'sizes' must be a int, float or dict.")
    if isinstance(sizes, dict) and not all(node["id"] in sizes for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'sizes' argument."
        )
    if colors is not None and not isinstance(colors, (tuple, list, dict)):
        raise ArgumentTypeError(
            "Nodes 'colors' must be a tuple, list, NodeColor or dict."
        )
    if isinstance(colors, dict) and not all(node["id"] in colors for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'colors' argument."
        )
    if labels is not None and not isinstance(labels, (str, dict)):
        raise ArgumentTypeError("Nodes 'labels' must be a str or dict.")
    if isinstance(labels, dict) and not all(node["id"] in labels for node in net.nodes):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'labels' argument."
        )
    if label_sizes is not None and not isinstance(label_sizes, (int, dict)):
        raise ArgumentTypeError("Nodes 'label_sizes' must be a int or dict.")
    if isinstance(label_sizes, dict) and not all(
        node["id"] in label_sizes for node in net.nodes
    ):
        raise ArgumentValueError(
            "Some node ids are not present in the corresponding 'label_sizes' argument."
        )
    if sizes is not None:
        for node in net.nodes:
            node["size"] = sizes if not isinstance(sizes, dict) else sizes[node["id"]]
    if colors is not None:
        for node in net.nodes:
            node["color"] = (
                colors if not isinstance(colors, dict) else colors[node["id"]]
            )
    if labels is not None:
        for node in net.nodes:
            node["label"] = (
                labels if not isinstance(labels, dict) else labels[node["id"]]
            )
    if label_sizes is not None:
        for node in net.nodes:
            node["font"] = (
                f"{label_sizes}px arial black"
                if not isinstance(label_sizes, dict)
                else f"{label_sizes[node['id']]}px arial black"
            )


def _convert_color(color):
    if isinstance(color, str):
        color_str = color.strip().lower()
        if color_str.startswith("#"):
            color_str = color_str[1:]
            try:
                if len(color_str) == 6:
                    return tuple(int(color_str[i : i + 2], 16) for i in (0, 2, 4))
                elif len(color_str) == 3:
                    return tuple(int(c * 2, 16) for c in color_str)
                else:
                    return color
            except Exception:
                return color
        elif color_str.startswith(("rgb(", "rgba(")):
            try:
                values = color_str[color_str.find("(") + 1 : color_str.find(")")].split(
                    ","
                )
                values = [int(v.strip()) for v in values]
                if len(values) == 3:  # RGB
                    return tuple(values)
                elif len(values) == 4:  # RGBA
                    return tuple(values)
                else:
                    return color
            except Exception:
                return color
        else:
            return color
    elif isinstance(color, (list, tuple)):
        try:
            return tuple(int(c) for c in color)
        except Exception:
            return color
    return color


def pyvis_to_networkx(pyvis_network: "VisNetwork") -> Union[Graph, DiGraph]:
    if not hasattr(pyvis_network, "nodes") or not hasattr(pyvis_network, "edges"):
        raise TypeError(
            "Input must be a PyVis network with 'nodes' and 'edges' attributes."
        )

    directed = getattr(pyvis_network, "directed", False)
    nx_graph = DiGraph() if directed else Graph()
    for node in pyvis_network.nodes:
        if not isinstance(node, dict):
            try:
                node = dict(node)
            except Exception as e:
                raise ValueError("A node cannot be converted to a dictionary.") from e
        if "id" not in node:
            raise ValueError("Every node must have an 'id' attribute.")
        node_id = node["id"]
        node_attrs = {}
        for key, value in node.items():
            if key == "id":
                continue
            if key == "label":
                node_attrs["label"] = value
                node_attrs["name"] = value
            elif key == "color":
                node_attrs["color"] = _convert_color(value)
            else:
                node_attrs[key] = value
        nx_graph.add_node(node_id, **node_attrs)
    for edge in pyvis_network.edges:
        if not isinstance(edge, dict):
            try:
                edge = dict(edge)
            except Exception as e:
                raise ValueError("An edge cannot be converted to a dictionary.") from e
        if "from" not in edge or "to" not in edge:
            raise ValueError("Every edge must have 'from' and 'to' attributes.")
        source = edge["from"]
        target = edge["to"]
        edge_attrs = {}
        for key, value in edge.items():
            if key in ["from", "to"]:
                continue
            if key == "width":
                edge_attrs["weight"] = value
            elif key == "color":
                edge_attrs["color"] = _convert_color(value)
            else:
                edge_attrs[key] = value
        nx_graph.add_edge(source, target, **edge_attrs)

    return nx_graph


def draw_nx_colored_graph(
    G: Graph,
    pos_G: Dict[Any, Tuple[float, float]],
    node_colors: NodesColors,
    edge_widths: Dict[float, List[Tuple[Any, Any]]],
    node_size: int = 10,
    edge_alpha: float = 1.0,
    width_scale: float = 10.0,
):
    if not isinstance(G, Graph):
        raise ArgumentTypeError("G must be a NetworkX graph.")
    if not isinstance(pos_G, dict):
        raise ArgumentTypeError("pos_G must be a dictionary of node positions.")
    for color, nodes in node_colors.items():
        if not all(node in G for node in nodes):
            raise ArgumentValueError("Some nodes in 'nodes' are not in the graph.")
        nx.draw_networkx_nodes(
            G, pos_G, nodelist=nodes, node_color=color, node_size=node_size
        )
    for width, edges in edge_widths.items():
        if not all(G.has_edge(u, v) for u, v in edges):
            raise ArgumentValueError(
                "Some edges in 'edges' are not present in the graph."
            )
        nx.draw_networkx_edges(
            G, pos_G, edgelist=edges, width=width / width_scale, alpha=edge_alpha
        )


def draw_nx(g: Union[Graph, DiGraph], with_labels: bool = True):
    pos = nx.spring_layout(g)

    default_node_color = (0, 0, 1)
    default_node_size = 10
    default_edge_color = "black"
    default_edge_width = 1.0

    node_groups = {}
    for node, node_data in g.nodes(data=True):
        shape = node_data.get("shape", "dot")
        if shape == "dot":
            marker = "o"
        elif shape == "square":
            marker = "s"
        elif shape == "triangle":
            marker = "^"
        else:
            marker = shape  # allow custom markers if valid in matplotlib
        node_groups.setdefault(marker, []).append(node)

    edge_colors = []
    edge_widths = []
    for u, v, node_data in g.edges(data=True):
        color = _convert_color(node_data.get("color", default_edge_color))
        if isinstance(color, (list, tuple)) and all(
            isinstance(x, (int, float)) for x in color
        ):
            if max(color) > 1:
                color = tuple(x / 255 for x in color)
            else:
                color = tuple(color)
        edge_colors.append(color)
        edge_widths.append(node_data.get("weight", default_edge_width))
    nx.draw_networkx_edges(g, pos, edge_color=edge_colors, width=edge_widths)
    for marker, nodes in node_groups.items():
        node_colors = []
        node_sizes = []
        node_labels = {}
        for n in nodes:
            node_data = g.nodes[n]
            color = node_data.get("color", default_node_color)
            if isinstance(color, (list, tuple)) and all(
                isinstance(x, (int, float)) for x in color
            ):
                if max(color) > 1:
                    color = tuple(x / 255 for x in color)
                else:
                    color = tuple(color)
            node_colors.append(color)
            size = node_data.get("size", default_node_size)
            node_sizes.append(size * 100)
            if "name" in node_data:
                node_labels[n] = node_data["name"]
            elif "label" in node_data:
                node_labels[n] = node_data["label"]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nodes,
            node_color=node_colors,
            node_size=node_sizes,
            node_shape=marker,
        )
        if with_labels:
            nx.draw_networkx_labels(g, pos, labels=node_labels)


def distribute_items_in_communities(items: Sequence, n_communities: int) -> Sequence:
    if n_communities < 1:
        raise ArgumentValueError("The number of communities must be greater than zero.")
    if len(items) < n_communities:
        raise ArgumentValueError(
            "The number of items must be greater or equal to the number of communities."
        )
    np.random.shuffle(items)
    size = len(items) // n_communities
    remainder = len(items) % n_communities
    communities = []
    start = 0
    for i in range(n_communities):
        end = start + size + (1 if i < remainder else 0)
        communities.append(items[start:end])
        start = end
    return communities


def average_strength_of_links_within_community(G: Graph, community: List[Any]) -> float:
    links = G.edges(community, data=True)
    strengths = [
        d.get("width", d.get("weight", 0.0))  # Handle missing 'width' and 'weight'
        for u, v, d in links
        if v in community
    ]
    return np.mean(strengths) if strengths else np.nan


def average_strength_of_links_within_communities(
    G: Graph, communities: List[List[Any]]
) -> Dict[str, Union[float, int]]:
    strengths = [
        average_strength_of_links_within_community(G, community)
        for community in communities
    ]
    strengths = [s for s in strengths if not np.isnan(s)]
    return {
        "mean": np.mean(strengths),
        "std": np.std(strengths),
        "max": np.max(strengths),
        "min": np.min(strengths),
    }


def average_strength_of_links_from_community(G: Graph, community: List[Any]) -> float:
    links = G.edges(data=True)
    strengths = [
        d.get("width", d.get("weight", 0.0))  # Handle missing 'width' and 'weight'
        for u, v, d in links
        if u in community and v not in community
    ]
    return np.mean(strengths) if strengths else np.nan


def average_strength_of_links_from_communities(
    G: Graph, communities: List[List[Any]]
) -> Dict[str, Union[float, int]]:
    strengths = [
        average_strength_of_links_from_community(G, community)
        for community in communities
    ]
    strengths = [s for s in strengths if not np.isnan(s)]
    return {
        "mean": np.mean(strengths),
        "std": np.std(strengths),
        "max": np.max(strengths),
        "min": np.min(strengths),
    }
