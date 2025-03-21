import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import TestCase

import networkx as nx
import numpy as np
from networkx import DiGraph, Graph
from pandas import DataFrame, Interval
from pyvis.network import Network as VisNetwork

from mitoolspro.exceptions import ArgumentTypeError, ArgumentValueError
from mitoolspro.networks import (
    EdgesWidthsBins,
    NodesColors,
    NodesLabels,
    NodesSizes,
    assign_net_edges_attributes,
    assign_net_nodes_attributes,
    average_strength_of_links_from_communities,
    average_strength_of_links_from_community,
    average_strength_of_links_within_communities,
    average_strength_of_links_within_community,
    build_mst_graph,
    build_mst_graphs,
    build_nx_graph,
    build_nx_graphs,
    build_vis_graph,
    build_vis_graphs,
    distribute_items_in_communities,
    draw_nx_colored_graph,
    pyvis_to_networkx,
)
from mitoolspro.networks.networks import _convert_color


class TestBuildNxGraph(TestCase):
    def setUp(self):
        self.proximity_vectors = DataFrame(
            {
                "node_i": ["A", "A", "B"],
                "node_j": ["B", "C", "C"],
                "weight": [0.8, 0.4, 0.5],
            }
        )

    def test_valid_graph(self):
        G = build_nx_graph(self.proximity_vectors)
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 3)
        self.assertAlmostEqual(G["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(G["A"]["C"]["weight"], 0.4)
        self.assertAlmostEqual(G["B"]["C"]["weight"], 0.5)

    def test_missing_column(self):
        with self.assertRaises(ArgumentValueError):
            invalid_vectors = self.proximity_vectors.drop(columns=["node_i"])
            build_nx_graph(invalid_vectors)

    def test_empty_dataframe(self):
        empty_df = DataFrame(columns=["node_i", "node_j", "weight"])
        G = build_nx_graph(empty_df)
        self.assertEqual(len(G.nodes), 0)
        self.assertEqual(len(G.edges), 0)

    def test_graph_with_additional_attributes(self):
        vectors_with_extra_attr = self.proximity_vectors.assign(year=[2020, 2021, 2022])
        G = build_nx_graph(vectors_with_extra_attr)
        self.assertEqual(len(G.nodes), 3)
        self.assertEqual(len(G.edges), 3)
        self.assertEqual(G["A"]["B"]["year"], 2020)
        self.assertEqual(G["A"]["C"]["year"], 2021)
        self.assertEqual(G["B"]["C"]["year"], 2022)


class TestBuildNxGraphs(TestCase):
    def setUp(self):
        self.proximity_vectors = {
            1: DataFrame(
                {"node_i": ["A", "A"], "node_j": ["B", "C"], "weight": [0.8, 0.4]}
            ),
            2: DataFrame({"node_i": ["B"], "node_j": ["C"], "weight": [0.5]}),
        }

    def test_build_and_store_graphs(self):
        graphs, graph_files = build_nx_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            recalculate=True,
        )
        for key, _ in graph_files.items():
            self.assertTrue(isinstance(graphs[key], nx.Graph))
        self.assertEqual(len(graphs[1].nodes), 3)
        self.assertEqual(len(graphs[1].edges), 2)
        self.assertAlmostEqual(graphs[1]["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(graphs[1]["A"]["C"]["weight"], 0.4)
        self.assertEqual(len(graphs[2].nodes), 2)
        self.assertEqual(len(graphs[2].edges), 1)
        self.assertAlmostEqual(graphs[2]["B"]["C"]["weight"], 0.5)

    def test_load_existing_graphs(self):
        build_nx_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            recalculate=True,
        )
        graphs, graph_files = build_nx_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            recalculate=False,
        )
        for key, _ in graph_files.items():
            self.assertTrue(isinstance(graphs[key], nx.Graph))
        self.assertEqual(len(graphs[1].nodes), 3)
        self.assertEqual(len(graphs[1].edges), 2)
        self.assertAlmostEqual(graphs[1]["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(graphs[1]["A"]["C"]["weight"], 0.4)
        self.assertEqual(len(graphs[2].nodes), 2)
        self.assertEqual(len(graphs[2].edges), 1)
        self.assertAlmostEqual(graphs[2]["B"]["C"]["weight"], 0.5)

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_nx_graphs(
                self.proximity_vectors,
                origin="node_i",
                destination="node_j",
                networks_folder="non_existent_folder",
                recalculate=False,
            )

    def test_empty_proximity_vectors(self):
        graphs, graph_files = build_nx_graphs(
            {},
            origin="node_i",
            destination="node_j",
            recalculate=True,
        )
        self.assertEqual(len(graphs), 0)
        self.assertEqual(len(graph_files), 0)


class TestBuildMSTGraph(TestCase):
    def setUp(self):
        self.proximity_vectors = DataFrame(
            {
                "node_i": ["A", "A", "B", "C"],
                "node_j": ["B", "C", "C", "D"],
                "weight": [0.8, 0.4, 0.5, 0.6],
            }
        )
        self.G = build_nx_graph(self.proximity_vectors)

    def test_mst_no_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors)
        self.assertEqual(len(mst.edges), 3)  # Expected 3 edges in the MST

    def test_mst_with_attribute_threshold(self):
        mst = build_mst_graph(self.proximity_vectors, attribute_th=0.5)
        self.assertEqual(len(mst.edges), 3)  # All edges with weight >= 0.5

    def test_mst_with_n_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, n_extra_edges=1)
        self.assertEqual(len(mst.edges), 4)  # 3 MST edges + 1 extra edge

    def test_mst_with_pct_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, pct_extra_edges=1.0)
        self.assertEqual(len(mst.edges), 4)  # 50% of remaining edges added

    def test_mst_with_all_extra_edges(self):
        mst = build_mst_graph(self.proximity_vectors, pct_extra_edges=1.0)
        self.assertEqual(len(mst.edges), 4)  # All edges added to the MST

    def test_missing_columns(self):
        invalid_vectors = self.proximity_vectors.drop(columns=["weight"])
        with self.assertRaises(ArgumentValueError):
            build_mst_graph(invalid_vectors)

    def test_empty_proximity_vectors(self):
        empty_vectors = DataFrame(columns=["node_i", "node_j", "weight"])
        mst = build_mst_graph(empty_vectors)
        self.assertEqual(len(mst.edges), 0)

    def test_preserve_original_weights(self):
        mst = build_mst_graph(self.proximity_vectors, n_extra_edges=1)
        self.assertAlmostEqual(mst["A"]["B"]["weight"], 0.8)
        self.assertAlmostEqual(mst["C"]["D"]["weight"], 0.6)

    def test_graph_with_custom_attributes(self):
        vectors_with_attr = self.proximity_vectors.assign(year=[2020, 2021, 2022, 2023])
        mst = build_mst_graph(vectors_with_attr, attribute="year")
        self.assertTrue(all("year" in data for _, _, data in mst.edges(data=True)))


class TestBuildMSTGraphs(TestCase):
    def setUp(self):
        self.proximity_vectors = {
            1: DataFrame(
                {
                    "node_i": ["A", "A", "B", "C", "D"],
                    "node_j": ["B", "C", "C", "D", "A"],
                    "weight": [0.8, 0.4, 0.5, 0.6, 0.1],
                }
            ),
            2: DataFrame(
                {
                    "node_i": ["A", "A", "B", "C", "D"],
                    "node_j": ["B", "C", "C", "D", "A"],
                    "weight": [0.8, 0.4, 0.5, 0.6, 0.1],
                }
            ),
        }

    def test_build_and_store_mst_graphs(self):
        graphs, graph_files = build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            recalculate=True,
        )
        for key, _ in graph_files.items():
            self.assertIsInstance(graphs[key], Graph)

    def test_load_existing_mst_graphs(self):
        build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            recalculate=True,
        )
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            recalculate=False,
        )
        for key, graph in graphs.items():
            self.assertIsInstance(graph, Graph)

    def test_with_n_extra_edges(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            n_extra_edges=1,
            recalculate=True,
        )
        for graph in graphs.values():
            self.assertEqual(len(graph.edges), 4)  # 2 MST edges + 1 extra edge

    def test_attribute_threshold(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            attribute_th=0.4,
            recalculate=True,
        )
        for graph in graphs.values():
            for _, _, data in graph.edges(data=True):
                self.assertGreaterEqual(data["weight"], 0.4)

    def test_with_pct_extra_edges(self):
        graphs, _ = build_mst_graphs(
            self.proximity_vectors,
            origin="node_i",
            destination="node_j",
            attribute="weight",
            pct_extra_edges=0.0,
            recalculate=True,
        )
        for graph in graphs.values():
            self.assertEqual(len(graph.edges), 3)  # 0% of remaining edges added

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_mst_graphs(
                self.proximity_vectors,
                networks_folder="non_existent_folder",
                origin="node_i",
                destination="node_j",
                attribute="weight",
            )

    def test_empty_proximity_vectors(self):
        graphs, graph_files = build_mst_graphs(
            {},
            origin="node_i",
            destination="node_j",
            attribute="weight",
            recalculate=True,
        )
        self.assertEqual(len(graphs), 0)  # No graphs should be built
        self.assertEqual(len(graph_files), 0)


class TestAssignNetNodesAttributes(TestCase):
    def setUp(self):
        self.net = VisNetwork()
        self.net.add_node(1)
        self.net.add_node(2)
        self.net.add_node(3)

    def test_assign_valid_sizes(self):
        sizes = {1: 10, 2: 15, 3: 20}
        assign_net_nodes_attributes(self.net, sizes=sizes)
        for node in self.net.nodes:
            self.assertEqual(node["size"], sizes[node["id"]])

    def test_assign_single_size(self):
        assign_net_nodes_attributes(self.net, sizes=12)
        for node in self.net.nodes:
            self.assertEqual(node["size"], 12)

    def test_invalid_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, sizes="invalid_size")

    def test_missing_node_in_sizes(self):
        sizes = {1: 10, 2: 15}  # Missing size for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, sizes=sizes)

    def test_assign_valid_colors(self):
        colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        assign_net_nodes_attributes(self.net, colors=colors)
        for node in self.net.nodes:
            self.assertEqual(node["color"], colors[node["id"]])

    def test_invalid_colors_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, colors="invalid_color")

    def test_missing_node_in_colors(self):
        colors = {1: (255, 0, 0), 2: (0, 255, 0)}  # Missing color for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, colors=colors)

    def test_assign_valid_labels(self):
        labels = {1: "Node A", 2: "Node B", 3: "Node C"}
        assign_net_nodes_attributes(self.net, labels=labels)
        for node in self.net.nodes:
            self.assertEqual(node["label"], labels[node["id"]])

    def test_single_label_assignment(self):
        assign_net_nodes_attributes(self.net, labels="Common Label")
        for node in self.net.nodes:
            self.assertEqual(node["label"], "Common Label")

    def test_invalid_labels_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, labels=123)

    def test_missing_node_in_labels(self):
        labels = {1: "Node A", 2: "Node B"}  # Missing label for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, labels=labels)

    def test_assign_valid_label_sizes(self):
        label_sizes = {1: 15, 2: 20, 3: 25}
        assign_net_nodes_attributes(self.net, label_sizes=label_sizes)
        for node in self.net.nodes:
            self.assertEqual(node["font"], f"{label_sizes[node['id']]}px arial black")

    def test_single_label_size_assignment(self):
        assign_net_nodes_attributes(self.net, label_sizes=18)
        for node in self.net.nodes:
            self.assertEqual(node["font"], "18px arial black")

    def test_invalid_label_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            assign_net_nodes_attributes(self.net, label_sizes="invalid_size")

    def test_missing_node_in_label_sizes(self):
        label_sizes = {1: 15, 2: 20}  # Missing size for node 3
        with self.assertRaises(ArgumentValueError):
            assign_net_nodes_attributes(self.net, label_sizes=label_sizes)


class TestBuildVisGraph(TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edge(1, 2, weight=3.0)
        self.graph.add_edge(2, 3, weight=7.0)
        self.nodes_sizes: NodesSizes = {1: 15, 2: 20, 3: 25}
        self.nodes_colors: NodesColors = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
        }
        self.nodes_labels: NodesLabels = {1: "A", 2: "B", 3: "C"}
        self.edges_widths: EdgesWidthsBins = {
            Interval(0, 5, closed="both"): 2.0,
            Interval(5, 10, closed="both"): 5.0,
        }

    def test_build_with_valid_attributes(self):
        net = build_vis_graph(
            graph=self.graph,
            nodes_sizes=self.nodes_sizes,
            nodes_colors=self.nodes_colors,
            nodes_labels=self.nodes_labels,
            edges_widths=self.edges_widths,
        )
        self.assertIsInstance(net, VisNetwork)
        self.assertEqual(len(net.nodes), 3)  # Ensure all nodes are added
        self.assertEqual(len(net.edges), 2)  # Ensure all edges are added

    def test_build_with_single_size(self):
        net = build_vis_graph(graph=self.graph, nodes_sizes=12)
        for node in net.nodes:
            self.assertEqual(node["size"], 12)

    def test_build_with_single_color(self):
        net = build_vis_graph(graph=self.graph, nodes_colors=(255, 0, 0))
        for node in net.nodes:
            self.assertEqual(node["color"], (255, 0, 0))

    def test_build_with_single_label(self):
        net = build_vis_graph(graph=self.graph, nodes_labels="Common Label")
        for node in net.nodes:
            self.assertEqual(node["label"], "Common Label")

    def test_invalid_nodes_sizes_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_sizes="invalid_size")

    def test_invalid_nodes_colors_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_colors="invalid_color")

    def test_invalid_nodes_labels_type(self):
        with self.assertRaises(ArgumentTypeError):
            build_vis_graph(graph=self.graph, nodes_labels=123)

    def test_physics_settings(self):
        physics_kwargs = {
            "gravity": -5000,
            "spring_length": 300,
            "damping": 0.2,
        }
        net = build_vis_graph(
            graph=self.graph, physics=True, physics_kwargs=physics_kwargs
        )
        self.assertIsInstance(net, VisNetwork)  # Ensure a network is returned

    def test_empty_graph(self):
        empty_graph = nx.Graph()
        net = build_vis_graph(graph=empty_graph)
        self.assertEqual(len(net.nodes), 0)
        self.assertEqual(len(net.edges), 0)

    def test_missing_edges_widths(self):
        net = build_vis_graph(graph=self.graph)
        for edge in net.edges:
            self.assertIn("width", edge)  # Ensure width attribute exists

    def test_edge_width_not_in_bins(self):
        self.graph.add_edge(3, 4, weight=12.0)  # Add an edge with a weight not in bins
        with self.assertRaises(ArgumentValueError):
            build_vis_graph(graph=self.graph, edges_widths=self.edges_widths)

    def test_assign_label_sizes(self):
        label_sizes = {1: 15, 2: 20, 3: 25}
        net = build_vis_graph(graph=self.graph, node_label_size=label_sizes)
        for node in net.nodes:
            expected_size = label_sizes[node["id"]]
            self.assertEqual(node["font"], f"{expected_size}px arial black")

    def test_assign_single_label_size(self):
        net = build_vis_graph(graph=self.graph, node_label_size=18)
        for node in net.nodes:
            self.assertEqual(node["font"], "18px arial black")


class TestBuildVisGraphs(TestCase):
    def setUp(self):
        self.graphs_data = {
            1: nx.Graph([(1, 2, {"weight": 0.8}), (1, 3, {"weight": 0.4})]),
            2: nx.Graph([(2, 3, {"weight": 0.5})]),
        }

    def test_build_and_store_vis_graphs(self):
        vis_graphs, graph_files = build_vis_graphs(self.graphs_data)
        for key, _ in graph_files.items():
            self.assertIsInstance(vis_graphs[key], VisNetwork)
        self.assertEqual(len(vis_graphs[1].nodes), 3)
        self.assertEqual(len(vis_graphs[1].edges), 2)
        self.assertEqual(len(vis_graphs[2].nodes), 2)
        self.assertEqual(len(vis_graphs[2].edges), 1)

    def test_missing_network_folder(self):
        with self.assertRaises(ArgumentValueError):
            build_vis_graphs(self.graphs_data, networks_folder="non_existent_folder")

    def test_empty_graph_data(self):
        vis_graphs, graph_files = build_vis_graphs(
            {},
        )
        self.assertEqual(len(vis_graphs), 0)
        self.assertEqual(len(graph_files), 0)

    def test_build_with_custom_node_sizes(self):
        node_sizes = {1: 10, 2: 15, 3: 20}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            nodes_sizes=node_sizes,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["size"], node_sizes[node["id"]])

    def test_build_with_custom_node_colors(self):
        node_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            nodes_colors=node_colors,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["color"], node_colors[node["id"]])

    def test_build_with_custom_labels(self):
        node_labels = {1: "A", 2: "B", 3: "C"}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            nodes_labels=node_labels,
        )
        for node in vis_graphs[1].nodes:
            self.assertEqual(node["label"], node_labels[node["id"]])

    def test_build_with_custom_physics(self):
        physics_kwargs = {"gravity": -5000, "spring_length": 300}
        vis_graphs, _ = build_vis_graphs(
            self.graphs_data,
            physics=True,
            physics_kwargs=physics_kwargs,
        )
        self.assertIsInstance(vis_graphs[1], VisNetwork)


class TestAssignNetEdgesAttributes(TestCase):
    def setUp(self):
        self.net = VisNetwork()
        self.net.add_node(1)
        self.net.add_node(2)
        self.net.add_node(3)
        self.net.add_edge(1, 2, width=3.0)
        self.net.add_edge(2, 3, width=7.0)

        self.edges_widths = {
            Interval(0, 5, closed="both"): 2.0,
            Interval(5, 10, closed="both"): 5.0,
        }

    def test_assign_valid_edges_widths(self):
        assign_net_edges_attributes(self.net, self.edges_widths)
        expected_widths = [2.0, 5.0]
        for edge, expected in zip(self.net.edges, expected_widths):
            self.assertEqual(edge["width"], expected)

    def test_edge_width_not_in_bins(self):
        self.net.add_edge(3, 1, width=12.0)
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, self.edges_widths)

    def test_empty_edges_widths(self):
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, {})

    def test_no_matching_bins_for_edges(self):
        invalid_bins = {Interval(20, 30, closed="both"): 10.0}
        with self.assertRaises(ArgumentValueError):
            assign_net_edges_attributes(self.net, invalid_bins)

    def test_multiple_edges_with_same_width(self):
        self.net.add_edge(1, 3, width=3.0)  # Another edge with width 3.0
        assign_net_edges_attributes(self.net, self.edges_widths)
        for edge, val in zip(self.net.edges, [2.0, 5.0, 2.0]):
            self.assertEqual(edge["width"], val)


class TestPyvisToNetworkx(TestCase):
    def setUp(self):
        self.undirected_network = VisNetwork()
        self.undirected_network.add_node(1, label="Node A", size=10)
        self.undirected_network.add_node(2, label="Node B", size=15)
        self.undirected_network.add_edge(1, 2, width=2.0, title="Edge A-B")

        self.directed_network = VisNetwork(directed=True)
        self.directed_network.add_node(3, label="Node C", size=20)
        self.directed_network.add_node(4, label="Node D", size=25)
        self.directed_network.add_edge(3, 4, width=3.0, title="Edge C-D")

    def test_convert_undirected_network(self):
        nx_graph = pyvis_to_networkx(self.undirected_network)
        self.assertIsInstance(nx_graph, Graph)
        self.assertEqual(len(nx_graph.nodes), 2)
        self.assertEqual(len(nx_graph.edges), 1)
        self.assertEqual(nx_graph.nodes[1]["name"], "Node A")
        self.assertAlmostEqual(nx_graph[1][2]["weight"], 2.0)
        self.assertEqual(nx_graph[1][2]["title"], "Edge A-B")

    def test_convert_directed_network(self):
        nx_graph = pyvis_to_networkx(self.directed_network)
        self.assertIsInstance(nx_graph, DiGraph)
        self.assertEqual(len(nx_graph.nodes), 2)
        self.assertEqual(len(nx_graph.edges), 1)
        self.assertEqual(nx_graph.nodes[3]["name"], "Node C")
        self.assertAlmostEqual(nx_graph[3][4]["weight"], 3.0)
        self.assertEqual(nx_graph[3][4]["title"], "Edge C-D")

    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            pyvis_to_networkx([1, 2, 3])

    def test_default_edge_weight(self):
        self.undirected_network.add_edge(2, 1)
        nx_graph = pyvis_to_networkx(self.undirected_network)
        self.assertAlmostEqual(nx_graph[2][1]["weight"], 2.0)

    def test_comprehensive_attribute_transfer(self):
        network = VisNetwork()
        network.add_node(
            "1",
            label="Node A",
            size=10,
            color="#FF0000",  # Direct hex color
            font="20px Arial",
            title="Node A Title",
            level=1,
            shape="dot",
            x=100,
            y=200,
            fixed=True,
            mass=2.0,
            physics=True,
        )
        network.add_node(
            "2",
            label="Node B",
            size=15,
            color=(0, 255, 0),  # Direct RGB color
            font="15px Arial",
            title="Node B Title",
            level=2,
            shape="square",
            x=300,
            y=400,
            fixed=False,
            mass=1.5,
            physics=True,
        )
        network.add_edge(
            "1",
            "2",
            width=2.0,
            title="Edge A-B",
            color="#0000FF",  # Direct hex color for edge
            arrows="to",
            smooth={"type": "curvedCW"},
            dashes=True,
            label="Edge Label",
            font="12px Arial",
            hidden=False,
            selectionWidth=3.0,
            selfReferenceSize=20,
            selfReferenceAngle=90,
        )
        nx_graph = pyvis_to_networkx(network)

        node1_attrs = nx_graph.nodes["1"]
        self.assertEqual(node1_attrs["name"], "Node A")
        self.assertEqual(node1_attrs["label"], "Node A")
        self.assertEqual(node1_attrs["size"], 10)
        self.assertEqual(node1_attrs["color"], (255, 0, 0))  # Converted from hex
        self.assertEqual(node1_attrs["font"], "20px Arial")
        self.assertEqual(node1_attrs["title"], "Node A Title")
        self.assertEqual(node1_attrs["level"], 1)
        self.assertEqual(node1_attrs["shape"], "dot")
        self.assertEqual(node1_attrs["x"], 100)
        self.assertEqual(node1_attrs["y"], 200)
        self.assertTrue(node1_attrs["fixed"])
        self.assertEqual(node1_attrs["mass"], 2.0)
        self.assertTrue(node1_attrs["physics"])

        node2_attrs = nx_graph.nodes["2"]
        self.assertEqual(node2_attrs["name"], "Node B")
        self.assertEqual(node2_attrs["label"], "Node B")
        self.assertEqual(node2_attrs["size"], 15)
        self.assertEqual(node2_attrs["color"], (0, 255, 0))  # RGB tuple preserved
        self.assertEqual(node2_attrs["font"], "15px Arial")
        self.assertEqual(node2_attrs["title"], "Node B Title")
        self.assertEqual(node2_attrs["level"], 2)
        self.assertEqual(node2_attrs["shape"], "square")
        self.assertEqual(node2_attrs["x"], 300)
        self.assertEqual(node2_attrs["y"], 400)
        self.assertFalse(node2_attrs["fixed"])
        self.assertEqual(node2_attrs["mass"], 1.5)
        self.assertTrue(node2_attrs["physics"])

        edge_attrs = nx_graph["1"]["2"]
        self.assertEqual(edge_attrs["weight"], 2.0)  # width converted to weight
        self.assertEqual(edge_attrs["title"], "Edge A-B")
        self.assertEqual(edge_attrs["color"], (0, 0, 255))  # Converted from hex
        self.assertEqual(edge_attrs["arrows"], "to")
        self.assertEqual(edge_attrs["smooth"], {"type": "curvedCW"})
        self.assertTrue(edge_attrs["dashes"])
        self.assertEqual(edge_attrs["label"], "Edge Label")
        self.assertEqual(edge_attrs["font"], "12px Arial")
        self.assertFalse(edge_attrs["hidden"])
        self.assertEqual(edge_attrs["selectionWidth"], 3.0)
        self.assertEqual(edge_attrs["selfReferenceSize"], 20)
        self.assertEqual(edge_attrs["selfReferenceAngle"], 90)

        self.assertEqual(len(nx_graph), 2)
        self.assertIsInstance(nx_graph, Graph)  # Should be undirected by default

    def test_directed_graph_attributes(self):
        network = VisNetwork(directed=True)
        network.add_node(1, label="Node A", size=10)
        network.add_node(2, label="Node B", size=15)
        network.add_edge(1, 2, width=2.0, arrows="to")

        nx_graph = pyvis_to_networkx(network)

        self.assertIsInstance(nx_graph, DiGraph)

        self.assertTrue(nx_graph.has_edge(1, 2))
        self.assertFalse(nx_graph.has_edge(2, 1))

    def test_color_conversion_edge_cases(self):
        network = VisNetwork()

        network.add_node(1, color="#abc")  # 3-digit hex
        network.add_node(2, color="#aabbcc")  # 6-digit hex
        network.add_node(3, color=[100, 200, 300])  # List
        network.add_node(4, color=(255, 128, 64))  # Tuple
        network.add_node(5, color="rgb(255, 128, 64)")  # RGB string
        network.add_node(6, color="red")  # Named color

        nx_graph = pyvis_to_networkx(network)

        self.assertEqual(nx_graph.nodes[1]["color"], (170, 187, 204))  # #abc expanded
        self.assertEqual(nx_graph.nodes[2]["color"], (170, 187, 204))  # #aabbcc
        self.assertEqual(nx_graph.nodes[3]["color"], (100, 200, 300))  # List preserved
        self.assertEqual(nx_graph.nodes[4]["color"], (255, 128, 64))  # Tuple preserved

    def test_gml_serialization(self):
        network = VisNetwork()
        network.add_node(1, label="Node A", size=10, color="#FF0000")
        network.add_node(2, label="Node B", size=15, color=(0, 255, 0))
        network.add_edge(1, 2, width=2.0, color="#0000FF")

        nx_graph = pyvis_to_networkx(network)

        with tempfile.TemporaryDirectory() as temp_dir:
            test_gml_path = os.path.join(temp_dir, "test_network.gml")
            nx.write_gml(nx_graph, test_gml_path)

            loaded_graph = nx.read_gml(test_gml_path)

            self.assertEqual(loaded_graph.nodes["1"]["name"], "Node A")
            self.assertEqual(loaded_graph.nodes["1"]["size"], 10)
            self.assertEqual(
                loaded_graph.nodes["1"]["color"], [255, 0, 0]
            )  # GML stores as list
            self.assertEqual(loaded_graph.nodes["2"]["name"], "Node B")
            self.assertEqual(loaded_graph.nodes["2"]["size"], 15)
            self.assertEqual(loaded_graph.nodes["2"]["color"], [0, 255, 0])

            # Test edge attributes
            self.assertEqual(loaded_graph["1"]["2"]["weight"], 2.0)
            self.assertEqual(loaded_graph["1"]["2"]["color"], [0, 0, 255])


class TestDrawNxColoredGraph(TestCase):
    def setUp(self):
        self.G = Graph()
        self.G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        self.pos_G = nx.spring_layout(self.G)
        self.node_colors = {"red": [1, 2], "blue": [3, 4]}
        self.edge_widths = {
            2.0: [(1, 2), (3, 4)],
            3.0: [(2, 3)],
        }

    def test_valid_input(self):
        draw_nx_colored_graph(self.G, self.pos_G, self.node_colors, self.edge_widths)

    def test_invalid_graph_type(self):
        with self.assertRaises(ArgumentTypeError):
            draw_nx_colored_graph(
                "not_a_graph", self.pos_G, self.node_colors, self.edge_widths
            )

    def test_invalid_pos_G_type(self):
        with self.assertRaises(ArgumentTypeError):
            draw_nx_colored_graph(
                self.G, "not_a_dict", self.node_colors, self.edge_widths
            )

    def test_missing_nodes(self):
        self.node_colors["red"].append(5)  # Node 5 is not in the graph
        with self.assertRaises(ArgumentValueError):
            draw_nx_colored_graph(
                self.G, self.pos_G, self.node_colors, self.edge_widths
            )

    def test_missing_edges(self):
        self.edge_widths[2.0].append((5, 6))  # Edge (5, 6) is not in the graph
        with self.assertRaises(ArgumentValueError):
            draw_nx_colored_graph(
                self.G, self.pos_G, self.node_colors, self.edge_widths
            )

    def test_custom_node_size(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, node_size=20
        )

    def test_custom_width_scale(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, width_scale=5.0
        )

    def test_edge_alpha(self):
        draw_nx_colored_graph(
            self.G, self.pos_G, self.node_colors, self.edge_widths, edge_alpha=0.5
        )


class TestDistributeItemsInCommunities(TestCase):
    def setUp(self):
        self.items = list(range(10))  # Example items [0, 1, 2, ..., 9]

    def test_equal_distribution(self):
        result = distribute_items_in_communities(self.items, n_communities=2)
        self.assertEqual(len(result), 2)  # Check correct number of communities
        self.assertEqual(len(result[0]), 5)  # Check size of each community
        self.assertEqual(len(result[1]), 5)

    def test_unequal_distribution(self):
        result = distribute_items_in_communities(self.items, n_communities=3)
        self.assertEqual(len(result), 3)  # Check correct number of communities
        sizes = [len(community) for community in result]
        self.assertIn(4, sizes)  # At least one community should have size 4
        self.assertIn(3, sizes)  # Others should have size 3

    def test_single_community(self):
        result = distribute_items_in_communities(self.items, n_communities=1)
        self.assertEqual(len(result), 1)  # Only one community
        self.assertEqual(
            len(result[0]), 10
        )  # All items should be in the first community

    def test_community_per_item(self):
        result = distribute_items_in_communities(
            self.items, n_communities=len(self.items)
        )
        self.assertEqual(len(result), len(self.items))  # One community per item
        for community in result:
            self.assertEqual(
                len(community), 1
            )  # Each community should contain one item

    def test_more_communities_than_items(self):
        with self.assertRaises(ArgumentValueError):
            distribute_items_in_communities(self.items, n_communities=15)

    def test_empty_items(self):
        with self.assertRaises(ArgumentValueError):
            distribute_items_in_communities([], n_communities=3)

    def test_invalid_number_of_communities(self):
        with self.assertRaises(ArgumentValueError):
            distribute_items_in_communities(self.items, n_communities=0)

    def test_random_distribution(self):
        np.random.seed(42)  # Fix the random seed for reproducibility
        result1 = distribute_items_in_communities(self.items, n_communities=2)
        result2 = distribute_items_in_communities(self.items, n_communities=2)
        self.assertNotEqual(result1, result2)  #


class TestLinkStrengthFunctions(TestCase):
    def setUp(self):
        self.G = Graph()
        self.G.add_edge(1, 2, weight=0.8)
        self.G.add_edge(2, 3, weight=0.5)
        self.G.add_edge(3, 4, weight=1.5)
        self.G.add_edge(4, 1, weight=1.2)
        self.community1 = [1, 2]
        self.community2 = [3, 4]
        self.communities = [self.community1, self.community2]

    def test_average_strength_within_community(self):
        result = average_strength_of_links_within_community(self.G, self.community1)
        expected = np.mean([0.8])
        self.assertAlmostEqual(result, expected)

    def test_average_strength_within_empty_community(self):
        result = average_strength_of_links_within_community(self.G, [])
        self.assertTrue(np.isnan(result))

    def test_average_strength_within_communities(self):
        result = average_strength_of_links_within_communities(self.G, self.communities)
        expected_mean = np.mean([0.8, 1.5])
        expected = {
            "mean": expected_mean,
            "std": np.std([0.8, 1.5]),
            "max": 1.5,
            "min": 0.8,
        }
        self.assertDictEqual(result, expected)

    def test_average_strength_from_community(self):
        result = average_strength_of_links_from_community(self.G, self.community1)
        expected = np.mean([0.85])
        self.assertAlmostEqual(result, expected)

    def test_average_strength_from_empty_community(self):
        result = average_strength_of_links_from_community(self.G, [])
        self.assertTrue(np.isnan(result))

    def test_average_strength_from_communities(self):
        result = average_strength_of_links_from_communities(self.G, self.communities)
        expected_mean = np.mean([0.85])
        expected = {
            "mean": expected_mean,
            "std": np.std([0.0]),
            "max": 0.85,
            "min": 0.85,
        }
        self.assertDictEqual(result, expected)

    def test_community_with_missing_edges(self):
        self.G.add_node(5)  # Add a disconnected node
        result_within = average_strength_of_links_within_community(self.G, [5])
        result_from = average_strength_of_links_from_community(self.G, [5])
        self.assertTrue(np.isnan(result_within))
        self.assertTrue(np.isnan(result_from))


class TestConvertColor(TestCase):
    def test_hex_colors(self):
        # Test 6-digit hex colors
        self.assertEqual(_convert_color("#FF0000"), (255, 0, 0))  # Red
        self.assertEqual(_convert_color("#00FF00"), (0, 255, 0))  # Green
        self.assertEqual(_convert_color("#0000FF"), (0, 0, 255))  # Blue
        self.assertEqual(_convert_color("#FFFFFF"), (255, 255, 255))  # White
        self.assertEqual(_convert_color("#000000"), (0, 0, 0))  # Black
        # Test 3-digit hex colors
        self.assertEqual(_convert_color("#F00"), (255, 0, 0))  # Red
        self.assertEqual(_convert_color("#0F0"), (0, 255, 0))  # Green
        self.assertEqual(_convert_color("#00F"), (0, 0, 255))  # Blue
        self.assertEqual(_convert_color("#FFF"), (255, 255, 255))  # White
        self.assertEqual(_convert_color("#000"), (0, 0, 0))  # Black
        # Test hex colors with spaces and mixed case
        self.assertEqual(_convert_color(" #FF0000 "), (255, 0, 0))
        self.assertEqual(_convert_color("#ff0000"), (255, 0, 0))
        self.assertEqual(_convert_color("#Ff0000"), (255, 0, 0))
        # Test invalid hex colors
        with self.assertRaises(ArgumentValueError):
            _convert_color("#FF")  # Too short
        with self.assertRaises(ArgumentValueError):
            _convert_color("#FFFF")  # Invalid length
        with self.assertRaises(ArgumentValueError):
            _convert_color("#FF00000")  # Too long
        with self.assertRaises(ArgumentValueError):
            _convert_color("#FF000G")  # Invalid character
        with self.assertRaises(ArgumentValueError):
            _convert_color("FF0000")  # Missing #

    def test_rgb_colors(self):
        # Test RGB colors
        self.assertEqual(_convert_color("rgb(255, 0, 0)"), (255, 0, 0))  # Red
        self.assertEqual(_convert_color("rgb(0, 255, 0)"), (0, 255, 0))  # Green
        self.assertEqual(_convert_color("rgb(0, 0, 255)"), (0, 0, 255))  # Blue
        self.assertEqual(_convert_color("rgb(255, 255, 255)"), (255, 255, 255))  # White
        self.assertEqual(_convert_color("rgb(0, 0, 0)"), (0, 0, 0))  # Black
        # Test RGBA colors
        self.assertEqual(
            _convert_color("rgba(255, 0, 0, 1)"), (255, 0, 0, 1.0)
        )  # Red with alpha
        self.assertEqual(
            _convert_color("rgba(0, 255, 0, 0.5)"), (0, 255, 0, 0.5)
        )  # Green with alpha
        # Test RGB/RGBA colors with spaces
        self.assertEqual(_convert_color(" rgb(255, 0, 0) "), (255, 0, 0))
        self.assertEqual(_convert_color("rgba(255, 0, 0, 1) "), (255, 0, 0, 1.0))

    def test_invalid_rgb_colors(self):
        # Test invalid RGB/RGBA colors
        with self.assertRaises(ArgumentValueError):
            _convert_color("rgb(255)")  # Too few values
        with self.assertRaises(ArgumentValueError):
            _convert_color("rgb(255, 0, 0, 1)")  # Too many values
        with self.assertRaises(ArgumentValueError):
            _convert_color("rgb(255, 0, 0, 1, 2)")  # Too many values
        with self.assertRaises(ArgumentValueError):
            _convert_color("rgb(255, 0, 0, x)")  # Invalid value
        with self.assertRaises(ArgumentValueError):
            _convert_color("rgb(255, 0, 0")  # Missing closing parenthesis

    def test_tuple_list_colors(self):
        # Test tuple colors
        self.assertEqual(_convert_color((255, 0, 0)), (255, 0, 0))  # Red
        self.assertEqual(_convert_color((0, 255, 0)), (0, 255, 0))  # Green
        self.assertEqual(_convert_color((0, 0, 255)), (0, 0, 255))  # Blue
        self.assertEqual(_convert_color((255, 255, 255)), (255, 255, 255))  # White
        self.assertEqual(_convert_color((0, 0, 0)), (0, 0, 0))  # Black
        # Test list colors
        self.assertEqual(_convert_color([255, 0, 0]), (255, 0, 0))  # Red
        self.assertEqual(_convert_color([0, 255, 0]), (0, 255, 0))  # Green
        self.assertEqual(_convert_color([0, 0, 255]), (0, 0, 255))  # Blue
        self.assertEqual(_convert_color([255, 255, 255]), (255, 255, 255))  # White
        self.assertEqual(_convert_color([0, 0, 0]), (0, 0, 0))  # Black

    def test_invalid_tuple_list_colors(self):
        with self.assertRaises(ArgumentValueError):
            _convert_color((255, 0))  # Too few values
        with self.assertRaises(ArgumentValueError):
            _convert_color((255, 0, 0, 1, 2))  # Too many values
        with self.assertRaises(ArgumentValueError):
            _convert_color((255, 0, "x"))  # Invalid value
        with self.assertRaises(ArgumentValueError):
            _convert_color([255, 0])  # Too few values
        with self.assertRaises(ArgumentValueError):
            _convert_color([255, 0, 0, 1, 2])  # Too many values
        with self.assertRaises(ArgumentValueError):
            _convert_color([255, 0, "x"])  # Invalid value

    def test_other_color_formats(self):
        # Test named colors
        self.assertEqual(_convert_color("red"), (255, 0, 0))
        self.assertEqual(_convert_color("blue"), (0, 0, 255))
        self.assertEqual(_convert_color("green"), (0, 128, 0))

    def test_invalid_color_formats(self):
        with self.assertRaises(ArgumentValueError):
            _convert_color(123)  # Number
        with self.assertRaises(ArgumentValueError):
            _convert_color(None)  # None
        with self.assertRaises(ArgumentValueError):
            _convert_color(True)  # Boolean
        with self.assertRaises(ArgumentValueError):
            _convert_color(1.5)  # Float
        with self.assertRaises(ArgumentValueError):
            _convert_color("invalid")  # Invalid string
        with self.assertRaises(ArgumentValueError):
            _convert_color("")  # Empty string
        with self.assertRaises(ArgumentValueError):
            _convert_color(" ")  # Whitespace string


if __name__ == "__main__":
    unittest.main()
