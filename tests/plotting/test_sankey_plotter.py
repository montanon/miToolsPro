import json
import unittest

import numpy as np
import pandas as pd

from mitoolspro.exceptions import ArgumentValueError
from mitoolspro.plotting.plots.sankey_plotter import (
    PLAIN_GRAY_COLOR,
    SankeyColumn,
    SankeyLink,
    SankeyNode,
    SankeyPlotter,
    SankeySinkNode,
    _scale_array,
)


class TestScaleArray(unittest.TestCase):
    def test_scale_array_ascending(self):
        array = np.array([1, 2, 3, 4, 5])
        scaled = _scale_array(array, ascending=True)
        self.assertTrue(np.all(scaled >= 0.001))
        self.assertTrue(np.all(scaled <= 0.999))
        self.assertTrue(np.all(np.diff(scaled) > 0))

    def test_scale_array_descending(self):
        array = np.array([1, 2, 3, 4, 5])
        scaled = _scale_array(array, ascending=False)
        self.assertTrue(np.all(scaled >= 0.001))
        self.assertTrue(np.all(scaled <= 0.999))
        self.assertTrue(np.all(np.diff(scaled) < 0))

    def test_scale_array_single_value(self):
        array = np.array([5])
        scaled = _scale_array(array)
        self.assertEqual(scaled.shape, (1,))
        self.assertEqual(scaled.dtype, np.float64)
        self.assertEqual(scaled[0], 0.001)

    def test_scale_array_constant(self):
        array = np.array([5, 5, 5])
        scaled = _scale_array(array)
        self.assertEqual(scaled.shape, (3,))
        self.assertEqual(scaled.dtype, np.float64)
        self.assertTrue(np.all(scaled == 0.001))


class TestSankeyNode(unittest.TestCase):
    def setUp(self):
        self.node = SankeyNode("Test", 10.0, 1, 1)

    def test_initialization(self):
        self.assertEqual(self.node.name, "Test")
        self.assertEqual(self.node.count, 10.0)
        self.assertEqual(self.node.period, 1)
        self.assertEqual(self.node.rank, 1)
        self.assertIsNone(self.node.id)
        self.assertIsNone(self.node.x_pos)
        self.assertIsNone(self.node.y_pos)
        self.assertIsNone(self.node.color)

    def test_to_dict(self):
        self.node.id = 1
        self.node.x_pos = 0.5
        self.node.y_pos = 0.5
        self.node.color = "red"
        result = self.node.to_dict()
        expected = {
            "name": "Test",
            "count": 10.0,
            "period": 1,
            "rank": 1,
            "id": 1,
            "x_pos": 0.5,
            "y_pos": 0.5,
            "color": "red",
        }
        self.assertEqual(result, expected)

    def test_from_dict(self):
        data = {
            "name": "Test",
            "count": 10.0,
            "period": 1,
            "rank": 1,
            "id": 1,
            "x_pos": 0.5,
            "y_pos": 0.5,
            "color": "red",
        }
        node = SankeyNode.from_dict(data)
        self.assertEqual(node.name, "Test")
        self.assertEqual(node.count, 10.0)
        self.assertEqual(node.period, 1)
        self.assertEqual(node.rank, 1)
        self.assertEqual(node.id, 1)
        self.assertEqual(node.x_pos, 0.5)
        self.assertEqual(node.y_pos, 0.5)
        self.assertEqual(node.color, "red")


class TestSankeySinkNode(unittest.TestCase):
    def setUp(self):
        self.sink_node = SankeySinkNode(1)

    def test_initialization(self):
        self.assertEqual(self.sink_node.name, "")
        self.assertEqual(self.sink_node.count, 1e-5)
        self.assertEqual(self.sink_node.period, 1)
        self.assertEqual(self.sink_node.rank, -1)
        self.assertEqual(
            self.sink_node.color,
            f"rgba({PLAIN_GRAY_COLOR[0]},{PLAIN_GRAY_COLOR[1]},{PLAIN_GRAY_COLOR[2]},{PLAIN_GRAY_COLOR[3]})",
        )


class TestSankeyLink(unittest.TestCase):
    def setUp(self):
        self.source = SankeyNode("Source", 10.0, 1, 1)
        self.target = SankeyNode("Target", 5.0, 2, 1)
        self.link = SankeyLink(self.source, self.target, 5.0)

    def test_initialization(self):
        self.assertEqual(self.link.source, self.source)
        self.assertEqual(self.link.target, self.target)
        self.assertEqual(self.link.value, 5.0)
        self.assertIsNone(self.link.color)

    def test_same_period_error(self):
        target = SankeyNode("Target", 5.0, 1, 1)
        with self.assertRaises(ArgumentValueError):
            SankeyLink(self.source, target, 5.0)

    def test_zero_value_error(self):
        with self.assertRaises(ArgumentValueError):
            SankeyLink(self.source, self.target, 0)

    def test_negative_value_error(self):
        with self.assertRaises(ArgumentValueError):
            SankeyLink(self.source, self.target, -5.0)

    def test_to_dict(self):
        self.link.color = "red"
        result = self.link.to_dict()
        expected = {
            "source": "Source",
            "target": "Target",
            "source_period": 1,
            "target_period": 2,
            "value": 5.0,
            "color": "red",
        }
        self.assertEqual(result, expected)


class TestSankeyColumn(unittest.TestCase):
    def setUp(self):
        self.column = SankeyColumn("TestColumn", 1)
        self.node1 = SankeyNode("Node1", 10.0, 1, 1)
        self.node2 = SankeyNode("Node2", 5.0, 1, 2)

    def test_initialization(self):
        self.assertEqual(self.column.name, "TestColumn")
        self.assertEqual(self.column.period, 1)
        self.assertEqual(self.column.nodes, [])

    def test_add_node(self):
        self.column.add_node("Node1", 10.0, 1, 1)
        self.assertEqual(len(self.column.nodes), 1)
        self.assertEqual(self.column.nodes[0].name, "Node1")
        self.assertEqual(self.column.nodes[0].count, 10.0)
        self.assertEqual(self.column.nodes[0].period, 1)
        self.assertEqual(self.column.nodes[0].rank, 1)

    def test_get_node(self):
        self.column.nodes = [self.node1, self.node2]
        result = self.column.get_node("Node1")
        self.assertEqual(result, self.node1)
        result = self.column.get_node("NonExistent")
        self.assertIsNone(result)

    def test_normalize_y_positions(self):
        self.column.nodes = [self.node1, self.node2]
        self.column.normalize_y_positions(ascending=True)
        y_positions = self.column.y_positions()
        self.assertTrue(all(0.001 <= y <= 0.999 for y in y_positions))
        self.assertTrue(y_positions[0] < y_positions[1])

    def test_set_x_positions(self):
        self.column.nodes = [self.node1, self.node2]
        self.column.set_x_positions(0.5)
        self.assertEqual(self.node1.x_pos, 0.5)
        self.assertEqual(self.node2.x_pos, 0.5)

    def test_set_y_positions(self):
        self.column.nodes = [self.node1, self.node2]
        self.column.set_y_positions([0.2, 0.8])
        self.assertEqual(self.node1.y_pos, 0.2)
        self.assertEqual(self.node2.y_pos, 0.8)

    def test_x_positions(self):
        self.column.nodes = [self.node1, self.node2]
        self.node1.x_pos = 0.2
        self.node2.x_pos = 0.8
        result = self.column.x_positions()
        self.assertEqual(result, [0.2, 0.8])

    def test_y_positions(self):
        self.column.nodes = [self.node1, self.node2]
        self.node1.y_pos = 0.2
        self.node2.y_pos = 0.8
        result = self.column.y_positions()
        self.assertEqual(result, [0.2, 0.8])

    def test_names(self):
        self.column.nodes = [self.node1, self.node2]
        result = self.column.names()
        self.assertEqual(result, ["Node1", "Node2"])


class TestSankeyPlotter(unittest.TestCase):
    def setUp(self):
        self.plotter = SankeyPlotter()
        self.column1 = SankeyColumn("Column1", 1)
        self.column2 = SankeyColumn("Column2", 2)
        self.node1 = SankeyNode("Node1", 10.0, 1, 1)
        self.node2 = SankeyNode("Node2", 5.0, 2, 1)
        self.column1.nodes = [self.node1]
        self.column2.nodes = [self.node2]

    def test_initialization(self):
        self.assertEqual(self.plotter.columns, {})
        self.assertEqual(self.plotter.column_order, [])
        self.assertEqual(self.plotter.links, [])
        self.assertEqual(self.plotter.sink_nodes, {})
        self.assertEqual(self.plotter.sink_links, [])

    def test_add_column(self):
        self.plotter.add_column(self.column1)
        self.assertEqual(self.plotter.columns[1], self.column1)
        self.assertEqual(self.plotter.column_order, [1])

    def test_get_column_by_index(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        result = self.plotter.get_column_by_index(0)
        self.assertEqual(result, self.column1)
        result = self.plotter.get_column_by_index(-1)
        self.assertEqual(result, self.column2)
        with self.assertRaises(ArgumentValueError):
            self.plotter.get_column_by_index(2)

    def test_get_column_index(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.assertEqual(self.plotter.get_column_index(1), 0)
        self.assertEqual(self.plotter.get_column_index(2), 1)
        with self.assertRaises(ArgumentValueError):
            self.plotter.get_column_index(3)

    def test_add_link(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        link = SankeyLink(self.node1, self.node2, 5.0)
        self.plotter.add_link(link)
        self.assertEqual(len(self.plotter.links), 1)
        self.assertEqual(self.plotter.links[0], link)

    def test_add_columns(self):
        self.plotter.add_columns([self.column1, self.column2])
        self.assertEqual(self.plotter.columns[1], self.column1)
        self.assertEqual(self.plotter.columns[2], self.column2)
        self.assertEqual(self.plotter.column_order, [1, 2])

    def test_add_links(self):
        link1 = SankeyLink(self.node1, self.node2, 5.0)
        link2 = SankeyLink(self.node2, self.node1, 3.0)
        self.plotter.add_links([link1, link2])
        self.assertEqual(len(self.plotter.links), 2)
        self.assertEqual(self.plotter.links[0], link1)
        self.assertEqual(self.plotter.links[1], link2)

    def test_connect_columns(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.plotter.connect_columns()
        self.assertEqual(len(self.plotter.links), 0)
        self.assertEqual(len(self.plotter.sink_links), 0)

    def test_assign_node_ids(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.plotter.assign_node_ids()
        self.assertEqual(self.node1.id, 0)
        self.assertEqual(self.node2.id, 1)

    def test_normalize_x_positions(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.plotter.normalize_x_positions()
        self.assertTrue(0.001 <= self.node1.x_pos <= 0.999)
        self.assertTrue(0.001 <= self.node2.x_pos <= 0.999)

    def test_normalize_positions(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.plotter.normalize_positions()
        self.assertTrue(0.001 <= self.node1.x_pos <= 0.999)
        self.assertTrue(0.001 <= self.node1.y_pos <= 0.999)
        self.assertTrue(0.001 <= self.node2.x_pos <= 0.999)
        self.assertTrue(0.001 <= self.node2.y_pos <= 0.999)

    def test_assign_colors(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        self.plotter.assign_colors()
        self.assertIsNotNone(self.node1.color)
        self.assertIsNotNone(self.node2.color)

    def test_to_json(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        json_str = self.plotter.to_json()
        data = json.loads(json_str)
        self.assertEqual(len(data["columns"]), 2)
        self.assertEqual(data["columns"][0]["name"], "Column1")
        self.assertEqual(data["columns"][1]["name"], "Column2")

    def test_from_json(self):
        json_str = self.plotter.to_json()
        new_plotter = SankeyPlotter.from_json(json_str)
        self.assertEqual(len(new_plotter.columns), 2)
        self.assertEqual(new_plotter.column_order, [1, 2])

    def test_to_dataframe(self):
        self.plotter.add_column(self.column1)
        self.plotter.add_column(self.column2)
        node_df, link_df = self.plotter.to_dataframe(include_links=True)
        self.assertEqual(len(node_df), 2)
        self.assertEqual(len(link_df), 0)

    def test_from_dataframe(self):
        node_df = pd.DataFrame(
            {
                "column_name": ["Column1", "Column2"],
                "name": ["Node1", "Node2"],
                "count": [10.0, 5.0],
                "period": [1, 2],
                "rank": [1, 1],
                "x_pos": [0.2, 0.8],
                "y_pos": [0.2, 0.8],
                "color": ["red", "blue"],
                "is_sink": [False, False],
            }
        )
        new_plotter = SankeyPlotter.from_dataframe(node_df)
        self.assertEqual(len(new_plotter.columns), 2)
        self.assertEqual(new_plotter.column_order, [1, 2])


if __name__ == "__main__":
    unittest.main()
