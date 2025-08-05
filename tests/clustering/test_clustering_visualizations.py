import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse

from mitoolspro.clustering.clustering_visualizations import (
    _create_figure,
    plot_silhouette_scores,
    plot_inertia,
    plot_clustering_ncluster_search,
    plot_df_col_distribution,
    plot_dfs_col_distribution,
    plot_clusters,
    plot_clusters_groupings,
    confidence_ellipse,
    add_clusters_ellipse,
    add_clusters_centroids,
    plot_clusters_growth,
    plot_clusters_growth_stacked,
    X_Y_SIZE_ERROR
)
from mitoolspro.exceptions import ArgumentStructureError


class TestCreateFigure(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
    
    def test_create_figure_with_inertia(self):
        fig, axes = _create_figure(with_inertia=True)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 2)
        self.assertIsInstance(axes[0], Axes)
        self.assertIsInstance(axes[1], Axes)
        
    def test_create_figure_without_inertia(self):
        fig, axes = _create_figure(with_inertia=False)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(axes), 1)
        self.assertIsInstance(axes[0], Axes)


class TestPlotSilhouetteScores(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_silhouette_scores_basic(self):
        silhouette_scores = [0.5, 0.6, 0.4, 0.7]
        algorithm_name = "K-Means"
        
        plot_silhouette_scores(self.ax, silhouette_scores, algorithm_name)
        
        # Check that the plot was created (at least one line)
        lines = self.ax.get_lines()
        self.assertGreater(len(lines), 0)
        
        # Check title and labels
        self.assertEqual(self.ax.get_title(), "K-Means Silhouette Score")
        self.assertEqual(self.ax.get_xlabel(), "N° of Clusters")
        self.assertEqual(self.ax.get_ylabel(), "Silhouette Score")
        
        # Check x-ticks
        expected_xticks = [2, 3, 4, 5]
        actual_xticks = self.ax.get_xticks().astype(int)
        np.testing.assert_array_equal(actual_xticks, expected_xticks)
    
    def test_plot_silhouette_scores_empty_list(self):
        silhouette_scores = []
        algorithm_name = "Empty Test"
        
        plot_silhouette_scores(self.ax, silhouette_scores, algorithm_name)
        
        # Should still set title and labels
        self.assertEqual(self.ax.get_title(), "Empty Test Silhouette Score")
        self.assertEqual(self.ax.get_xlabel(), "N° of Clusters")
        self.assertEqual(self.ax.get_ylabel(), "Silhouette Score")


class TestPlotInertia(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_inertia_basic(self):
        inertia = [100, 50, 30, 20, 15]
        algorithm_name = "K-Means"
        max_clusters = 7
        
        plot_inertia(self.ax, inertia, algorithm_name, max_clusters)
        
        # Check title and labels
        self.assertEqual(self.ax.get_title(), "K-Means Inertia")
        self.assertEqual(self.ax.get_xlabel(), "N° of Clusters")
        
        # Check that legend was added
        legend = self.ax.get_legend()
        self.assertIsNotNone(legend)
        
        # Check that at least one line exists (inertia line)
        lines = self.ax.get_lines()
        self.assertGreater(len(lines), 0)
    
    def test_plot_inertia_with_small_dataset(self):
        inertia = [200, 100, 60, 40, 30]  # Need more points for elbow calculation
        algorithm_name = "Mini K-Means"
        max_clusters = 7
        
        plot_inertia(self.ax, inertia, algorithm_name, max_clusters)
        
        self.assertEqual(self.ax.get_title(), "Mini K-Means Inertia")


class TestPlotClusteringNclusterSearch(unittest.TestCase):
    def tearDown(self):
        plt.close('all')
    
    def test_plot_clustering_ncluster_search_with_inertia(self):
        silhouette_scores = [0.5, 0.6, 0.4, 0.7, 0.3]
        inertia = [200, 100, 60, 40, 30]
        
        axes = plot_clustering_ncluster_search(
            silhouette_scores=silhouette_scores,
            inertia=inertia,
            max_clusters=7,
            algorithm_name="Test Algorithm"
        )
        
        self.assertEqual(len(axes), 2)
        self.assertIsInstance(axes[0], Axes)
        self.assertIsInstance(axes[1], Axes)
        
        # Check silhouette plot
        self.assertEqual(axes[0].get_title(), "Test Algorithm Silhouette Score")
        
        # Check inertia plot
        self.assertEqual(axes[1].get_title(), "Test Algorithm Inertia")
    
    def test_plot_clustering_ncluster_search_without_inertia(self):
        silhouette_scores = [0.5, 0.6, 0.4]
        
        axes = plot_clustering_ncluster_search(
            silhouette_scores=silhouette_scores,
            inertia=None,
            algorithm_name="Test Algorithm"
        )
        
        self.assertEqual(len(axes), 1)
        self.assertIsInstance(axes[0], Axes)
        self.assertEqual(axes[0].get_title(), "Test Algorithm Silhouette Score")


class TestPlotDfColDistribution(unittest.TestCase):
    def setUp(self):
        # Create test dataframe
        np.random.seed(42)
        self.df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(2, 1.5, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_df_col_distribution_by_name(self):
        ax = plot_df_col_distribution(self.df, 'feature_1')
        
        self.assertIsInstance(ax, Axes)
        
        # Check title
        self.assertEqual(ax.get_title(), "Distributions of Feature 1")
        self.assertEqual(ax.get_xlabel(), "Feature 1")
        self.assertEqual(ax.get_ylabel(), "Frequency")
    
    def test_plot_df_col_distribution_by_index(self):
        ax = plot_df_col_distribution(self.df, 0)
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_df_col_distribution_with_bins(self):
        ax = plot_df_col_distribution(self.df, 'feature_1', bins=20)
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_df_col_distribution_normed(self):
        ax = plot_df_col_distribution(self.df, 'feature_1', normed=True)
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_df_col_distribution_with_custom_ax(self):
        fig, custom_ax = plt.subplots()
        
        ax = plot_df_col_distribution(self.df, 'feature_1', ax=custom_ax)
        
        self.assertEqual(ax, custom_ax)


class TestPlotDfsColDistribution(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.df1 = pd.DataFrame({'feature': np.random.normal(0, 1, 50)})
        self.df2 = pd.DataFrame({'feature': np.random.normal(2, 1, 50)})
        self.dataframes = [self.df1, self.df2]
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_dfs_col_distribution_basic(self):
        ax = plot_dfs_col_distribution(self.dataframes, 'feature')
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_dfs_col_distribution_with_custom_ax(self):
        fig, custom_ax = plt.subplots()
        
        ax = plot_dfs_col_distribution(self.dataframes, 'feature', ax=custom_ax)
        
        self.assertEqual(ax, custom_ax)


class TestPlotClusters(unittest.TestCase):
    def setUp(self):
        # Create test dataframe with MultiIndex
        np.random.seed(42)
        index = pd.MultiIndex.from_tuples([
            ('cluster_0', 'item_1'), ('cluster_0', 'item_2'),
            ('cluster_1', 'item_3'), ('cluster_1', 'item_4'),
            ('cluster_2', 'item_5'), ('cluster_2', 'item_6')
        ], names=['cluster', 'item'])
        
        self.data = pd.DataFrame({
            'x_coord': np.random.random(6),
            'y_coord': np.random.random(6)
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_clusters_basic(self):
        ax = plot_clusters(
            self.data, 
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord'
        )
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_clusters_with_custom_ax(self):
        fig, custom_ax = plt.subplots()
        
        ax = plot_clusters(
            self.data,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord',
            ax=custom_ax
        )
        
        self.assertEqual(ax, custom_ax)
    
    def test_plot_clusters_with_custom_labels_colors(self):
        custom_labels = ['A', 'B', 'C']
        custom_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        
        ax = plot_clusters(
            self.data,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord',
            labels=custom_labels,
            colors=custom_colors
        )
        
        self.assertIsInstance(ax, Axes)


class TestConfidenceEllipse(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        
    def tearDown(self):
        plt.close('all')
    
    def test_confidence_ellipse_basic(self):
        np.random.seed(42)
        x_values = np.random.normal(0, 1, 50)
        y_values = np.random.normal(0, 1, 50)
        
        ax = confidence_ellipse(x_values, y_values, self.ax)
        
        self.assertEqual(ax, self.ax)
        # Check that an ellipse patch was added
        patches = self.ax.patches
        self.assertEqual(len(patches), 1)
        self.assertIsInstance(patches[0], Ellipse)
    
    def test_confidence_ellipse_mismatched_sizes(self):
        x_values = np.array([1, 2, 3])
        y_values = np.array([1, 2])  # Different size
        
        with self.assertRaises(ArgumentStructureError) as cm:
            confidence_ellipse(x_values, y_values, self.ax)
        
        self.assertEqual(str(cm.exception), X_Y_SIZE_ERROR)
    
    def test_confidence_ellipse_with_custom_params(self):
        np.random.seed(42)
        x_values = np.random.normal(0, 1, 50)
        y_values = np.random.normal(0, 1, 50)
        
        ax = confidence_ellipse(
            x_values, y_values, self.ax,
            n_std=2.0,
            facecolor='red',
            edgecolor='blue'
        )
        
        patches = self.ax.patches
        self.assertEqual(len(patches), 1)
        ellipse = patches[0]
        self.assertEqual(ellipse.get_facecolor(), (1.0, 0.0, 0.0, 1.0))  # Red


class TestAddClustersEllipse(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        
        # Create test data
        index = pd.MultiIndex.from_tuples([
            ('cluster_0', 'item_1'), ('cluster_0', 'item_2'),
            ('cluster_1', 'item_3'), ('cluster_1', 'item_4')
        ], names=['cluster', 'item'])
        
        np.random.seed(42)
        self.data = pd.DataFrame({
            'x_coord': np.random.random(4),
            'y_coord': np.random.random(4)
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_add_clusters_ellipse_basic(self):
        ax = add_clusters_ellipse(
            self.ax, self.data,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord'
        )
        
        self.assertEqual(ax, self.ax)


class TestAddClustersCentroids(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        
        # Create test centroids data
        index = pd.MultiIndex.from_tuples([
            ('cluster_0', 'centroid'), ('cluster_1', 'centroid')
        ], names=['cluster', 'type'])
        
        self.centroids = pd.DataFrame({
            'x_coord': [0.5, 1.5],
            'y_coord': [0.3, 0.8]
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_add_clusters_centroids_basic(self):
        ax = add_clusters_centroids(
            self.ax, self.centroids,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord'
        )
        
        self.assertEqual(ax, self.ax)
    
    def test_add_clusters_centroids_with_custom_params(self):
        custom_labels = ['A', 'B']
        custom_colors = [(1, 0, 0), (0, 1, 0)]
        
        ax = add_clusters_centroids(
            self.ax, self.centroids,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord',
            labels=custom_labels,
            colors=custom_colors,
            marker='x',
            s=100
        )
        
        self.assertEqual(ax, self.ax)


class TestPlotClustersGrowth(unittest.TestCase):
    def setUp(self):
        # Create time-series cluster data
        dates = pd.date_range('2020-01-01', periods=12, freq='M')
        clusters = ['cluster_0', 'cluster_1', 'cluster_2']
        
        index_data = []
        for date in dates[:6]:  # First 6 months
            for cluster in clusters:
                for i in range(np.random.randint(5, 15)):  # Random number of items
                    index_data.append((date.year, cluster, f'item_{i}'))
        
        index = pd.MultiIndex.from_tuples(
            index_data, names=['year', 'cluster', 'item']
        )
        
        self.data = pd.DataFrame({
            'value': np.random.random(len(index))
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_clusters_growth_basic(self):
        ax = plot_clusters_growth(
            self.data,
            time_level='year',
            cluster_level='cluster'
        )
        
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Cluster Size Evolution")
        self.assertEqual(ax.get_xlabel(), "Year")
        self.assertEqual(ax.get_ylabel(), "N° Elements")
        
        # Should have legend
        legend = ax.get_legend()
        self.assertIsNotNone(legend)


class TestPlotClustersGrowthStacked(unittest.TestCase):
    def setUp(self):
        # Create time-series cluster data
        years = [2020, 2021, 2022]
        clusters = ['Topic_A', 'Topic_B', 'Topic_C']
        
        index_data = []
        for year in years:
            for cluster in clusters:
                for i in range(np.random.randint(10, 30)):
                    index_data.append((year, cluster, f'item_{i}'))
        
        index = pd.MultiIndex.from_tuples(
            index_data, names=['year', 'cluster', 'item']
        )
        
        self.data = pd.DataFrame({
            'value': np.random.random(len(index))
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_clusters_growth_stacked_basic(self):
        ax = plot_clusters_growth_stacked(
            self.data,
            time_level='year',
            cluster_level='cluster'
        )
        
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_title(), "Stacked Cluster Size Evolution")
        self.assertEqual(ax.get_xlabel(), "Year")
        self.assertEqual(ax.get_ylabel(), "N° Elements")
    
    def test_plot_clusters_growth_stacked_with_filters(self):
        filtered_clusters = ['Topic_A']
        
        ax = plot_clusters_growth_stacked(
            self.data,
            time_level='year',
            cluster_level='cluster',
            filtered_clusters=filtered_clusters
        )
        
        self.assertIsInstance(ax, Axes)
    
    def test_plot_clusters_growth_stacked_percentage(self):
        ax = plot_clusters_growth_stacked(
            self.data,
            time_level='year',
            cluster_level='cluster',
            share_pct=True
        )
        
        self.assertIsInstance(ax, Axes)


class TestPlotClustersGroupings(unittest.TestCase):
    def setUp(self):
        # Create complex multi-level index data
        np.random.seed(42)
        
        index_data = []
        for cluster in range(3):
            for group in range(10):  # 0-9 for group values
                for item in range(2):
                    index_data.append((f'cluster_{cluster}', group, f'item_{item}'))
        
        index = pd.MultiIndex.from_tuples(
            index_data, names=['cluster', 'group', 'item']
        )
        
        self.data = pd.DataFrame({
            'x_coord': np.random.random(len(index)),
            'y_coord': np.random.random(len(index))
        }, index=index)
        
    def tearDown(self):
        plt.close('all')
    
    def test_plot_clusters_groupings_basic(self):
        axes = plot_clusters_groupings(
            self.data,
            cluster_level='cluster',
            x_col='x_coord',
            y_col='y_coord',
            group_level='group',
            group_value_ranges=(3, 7)  # Split at groups 3-7 and 7+
        )
        
        self.assertIsInstance(axes, dict)
        self.assertIn('a', axes)
        self.assertIn('b', axes)
        self.assertIn('c', axes)


if __name__ == '__main__':
    unittest.main()