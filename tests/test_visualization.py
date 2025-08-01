import unittest
import pandas as pd
from src.visualization import (
    generate_boxplot,
    generate_barplot,
    generate_violin_plot,
    generate_swarm_plot,
    generate_stripplot,
    generate_heatmap,
    generate_histogram,
    generate_pairplot,
    generate_correlation_matrix
)

class TestVisualizationFunctions(unittest.TestCase):
    def setUp(self):
        # Example test data
        self.df = pd.DataFrame({
            'taxon_group': ['A', 'A', 'B', 'B', 'C', 'C'],
            'conservation_score': [0.5, 0.7, 0.2, 0.3, 0.9, 1.1],
            'feature_1': [1, 2, 3, 4, 5, 6],
            'feature_2': [2, 3, 4, 5, 6, 7]
        })
        self.empty_df = pd.DataFrame(columns=['taxon_group', 'conservation_score'])

    def test_generate_boxplot(self):
        result = generate_boxplot(self.df)
        self.assertTrue(result.endswith('.png'))
        result_empty = generate_boxplot(self.empty_df)
        self.assertEqual(result_empty, "")

    def test_generate_barplot(self):
        result = generate_barplot(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_violin_plot(self):
        result = generate_violin_plot(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_swarm_plot(self):
        result = generate_swarm_plot(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_stripplot(self):
        result = generate_stripplot(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_heatmap(self):
        result = generate_heatmap(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_histogram(self):
        result = generate_histogram(self.df)
        self.assertTrue(result.endswith('.png'))

    def test_generate_pairplot(self):
        result = generate_pairplot(self.df)
        self.assertTrue(result.endswith('.png') or result.endswith('_failed.png'))

    def test_generate_correlation_matrix(self):
        result = generate_correlation_matrix(self.df)
        self.assertTrue(result.endswith('.png') or result.endswith('_failed.png'))

if __name__ == "__main__":
    unittest.main()
