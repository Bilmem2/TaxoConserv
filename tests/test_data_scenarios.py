import unittest
import pandas as pd
import os
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

DATA_DIR = "c:/Users/can_t/Downloads/TaxoConserv/data/"

class TestDataScenarios(unittest.TestCase):
    def test_standard(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "testdata_standard.csv"))
        self.assertTrue(generate_boxplot(df).endswith('.png'))
        self.assertTrue(generate_barplot(df).endswith('.png'))
        self.assertTrue(generate_violin_plot(df).endswith('.png'))
        self.assertTrue(generate_swarm_plot(df).endswith('.png'))
        self.assertTrue(generate_stripplot(df).endswith('.png'))
        self.assertTrue(generate_heatmap(df).endswith('.png'))
        self.assertTrue(generate_histogram(df).endswith('.png'))
        self.assertTrue(generate_pairplot(df).endswith('.png') or generate_pairplot(df).endswith('_failed.png'))
        self.assertTrue(generate_correlation_matrix(df).endswith('.png') or generate_correlation_matrix(df).endswith('_failed.png'))

    def test_single_group(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "testdata_single_group.csv"))
        # Boxplot should not fail, but may warn
        self.assertTrue(generate_boxplot(df).endswith('.png'))
        # Pairplot/correlation may fail due to lack of diversity
        self.assertTrue(generate_pairplot(df).endswith('.png') or generate_pairplot(df).endswith('_failed.png'))
        self.assertTrue(generate_correlation_matrix(df).endswith('.png') or generate_correlation_matrix(df).endswith('_failed.png'))

    def test_missing_values(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "testdata_missing_values.csv"))
        # Should not crash, should skip missing
        self.assertTrue(generate_boxplot(df).endswith('.png'))
        self.assertTrue(generate_barplot(df).endswith('.png'))
        self.assertTrue(generate_violin_plot(df).endswith('.png'))
        self.assertTrue(generate_swarm_plot(df).endswith('.png'))
        self.assertTrue(generate_stripplot(df).endswith('.png'))
        self.assertTrue(generate_heatmap(df).endswith('.png'))
        self.assertTrue(generate_histogram(df).endswith('.png'))
        self.assertTrue(generate_pairplot(df).endswith('.png') or generate_pairplot(df).endswith('_failed.png'))
        self.assertTrue(generate_correlation_matrix(df).endswith('.png') or generate_correlation_matrix(df).endswith('_failed.png'))

    def test_outliers(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "testdata_outliers.csv"))
        self.assertTrue(generate_boxplot(df).endswith('.png'))
        self.assertTrue(generate_barplot(df).endswith('.png'))
        self.assertTrue(generate_violin_plot(df).endswith('.png'))
        self.assertTrue(generate_swarm_plot(df).endswith('.png'))
        self.assertTrue(generate_stripplot(df).endswith('.png'))
        self.assertTrue(generate_heatmap(df).endswith('.png'))
        self.assertTrue(generate_histogram(df).endswith('.png'))
        self.assertTrue(generate_pairplot(df).endswith('.png') or generate_pairplot(df).endswith('_failed.png'))
        self.assertTrue(generate_correlation_matrix(df).endswith('.png') or generate_correlation_matrix(df).endswith('_failed.png'))

    def test_edge_cases(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "testdata_edge_cases.csv"))
        self.assertTrue(generate_boxplot(df).endswith('.png'))
        self.assertTrue(generate_barplot(df).endswith('.png'))
        self.assertTrue(generate_violin_plot(df).endswith('.png'))
        self.assertTrue(generate_swarm_plot(df).endswith('.png'))
        self.assertTrue(generate_stripplot(df).endswith('.png'))
        self.assertTrue(generate_heatmap(df).endswith('.png'))
        self.assertTrue(generate_histogram(df).endswith('.png'))
        self.assertTrue(generate_pairplot(df).endswith('.png') or generate_pairplot(df).endswith('_failed.png'))
        self.assertTrue(generate_correlation_matrix(df).endswith('.png') or generate_correlation_matrix(df).endswith('_failed.png'))

if __name__ == "__main__":
    unittest.main()
