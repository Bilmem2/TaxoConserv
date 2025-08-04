import unittest
import pandas as pd
import numpy as np
import web_taxoconserv

class TestWebTaxoConservFunctions(unittest.TestCase):
    def setUp(self):
        self.demo_data = web_taxoconserv.create_demo_data()
        self.score_column = 'conservation_score'
        self.group_column = 'taxon_group'

    def test_create_demo_data(self):
        df = self.demo_data
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(self.score_column, df.columns)
        self.assertIn(self.group_column, df.columns)
        self.assertGreater(len(df), 0)

    def test_perform_statistical_analysis_multi_group(self):
        results = web_taxoconserv.perform_statistical_analysis(self.demo_data, self.score_column, self.group_column)
        self.assertIsInstance(results, dict)
        self.assertIn('stats_summary', results)
        self.assertIn('h_statistic', results)
        self.assertIn('p_value', results)
        self.assertGreaterEqual(results['n_groups'], 2)

    def test_perform_statistical_analysis_single_group(self):
        df = self.demo_data.copy()
        df[self.group_column] = 'SingleGroup'
        results = web_taxoconserv.perform_statistical_analysis(df, self.score_column, self.group_column)
        self.assertIsInstance(results, dict)
        self.assertEqual(results['n_groups'], 1)
        self.assertEqual(results['h_statistic'], 0)
        self.assertEqual(results['p_value'], 1)

    def test_create_visualization_static(self):
        fig, mode = web_taxoconserv.create_visualization(self.demo_data, self.score_column, self.group_column, 'boxplot', 'Set3', interactive=False)
        self.assertIsNotNone(fig)
        self.assertEqual(mode, 'static')

    def test_create_visualization_interactive(self):
        fig, mode = web_taxoconserv.create_visualization(self.demo_data, self.score_column, self.group_column, 'boxplot', 'Set3', interactive=True)
        self.assertIsNotNone(fig)
        self.assertEqual(mode, 'interactive')

    def test_create_visualization_invalid_plot(self):
        fig, mode = web_taxoconserv.create_visualization(self.demo_data, self.score_column, self.group_column, 'invalid_plot', 'Set3', interactive=False)
        self.assertIsNone(fig)
        self.assertIsNone(mode)

    def test_perform_statistical_analysis_invalid_column(self):
        with self.assertRaises(Exception):
            web_taxoconserv.perform_statistical_analysis(self.demo_data, 'invalid_column', self.group_column)

if __name__ == '__main__':
    unittest.main()
