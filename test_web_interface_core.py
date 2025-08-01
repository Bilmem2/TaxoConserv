import pandas as pd
import numpy as np
from src.input_parser import parse_input, validate_input
from src.taxon_grouping import group_by_taxon, calculate_stats
from src.stats_tests import run_kruskal_wallis
from src.visualization import generate_visualization
from web_taxoconserv import create_demo_data, perform_statistical_analysis, create_visualization

def test_demo_data_analysis():
    print("Testing demo data analysis...")
    data = create_demo_data()
    assert isinstance(data, pd.DataFrame)
    score_col = 'conservation_score'
    group_col = 'taxon_group'
    # Test statistical analysis
    results = perform_statistical_analysis(data, score_col, group_col)
    assert results is not None
    assert 'stats_summary' in results
    assert results['n_groups'] > 1
    print("Demo data analysis passed.")

def test_visualization():
    print("Testing visualization...")
    data = create_demo_data()
    score_col = 'conservation_score'
    group_col = 'taxon_group'
    fig, plot_type = create_visualization(data, score_col, group_col, 'boxplot', 'Set3', interactive=False)
    assert fig is not None
    print("Visualization test passed.")

def test_csv_load_and_analysis():
    print("Testing CSV load and analysis...")
    # Create a temporary CSV
    df = create_demo_data().copy()
    temp_path = "temp_test_data.csv"
    df.to_csv(temp_path, index=False)
    loaded = pd.read_csv(temp_path)
    assert loaded.shape == df.shape
    score_col = 'conservation_score'
    group_col = 'taxon_group'
    results = perform_statistical_analysis(loaded, score_col, group_col)
    assert results is not None
    print("CSV load and analysis test passed.")

if __name__ == "__main__":
    test_demo_data_analysis()
    test_visualization()
    test_csv_load_and_analysis()
    print("All core tests passed.")
