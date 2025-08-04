#!/usr/bin/env python3
"""
Comprehensive test script for taxon_grouping module
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from input_parser import parse_input, validate_input
from taxon_grouping import group_by_taxon, calculate_stats, get_group_summary

def test_taxon_grouping_comprehensive():
    """Test the taxon_grouping module with different scenarios."""
    
    print("🧪 Testing TaxoConserv Taxon Grouping Module")
    print("=" * 60)
    
    # Test 1: Normal case with example data
    print("\n📋 Test 1: Normal case with example data")
    try:
        df = parse_input("data/example_conservation_scores.csv")
        validated_df = validate_input(df)
        
        grouped_data = group_by_taxon(validated_df)
        stats_df = calculate_stats(grouped_data)
        
        print(f"✅ Success: {len(grouped_data)} groups, {len(stats_df)} statistics")
        print(f"   Groups: {list(grouped_data.keys())}")
        print(f"   Sample sizes: {dict(zip(stats_df['taxon_group'], stats_df['sample_size']))}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
    
    # Test 2: Empty DataFrame
    print("\n📋 Test 2: Empty DataFrame")
    try:
        empty_df = pd.DataFrame()
        grouped_data = group_by_taxon(empty_df)
        print("❌ Test 2 should have failed")
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: DataFrame with missing taxon_group column
    print("\n📋 Test 3: Missing taxon_group column")
    try:
        df_no_taxon = pd.DataFrame({
            'position': [1000, 2000, 3000],
            'conservation_score': [5.0, 6.0, 7.0],
            'gene': ['GENE1', 'GENE2', 'GENE3']
        })
        grouped_data = group_by_taxon(df_no_taxon)
        print("❌ Test 3 should have failed")
    except ValueError as e:
        print(f"✅ Expected error caught: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 4: DataFrame with NaN values in taxon_group
    print("\n📋 Test 4: NaN values in taxon_group")
    try:
        df_with_nan = pd.DataFrame({
            'position': [1000, 2000, 3000, 4000],
            'conservation_score': [5.0, 6.0, 7.0, 8.0],
            'taxon_group': ['primate', 'mammal', np.nan, 'vertebrate'],
            'gene': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
            'species': ['human', 'mouse', 'unknown', 'fish']
        })
        
        grouped_data = group_by_taxon(df_with_nan)
        stats_df = calculate_stats(grouped_data)
        
        print(f"✅ Success: {len(grouped_data)} groups created, NaN values handled")
        print(f"   Groups: {list(grouped_data.keys())}")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
    
    # Test 5: Single group data
    print("\n📋 Test 5: Single group data")
    try:
        df_single = pd.DataFrame({
            'position': [1000, 2000, 3000],
            'conservation_score': [5.0, 6.0, 7.0],
            'taxon_group': ['primate', 'primate', 'primate'],
            'gene': ['GENE1', 'GENE2', 'GENE3'],
            'species': ['human', 'chimp', 'bonobo']
        })
        
        grouped_data = group_by_taxon(df_single)
        stats_df = calculate_stats(grouped_data)
        
        print(f"✅ Success: {len(grouped_data)} group, {len(stats_df)} statistics")
        print(f"   Mean score: {stats_df['mean_score'].iloc[0]:.3f}")
        print(f"   Std dev: {stats_df['std_dev'].iloc[0]:.3f}")
        
    except Exception as e:
        print(f"❌ Test 5 failed: {e}")
    
    # Test 6: Missing conservation_score values
    print("\n📋 Test 6: Missing conservation_score values")
    try:
        df_missing_scores = pd.DataFrame({
            'position': [1000, 2000, 3000, 4000],
            'conservation_score': [5.0, np.nan, 7.0, np.nan],
            'taxon_group': ['primate', 'mammal', 'primate', 'vertebrate'],
            'gene': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
            'species': ['human', 'mouse', 'chimp', 'fish']
        })
        
        grouped_data = group_by_taxon(df_missing_scores)
        stats_df = calculate_stats(grouped_data)
        
        print(f"✅ Success: {len(grouped_data)} groups, missing scores handled")
        print(f"   Sample sizes: {dict(zip(stats_df['taxon_group'], stats_df['sample_size']))}")
        
        # Check for NaN values in statistics
        nan_stats = stats_df[stats_df['mean_score'].isna()]
        if len(nan_stats) > 0:
            print(f"   Groups with no valid scores: {list(nan_stats['taxon_group'])}")
        
    except Exception as e:
        print(f"❌ Test 6 failed: {e}")
    
    # Test 7: Test with TSV data
    print("\n📋 Test 7: TSV data format")
    try:
        df_tsv = parse_input("data/example_conservation_scores.tsv")
        validated_df_tsv = validate_input(df_tsv)
        
        grouped_data = group_by_taxon(validated_df_tsv)
        stats_df = calculate_stats(grouped_data)
        
        print(f"✅ Success: TSV format processed, {len(grouped_data)} groups")
        print(f"   Groups: {list(grouped_data.keys())}")
        
    except Exception as e:
        print(f"❌ Test 7 failed: {e}")
    
    # Test 8: Custom score column name
    print("\n📋 Test 8: Custom score column name")
    try:
        df_custom = pd.DataFrame({
            'position': [1000, 2000, 3000],
            'custom_score': [5.0, 6.0, 7.0],
            'taxon_group': ['primate', 'mammal', 'primate'],
            'gene': ['GENE1', 'GENE2', 'GENE3'],
            'species': ['human', 'mouse', 'chimp']
        })
        
        grouped_data = group_by_taxon(df_custom)
        stats_df = calculate_stats(grouped_data, score_column="custom_score")
        
        print(f"✅ Success: Custom score column used")
        print(f"   Mean scores: {dict(zip(stats_df['taxon_group'], stats_df['mean_score'].round(3)))}")
        
    except Exception as e:
        print(f"❌ Test 8 failed: {e}")
    
    print("\n🎉 Taxon Grouping Testing Complete!")

if __name__ == "__main__":
    test_taxon_grouping_comprehensive()
