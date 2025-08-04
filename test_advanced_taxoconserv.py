#!/usr/bin/env python3
"""
Advanced and realistic test cases for TaxoConserv modules
"""
import sys
import os
import pandas as pd
import numpy as np
import time
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.input_parser import parse_input, validate_input
from src.taxon_grouping import group_by_taxon, calculate_stats

def test_large_dataset():
    print("\n📋 Test: Large dataset (10,000+ rows)")
    n = 12000
    df = pd.DataFrame({
        'taxon_group': np.random.choice(['Mammals', 'Birds', 'Reptiles', 'Amphibians', 'Fish'], n),
        'conservation_score': np.random.normal(loc=5, scale=2, size=n),
        'gene': [f'GENE{i}' for i in range(n)],
        'species': np.random.choice(['human', 'mouse', 'zebrafish', 'frog', 'lizard'], n)
    })
    start = time.time()
    grouped = group_by_taxon(df)
    stats = calculate_stats(grouped)
    elapsed = time.time() - start
    print(f"✅ Success: {len(grouped)} groups, {len(stats)} stats, time: {elapsed:.2f}s")


def test_edge_cases():
    print("\n📋 Test: Edge cases")
    # Non-numeric conservation_score
    df_non_numeric = pd.DataFrame({
        'taxon_group': ['Mammals', 'Birds', 'Reptiles'],
        'conservation_score': ['high', 'medium', 'low'],
        'gene': ['GENE1', 'GENE2', 'GENE3']
    })
    try:
        group_by_taxon(df_non_numeric)
        print("❌ Should fail on non-numeric conservation_score")
    except Exception as e:
        print(f"✅ Expected error: {e}")
    # Mixed types
    df_mixed = pd.DataFrame({
        'taxon_group': ['Mammals', 'Birds', 'Reptiles'],
        'conservation_score': [5.0, 'medium', 7.0],
        'gene': ['GENE1', 'GENE2', 'GENE3']
    })
    try:
        group_by_taxon(df_mixed)
        print("❌ Should fail on mixed types")
    except Exception as e:
        print(f"✅ Expected error: {e}")
    # Duplicate group names
    df_dup = pd.DataFrame({
        'taxon_group': ['Mammals', 'Mammals', 'Birds'],
        'conservation_score': [5.0, 6.0, 7.0],
        'gene': ['GENE1', 'GENE2', 'GENE3']
    })
    grouped = group_by_taxon(df_dup)
    print(f"✅ Duplicate group names handled: {list(grouped.keys())}")
    # Extreme values
    df_extreme = pd.DataFrame({
        'taxon_group': ['Mammals', 'Birds', 'Reptiles'],
        'conservation_score': [1e9, -1e9, 0],
        'gene': ['GENE1', 'GENE2', 'GENE3']
    })
    grouped = group_by_taxon(df_extreme)
    stats = calculate_stats(grouped)
    print(f"✅ Extreme values: {stats[['taxon_group','mean_score','std_dev']].to_dict('records')}")


def test_encoding():
    print("\n📋 Test: File encoding issues")
    # Simulate reading with different encodings
    try:
        df_utf8 = pd.read_csv("data/example_conservation_scores.csv", encoding="utf-8")
        print("✅ UTF-8 encoding read successfully")
    except Exception as e:
        print(f"❌ UTF-8 encoding failed: {e}")
    try:
        df_latin1 = pd.read_csv("data/example_conservation_scores.csv", encoding="latin-1")
        print("✅ Latin-1 encoding read successfully")
    except Exception as e:
        print(f"❌ Latin-1 encoding failed: {e}")


def test_irrelevant_columns():
    print("\n📋 Test: Irrelevant columns")
    df = pd.DataFrame({
        'taxon_group': ['Mammals', 'Birds', 'Reptiles'],
        'conservation_score': [5.0, 6.0, 7.0],
        'gene': ['GENE1', 'GENE2', 'GENE3'],
        'extra_col1': ['foo', 'bar', 'baz'],
        'extra_col2': [123, 456, 789]
    })
    grouped = group_by_taxon(df)
    stats = calculate_stats(grouped)
    print(f"✅ Irrelevant columns ignored: {stats[['taxon_group','mean_score']].to_dict('records')}")


def run_all():
    test_large_dataset()
    test_edge_cases()
    test_encoding()
    test_irrelevant_columns()
    print("\n🎉 Advanced TaxoConserv tests complete!")

if __name__ == "__main__":
    run_all()
