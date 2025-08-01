#!/usr/bin/env python3
"""
TaxoConserv - Statistical Tests Module

This module handles statistical testing for conservation score differences
between taxonomic groups using non-parametric tests.

Functions:
    run_kruskal_wallis(grouped_data: dict, score_column: str = "conservation_score") -> dict
    format_test_results(test_results: dict, group_names: list) -> str
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Union, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_kruskal_wallis(grouped_data: Dict[str, pd.DataFrame], 
                      score_column: str = "conservation_score") -> Dict[str, Union[float, int]]:
    """
    Perform Kruskal-Wallis test for conservation score differences between taxonomic groups.
    
    The Kruskal-Wallis test is a non-parametric test that determines whether the 
    medians of two or more groups are different. It's suitable for conservation 
    score data which may not follow normal distribution.
    
    Args:
        grouped_data (dict): Dictionary with taxon group names as keys and 
                           DataFrames as values (output from group_by_taxon)
        score_column (str): Column name containing conservation scores
        
    Returns:
        dict: Dictionary containing test results:
              - H_statistic: Kruskal-Wallis H statistic
              - p_value: p-value of the test
              - degrees_of_freedom: Degrees of freedom (k-1)
              - sample_sizes: Sample sizes for each group
              - group_names: Names of the groups tested
              - significant: Boolean indicating if p < 0.05
              - test_type: "Kruskal-Wallis"
        
    Raises:
        ValueError: If less than 2 groups provided or score column missing
        KeyError: If grouped_data is empty or invalid
    """
    
    logger.info("🧪 Running Kruskal-Wallis test for taxonomic group differences...")
    
    # Validate input
    if not grouped_data:
        raise ValueError("❌ No grouped data provided")
    
    if len(grouped_data) < 2:
        raise ValueError("❌ Kruskal-Wallis test requires at least 2 groups. "
                        f"Found {len(grouped_data)} groups.")
    
    # Extract score arrays for each group
    score_arrays = []
    group_names = []
    sample_sizes = []
    
    for group_name in sorted(grouped_data.keys()):
        df = grouped_data[group_name]
        
        # Check if score column exists
        if score_column not in df.columns:
            raise ValueError(f"❌ Score column '{score_column}' not found in {group_name} group")
        
        # Get valid (non-NaN) scores
        valid_scores = df[score_column].dropna()
        
        if len(valid_scores) == 0:
            logger.warning(f"⚠️ No valid scores found for {group_name} group")
            continue
        
        score_arrays.append(valid_scores.values)
        group_names.append(group_name)
        sample_sizes.append(len(valid_scores))
        
        logger.info(f"📊 {group_name}: n={len(valid_scores)}, "
                   f"median={valid_scores.median():.3f}, "
                   f"range=({valid_scores.min():.3f}-{valid_scores.max():.3f})")
    
    # Check if we have enough valid groups
    if len(score_arrays) < 2:
        raise ValueError(f"❌ Insufficient valid groups for testing. "
                        f"Found {len(score_arrays)} groups with valid data.")
    
    # Perform Kruskal-Wallis test
    try:
        h_statistic, p_value = stats.kruskal(*score_arrays)
        
        # Calculate degrees of freedom
        degrees_of_freedom = len(score_arrays) - 1
        
        # Determine significance
        alpha = 0.05
        significant = p_value < alpha
        
        # Create results dictionary
        results = {
            "H_statistic": round(float(h_statistic), 2),
            "p_value": round(float(p_value), 4),
            "degrees_of_freedom": degrees_of_freedom,
            "sample_sizes": sample_sizes,
            "group_names": group_names,
            "significant": significant,
            "test_type": "Kruskal-Wallis",
            "alpha": alpha
        }
        
        # Log results
        logger.info(f"📈 Kruskal-Wallis Test Results:")
        logger.info(f"   H-statistic: {results['H_statistic']}")
        logger.info(f"   p-value: {results['p_value']}")
        logger.info(f"   Degrees of freedom: {results['degrees_of_freedom']}")
        logger.info(f"   Sample sizes: {dict(zip(group_names, sample_sizes))}")
        
        if significant:
            logger.info("✅ Significant difference detected between taxon groups (p < 0.05).")
        else:
            logger.info("❌ No significant difference detected between taxon groups (p >= 0.05).")
        
        return results
        
    except Exception as e:
        raise ValueError(f"❌ Error performing Kruskal-Wallis test: {str(e)}")


def format_test_results(test_results: Dict[str, Union[float, int]], 
                       group_names: Optional[List[str]] = None) -> str:
    """
    Format Kruskal-Wallis test results into a human-readable string.
    
    Args:
        test_results (dict): Results from run_kruskal_wallis function
        group_names (List[str], optional): Names of the groups tested
        
    Returns:
        str: Formatted string report of the test results
    """
    
    if group_names is None:
        gn = test_results.get('group_names', [])
        group_names = gn if isinstance(gn, list) and all(isinstance(g, str) for g in gn) else []
    # Ensure group_names is a list of strings
    if not isinstance(group_names, list):
        group_names = []
    else:
        group_names = [str(g) for g in group_names]

    report = []
    test_type = test_results.get('test_type', 'Statistical Test')
    report.append(f"🧪 {test_type} Results")
    report.append("=" * 50)
    # Test information
    report.append(f"Test Type: {test_type}")
    report.append(f"Groups Tested: {', '.join(group_names)}")
    report.append(f"Number of Groups: {len(group_names)}")
    # Sample sizes
    sample_sizes = test_results.get('sample_sizes', [])
    if isinstance(sample_sizes, list) and len(sample_sizes) == len(group_names):
        report.append("\n📊 Sample Sizes:")
        for name, size in zip(group_names, sample_sizes):
            report.append(f"   {name}: {size}")
    # Statistical results
    report.append("\n📈 Statistical Results:")
    if test_type == 'Kruskal-Wallis':
        report.append(f"   H-statistic: {test_results.get('H_statistic', 'N/A')}")
        report.append(f"   p-value: {test_results.get('p_value', 'N/A')}")
        report.append(f"   Degrees of freedom: {test_results.get('degrees_of_freedom', 'N/A')}")
    elif test_type == 'Independent t-test':
        report.append(f"   T-statistic: {test_results.get('T_statistic', 'N/A')}")
        report.append(f"   p-value: {test_results.get('p_value', 'N/A')}")
    elif test_type == 'Mann-Whitney U':
        report.append(f"   U-statistic: {test_results.get('U_statistic', 'N/A')}")
        report.append(f"   p-value: {test_results.get('p_value', 'N/A')}")
    # Significance interpretation
    alpha = test_results.get('alpha', 0.05)
    significant = test_results.get('significant', False)
    report.append(f"\n🎯 Interpretation (α = {alpha}):")
    if significant:
        report.append(f"   ✅ SIGNIFICANT: p < {alpha}")
        report.append("   There is a statistically significant difference")
        report.append("   between the conservation scores of taxonomic groups.")
    else:
        report.append(f"   ❌ NOT SIGNIFICANT: p >= {alpha}")
        report.append("   No statistically significant difference detected")
        report.append("   between the conservation scores of taxonomic groups.")
    # Recommendations
    report.append("\n💡 Recommendations:")
    if significant:
        report.append("   • Consider post-hoc pairwise comparisons (Dunn's test)")
        report.append("   • Examine boxplots to identify which groups differ")
        report.append("   • Check biological significance of the differences")
    else:
        report.append("   • Groups show similar conservation patterns")
        report.append("   • Consider increasing sample size if needed")
        report.append("   • Check for potential confounding factors")
    return "\n".join(report)
def get_test_summary(test_results: Dict[str, Union[float, int]]) -> Dict[str, str]:
    """
    Get a concise summary of test results for reporting.
    
    Args:
        test_results (dict): Results from run_kruskal_wallis function
        
    Returns:
        dict: Summary information for reports
    """
    
    significant = test_results.get('significant', False)
    p_value = test_results.get('p_value', 0.0)
    h_statistic = test_results.get('H_statistic', 0.0)
    
    return {
        "test_name": "Kruskal-Wallis",
        "significance": "Yes" if significant else "No",
        "p_value": f"{p_value:.4f}",
        "h_statistic": f"{h_statistic:.2f}",
        "interpretation": "Significant differences detected" if significant else "No significant differences"
    }


# Test function for development
def test_kruskal_wallis():
    """Test function for development purposes."""
    
    try:
        # Test with example data
        import sys
        from pathlib import Path
        
        # Add src directory to path
        src_path = Path(__file__).parent
        sys.path.insert(0, str(src_path))
        
        from input_parser import parse_input, validate_input
        from taxon_grouping import group_by_taxon, calculate_stats
        
        # Load and prepare test data
        test_file = "../data/example_conservation_scores.csv"
        if not Path(test_file).exists():
            test_file = "data/example_conservation_scores.csv"
        
        df = parse_input(test_file)
        validated_df = validate_input(df)
        grouped_data = group_by_taxon(validated_df)
        
        # Test Kruskal-Wallis
        test_results = run_kruskal_wallis(grouped_data)
        
        # Format results
        formatted_report = format_test_results(test_results)
        test_summary = get_test_summary(test_results)
        
        # Display results
        print("\n✅ Kruskal-Wallis Test Results:")
        print("=" * 50)
        print(f"Groups tested: {len(grouped_data)}")
        print(f"H-statistic: {test_results['H_statistic']}")
        print(f"p-value: {test_results['p_value']}")
        print(f"Significant: {test_results['significant']}")
        
        print("\n📋 Formatted Report:")
        print(formatted_report)
        
        print("\n📊 Test Summary:")
        for key, value in test_summary.items():
            print(f"   {key}: {value}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_kruskal_wallis()
