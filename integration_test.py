#!/usr/bin/env python3
"""
Integration test for input_parser and taxon_grouping modules
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from input_parser import parse_input, validate_input
from taxon_grouping import group_by_taxon, calculate_stats, get_group_summary

def integration_test():
    """Test the complete pipeline from input parsing to statistical analysis."""
    
    print("🔬 TaxoConserv Integration Test")
    print("=" * 50)
    
    # Step 1: Parse input
    print("\n📋 Step 1: Parse input data")
    df = parse_input("data/example_conservation_scores.csv")
    print(f"✅ Raw data loaded: {df.shape}")
    
    # Step 2: Validate input
    print("\n📋 Step 2: Validate input data")
    validated_df = validate_input(df)
    print(f"✅ Data validated: {validated_df.shape}")
    
    # Step 3: Group by taxon
    print("\n📋 Step 3: Group by taxonomic groups")
    grouped_data = group_by_taxon(validated_df)
    print(f"✅ Data grouped: {len(grouped_data)} groups")
    
    # Step 4: Calculate statistics
    print("\n📋 Step 4: Calculate descriptive statistics")
    stats_df = calculate_stats(grouped_data)
    print(f"✅ Statistics calculated: {stats_df.shape}")
    
    # Step 5: Display results
    print("\n📊 Final Results:")
    print("=" * 50)
    print(stats_df.round(3))
    
    # Step 6: Quick summary
    print("\n📋 Quick Summary:")
    get_group_summary(grouped_data)
    
    # Step 7: Key insights
    print("\n🔍 Key Insights:")
    print("=" * 50)
    
    # Find group with highest mean score
    highest_mean = stats_df.loc[stats_df['mean_score'].idxmax()]
    print(f"📈 Highest conservation: {highest_mean['taxon_group']} "
          f"(mean={highest_mean['mean_score']:.3f})")
    
    # Find group with lowest mean score
    lowest_mean = stats_df.loc[stats_df['mean_score'].idxmin()]
    print(f"📉 Lowest conservation: {lowest_mean['taxon_group']} "
          f"(mean={lowest_mean['mean_score']:.3f})")
    
    # Find group with highest variability
    highest_std = stats_df.loc[stats_df['std_dev'].idxmax()]
    print(f"📊 Most variable: {highest_std['taxon_group']} "
          f"(std={highest_std['std_dev']:.3f})")
    
    # Calculate overall statistics
    total_samples = stats_df['sample_size'].sum()
    print(f"📋 Total samples: {total_samples}")
    print(f"📋 Total groups: {len(stats_df)}")
    
    print("\n🎉 Integration test completed successfully!")

if __name__ == "__main__":
    integration_test()
