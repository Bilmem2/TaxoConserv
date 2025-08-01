#!/usr/bin/env python3
"""
TaxoConserv - Taxon Grouping Module

This module handles taxonomic grouping and descriptive statistics calculation
for conservation scores across different taxonomic levels.

Functions:
    group_by_taxon(df: pd.DataFrame) -> dict
    calculate_stats(grouped_data: dict, score_column: str = "conservation_score") -> pd.DataFrame
"""

import pandas as pd
import numpy as np
from typing import Dict, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def group_by_taxon(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Group DataFrame by taxonomic groups with normalization, mapping, and flexible column selection.
    Args:
        df (pd.DataFrame): Input DataFrame
        group_column (str): Column to use for grouping (default 'taxon_group')
        normalize (bool): Normalize group names (case, whitespace)
        custom_map (dict): Optional mapping of group names (e.g. {'primate': 'Mammalia'})
        hierarchy (list): Optional list of columns for hierarchical grouping (e.g. ['family', 'genus', 'species'])
    Returns:
        dict: Dictionary with group names as keys and DataFrames as values
    """
    logger.info("🔍 Grouping data by taxonomic groups...")
    # Flexible arguments
    group_column = 'taxon_group'
    normalize = True
    custom_map = None
    hierarchy = None
    # If user passes config via df.attrs, use it
    if hasattr(df, 'attrs'):
        group_column = df.attrs.get('group_column', group_column)
        normalize = df.attrs.get('normalize', normalize)
        custom_map = df.attrs.get('custom_map', custom_map)
        hierarchy = df.attrs.get('hierarchy', hierarchy)
    # Validate input DataFrame
    if df.empty:
        raise ValueError("❌ Input DataFrame is empty")
    if hierarchy:
        # Hierarchical grouping: use first available column
        for col in hierarchy:
            if col in df.columns:
                group_column = col
                break
    if group_column not in df.columns:
        raise ValueError(f"❌ '{group_column}' column not found in DataFrame")
    # Remove rows with missing group values
    df_clean = df.dropna(subset=[group_column])
    if len(df_clean) == 0:
        raise ValueError("❌ No valid taxonomic groups found in data")
    # Normalize group names
    if normalize:
        df_clean[group_column] = df_clean[group_column].astype(str).str.strip().str.lower()
    # Apply custom mapping
    if custom_map:
        df_clean[group_column] = df_clean[group_column].map(lambda x: custom_map.get(x, x))
    # Count dropped rows
    dropped_rows = len(df) - len(df_clean)
    if dropped_rows > 0:
        logger.warning(f"⚠️ Dropped {dropped_rows} rows with missing {group_column} values")
    # Group by group_column
    grouped_data = {}
    taxon_groups = df_clean[group_column].unique()
    taxon_groups = sorted(taxon_groups)
    for taxon in taxon_groups:
        taxon_df = df_clean[df_clean[group_column] == taxon].copy()
        grouped_data[taxon] = taxon_df
        logger.info(f"📊 {taxon}: {len(taxon_df)} records")
    logger.info(f"✅ {len(taxon_groups)} taxon groups detected: {taxon_groups}")
    # Warn on small/unknown groups
    for taxon in taxon_groups:
        if len(grouped_data[taxon]) < 3:
            logger.warning(f"⚠️ Group '{taxon}' has very few samples ({len(grouped_data[taxon])})")
        if taxon in ['unknown', 'other', 'misc']:
            logger.warning(f"⚠️ Ambiguous group name detected: '{taxon}'")
    return grouped_data


def calculate_stats(grouped_data: Dict[str, pd.DataFrame], 
                   score_column: str = "conservation_score") -> pd.DataFrame:
    """
    Calculate descriptive statistics for each taxonomic group.
    
    Args:
        grouped_data (dict): Dictionary of DataFrames grouped by taxon
        score_column (str): Column name containing conservation scores
        
    Returns:
        pd.DataFrame: Summary statistics with columns:
                     - taxon_group: Taxonomic group name
                     - sample_size: Number of records
                     - mean_score: Mean conservation score
                     - median_score: Median conservation score
                     - std_dev: Standard deviation
                     - min_score: Minimum score
                     - max_score: Maximum score
                     - q25: 25th percentile
                     - q75: 75th percentile
        
    Raises:
        ValueError: If score_column is missing from any group
        KeyError: If grouped_data is empty or invalid
    """
    
    logger.info("📊 Calculating descriptive statistics for each taxonomic group...")
    
    if not grouped_data:
        raise ValueError("❌ No grouped data provided")
    
    # Check if score column exists in at least one group
    score_column_exists = False
    for taxon, df in grouped_data.items():
        if score_column in df.columns:
            score_column_exists = True
            break
    
    if not score_column_exists:
        raise ValueError(f"❌ Score column '{score_column}' not found in any group")
    
    # Calculate statistics for each group
    stats_list = []
    # Process groups in alphabetical order
    for taxon in sorted(grouped_data.keys()):
        df = grouped_data[taxon]
        # Initialize stats dictionary
        stats = {
            'taxon_group': taxon,
            'sample_size': len(df),
            'mean_score': np.nan,
            'median_score': np.nan,
            'std_dev': np.nan,
            'min_score': np.nan,
            'max_score': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'iqr': np.nan,
            'outlier_count': 0
        }
        # Check if score column exists in this group
        if score_column not in df.columns:
            logger.warning(f"⚠️ Score column '{score_column}' not found in {taxon} group")
            stats_list.append(stats)
            continue
        # Get valid (non-NaN) scores
        valid_scores = df[score_column].dropna()
        if len(valid_scores) == 0:
            logger.warning(f"⚠️ No valid scores found for {taxon} group")
            stats_list.append(stats)
            continue
        # Calculate statistics
        try:
            q25 = float(valid_scores.quantile(0.25))
            q75 = float(valid_scores.quantile(0.75))
            iqr = q75 - q25
            # Outlier detection: 1.5*IQR rule
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_count = ((valid_scores < lower_bound) | (valid_scores > upper_bound)).sum()
            stats.update({
                'mean_score': float(valid_scores.mean()),
                'median_score': float(valid_scores.median()),
                'std_dev': float(valid_scores.std()),
                'min_score': float(valid_scores.min()),
                'max_score': float(valid_scores.max()),
                'q25': q25,
                'q75': q75,
                'iqr': iqr,
                'outlier_count': int(outlier_count)
            })
            logger.info(f"📈 {taxon}: n={len(valid_scores)}, "
                        f"mean={stats['mean_score']:.3f}, "
                        f"median={stats['median_score']:.3f}, "
                        f"std={stats['std_dev']:.3f}, "
                        f"IQR={iqr:.3f}, outliers={outlier_count}")
        except Exception as e:
            logger.error(f"❌ Error calculating statistics for {taxon}: {str(e)}")
            # Keep NaN values for this group
        stats_list.append(stats)
    # Create summary DataFrame
    summary_df = pd.DataFrame(stats_list)
    # Ensure proper data types
    if not summary_df.empty:
        summary_df['taxon_group'] = summary_df['taxon_group'].astype('category')
        summary_df['sample_size'] = summary_df['sample_size'].astype('int64')
        summary_df = summary_df.sort_values('taxon_group').reset_index(drop=True)
        logger.info(f"✅ Statistics calculated for {len(summary_df)} taxonomic groups")
        return summary_df
    else:
        columns = ['taxon_group','sample_size','mean_score','median_score','std_dev','min_score','max_score','q25','q75','iqr','outlier_count']
        logger.info("✅ Statistics calculated for 0 taxonomic groups")
        return pd.DataFrame(columns=columns)
from typing import Optional

def filter_groups(grouped_data: Dict[str, pd.DataFrame], include: Optional[list] = None, exclude: Optional[list] = None) -> Dict[str, pd.DataFrame]:
    """
    Filter grouped data by including or excluding specific taxon groups.
    Args:
        grouped_data (dict): Dictionary of DataFrames grouped by taxon
        include (list): List of taxon group names to include
        exclude (list): List of taxon group names to exclude
    Returns:
        dict: Filtered grouped data
    """
    result = {}
    for taxon, df in grouped_data.items():
        if include and taxon not in include:
            continue
        if exclude and taxon in exclude:
            continue
        result[taxon] = df
    return result
    
    # Sort by taxon group name
    summary_df = summary_df.sort_values('taxon_group').reset_index(drop=True)
    
    logger.info(f"✅ Statistics calculated for {len(summary_df)} taxonomic groups")
    
    return summary_df


def get_group_summary(grouped_data: Dict[str, pd.DataFrame], 
                     score_column: str = "conservation_score") -> None:
    """
    Print a quick summary of grouped data for debugging purposes.
    
    Args:
        grouped_data (dict): Dictionary of DataFrames grouped by taxon
        score_column (str): Column name containing conservation scores
    """
    
    print("\n🔍 Taxonomic Group Summary:")
    print("=" * 60)
    
    total_records = 0
    
    for taxon in sorted(grouped_data.keys()):
        df = grouped_data[taxon]
        total_records += len(df)
        if score_column in df.columns:
            valid_scores = df[score_column].dropna()
            if len(valid_scores) > 0:
                q25 = valid_scores.quantile(0.25)
                q75 = valid_scores.quantile(0.75)
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outlier_count = ((valid_scores < lower_bound) | (valid_scores > upper_bound)).sum()
                print(f"📊 {taxon.upper():<12} | "
                      f"n={len(valid_scores):<4} | "
                      f"mean={valid_scores.mean():.3f} | "
                      f"range=({valid_scores.min():.3f}-{valid_scores.max():.3f}) | "
                      f"IQR={iqr:.3f} | outliers={outlier_count}")
            else:
                print(f"📊 {taxon.upper():<12} | "
                      f"n={len(df):<4} | "
                      f"No valid scores")
        else:
            print(f"📊 {taxon.upper():<12} | "
                  f"n={len(df):<4} | "
                  f"Score column missing")
    
    print("-" * 60)
    print(f"📈 Total records: {total_records}")
    print(f"📈 Total groups: {len(grouped_data)}")


# Test function for development
def test_taxon_grouping():
    """Test function for development purposes."""
    
    try:
        # Test with example data
        import sys
        from pathlib import Path
        
        # Add src directory to path
        src_path = Path(__file__).parent
        sys.path.insert(0, str(src_path))
        
        from input_parser import parse_input, validate_input
        
        # Load test data
        test_file = "../data/example_conservation_scores.csv"
        if not Path(test_file).exists():
            test_file = "data/example_conservation_scores.csv"
        
        df = parse_input(test_file)
        validated_df = validate_input(df)
        
        # Test grouping
        grouped_data = group_by_taxon(validated_df)
        
        # Test statistics calculation
        stats_df = calculate_stats(grouped_data)
        
        # Display results
        print("\n✅ Test Results:")
        print("=" * 50)
        print(f"Original data shape: {validated_df.shape}")
        print(f"Number of groups: {len(grouped_data)}")
        print(f"Statistics shape: {stats_df.shape}")
        
        print("\n📊 Summary Statistics:")
        print(stats_df.round(3))
        
        # Quick summary
        get_group_summary(grouped_data)
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_taxon_grouping()
