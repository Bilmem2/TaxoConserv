def create_demo_data():
    """Create sample conservation data for demonstration."""
    import numpy as np
    import pandas as pd
    np.random.seed(42)  # For reproducible results
    # Main taxon groups
    mammals_data = np.random.beta(2, 2, 50) * 0.8 + 0.1
    birds_data = np.random.beta(3, 1.5, 30) * 0.9 + 0.1
    reptiles_data = np.random.beta(1.5, 3, 30) * 0.7 + 0.1
    unknown_data = np.random.beta(2, 2, 10) * 0.5 + 0.2
    misc_data = np.random.beta(2, 2, 10) * 0.5 + 0.2
    primate_data = np.random.beta(2, 2, 15) * 0.8 + 0.1
    hominid_data = np.random.beta(2, 2, 10) * 0.8 + 0.1
    n = 50 + 30 + 30 + 10 + 10 + 15 + 10
    family = (
        ['Mammalia'] * 50 + ['Aves'] * 30 + ['Reptilia'] * 30 + ['Unknown'] * 10 + ['Misc'] * 10 + ['Mammalia'] * 15 + ['Mammalia'] * 10
    )
    genus = (
        ['Canis'] * 25 + ['Felis'] * 25 + ['Corvus'] * 15 + ['Passer'] * 15 + ['Lacerta'] * 15 + ['Python'] * 15 + ['Unknown'] * 10 + ['Misc'] * 10 + ['Homo'] * 15 + ['Pan'] * 10
    )[:n]
    species = ([f'Species_{i}' for i in range(n)])
    feature_1 = np.concatenate([
        np.random.normal(1.2, 0.2, 50),
        np.random.normal(0.9, 0.15, 30),
        np.random.normal(1.5, 0.25, 30),
        np.random.normal(1.0, 0.3, 10),
        np.random.normal(1.1, 0.3, 10),
        np.random.normal(1.3, 0.2, 15),
        np.random.normal(1.4, 0.2, 10)
    ])
    feature_2 = np.concatenate([
        np.random.normal(0.8, 0.1, 50),
        np.random.normal(1.1, 0.1, 30),
        np.random.normal(0.7, 0.1, 30),
        np.random.normal(0.9, 0.2, 10),
        np.random.normal(1.0, 0.2, 10),
        np.random.normal(0.85, 0.1, 15),
        np.random.normal(0.95, 0.1, 10)
    ])
    feature_3 = np.concatenate([
        np.random.normal(3.1, 0.3, 50),
        np.random.normal(2.8, 0.25, 30),
        np.random.normal(3.5, 0.35, 30),
        np.random.normal(3.0, 0.4, 10),
        np.random.normal(2.9, 0.4, 10),
        np.random.normal(3.2, 0.3, 15),
        np.random.normal(3.3, 0.3, 10)
    ])
    feature_1[3] = 2.2
    feature_1[30] = 2.8
    feature_2[3] = 0.5
    feature_2[30] = 0.4
    feature_3[3] = 4.2
    feature_3[30] = 4.8
    demo_data = pd.DataFrame({
        'taxon_group': ['Mammals'] * 50 + ['Birds'] * 30 + ['Reptiles'] * 30 + ['unknown'] * 10 + ['misc'] * 10 + ['primate'] * 15 + ['hominid'] * 10,
        'conservation_score': np.concatenate([mammals_data, birds_data, reptiles_data, unknown_data, misc_data, primate_data, hominid_data]),
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'gene_name': [f'Gene_{i}' for i in range(n)],
        'chromosome': np.random.choice(['chr1', 'chr2', 'chr3', 'chr4', 'chr5'], n),
        'position': np.random.randint(1000, 999999, n),
        'family': family,
        'genus': genus,
        'species': species
    })
    return demo_data
#!/usr/bin/env python3
"""
TaxoConserv - Input Parser Module

This module handles parsing and validation of input CSV/TSV files
containing conservation scores and taxonomic information.

Functions:
    parse_input(file_path: str) -> pd.DataFrame
    validate_input(df: pd.DataFrame, required_columns: list) -> pd.DataFrame
"""

import pandas as pd
import os
from pathlib import Path
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required columns for TaxoConserv analysis
REQUIRED_COLUMNS = ["gene", "position", "species", "taxon_group", "conservation_score"]

# Alternative column names mapping (for flexibility)
COLUMN_MAPPING = {
    # Standard phyloP column names
    "phyloP_score": "conservation_score",
    "phylop_score": "conservation_score",
    "phyloP": "conservation_score",
    "phylop": "conservation_score",
    
    # Alternative position names
    "pos": "position",
    "genomic_position": "position",
    "chromosome_position": "position",
    
    # Alternative taxon group names
    "taxon": "taxon_group",
    "taxonomy": "taxon_group",
    "taxonomic_group": "taxon_group",
    
    # Alternative gene names
    "gene_name": "gene",
    "gene_id": "gene",
    "gene_symbol": "gene",
    
    # Alternative species names
    "organism": "species",
    "species_name": "species"
}

def get_demo_data() -> pd.DataFrame:
    """
    Return a small demo DataFrame for testing and UI demo purposes.
    """
    import numpy as np
    np.random.seed(42)
    demo = pd.DataFrame({
        'taxon_group': ['Mammals']*5 + ['Birds']*5 + ['Reptiles']*3,
        'conservation_score': np.concatenate([
            np.random.normal(0.6, 0.15, 5),
            np.random.normal(0.7, 0.12, 5),
            np.random.normal(0.4, 0.18, 3)
        ]),
        'gene': [f'Gene_{i}' for i in range(13)],
        'species': np.random.choice(['Homo sapiens', 'Gallus gallus', 'Python regius'], 13),
        'position': np.random.randint(1000, 999999, 13)
    })
    return demo


def parse_input(file_path: str) -> pd.DataFrame:
    """
    Parse input CSV or TSV file and return pandas DataFrame.
    
    Args:
        file_path (str): Path to the input file (CSV or TSV)
        
    Returns:
        pd.DataFrame: Parsed data from the input file
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is not supported
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"❌ Input file not found: {file_path}")
        logger.info("ℹ️ Loading demo data instead.")
        return get_demo_data()
    # Get file extension
    file_path_obj = Path(file_path)
    file_extension = file_path_obj.suffix.lower()
    logger.info(f"📁 Parsing input file: {file_path}")
    
    try:
        # Determine separator based on file extension or auto-detect
        if file_extension in ['.csv', '.tsv', '.txt']:
            # Try to auto-detect separator for .txt
            if file_extension == '.csv':
                separator = ','
            elif file_extension == '.tsv':
                separator = '\t'
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if '\t' in first_line:
                        separator = '\t'
                    elif ',' in first_line:
                        separator = ','
                    else:
                        separator = '\t'
                logger.info(f"📋 Auto-detected separator for .txt file: '{separator}'")
        else:
            logger.error(f"❌ Unsupported file format: {file_extension}. Supported: .csv, .tsv, .txt")
            logger.info("ℹ️ Loading demo data instead.")
            return get_demo_data()
        # Read the file
        df = pd.read_csv(file_path, sep=separator, encoding='utf-8')
        # Check if DataFrame is empty
        if df.empty:
            logger.error(f"❌ The input file is empty: {file_path}")
            logger.info("ℹ️ Loading demo data instead.")
            return get_demo_data()
        # Apply column mapping if needed
        df = _apply_column_mapping(df)
        logger.info(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"📊 Column names: {list(df.columns)}")
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"❌ The input file is empty: {file_path}")
        logger.info("ℹ️ Loading demo data instead.")
        return get_demo_data()
    except pd.errors.ParserError as e:
        logger.error(f"❌ Error parsing file {file_path}: {str(e)}")
        logger.info("ℹ️ Loading demo data instead.")
        return get_demo_data()
    except Exception as e:
        logger.error(f"❌ Unexpected error reading file {file_path}: {str(e)}")
        logger.info("ℹ️ Loading demo data instead.")
        return get_demo_data()


def validate_input(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Validate input DataFrame and ensure required columns are present.
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate
        required_columns (List[str], optional): List of required column names.
                                              Defaults to REQUIRED_COLUMNS.
        
    Returns:
        pd.DataFrame: Validated and cleaned DataFrame
        
    Raises:
        ValueError: If required columns are missing or data validation fails
    """
    
    if required_columns is None:
        # For demo mode, only require the basic columns we have
        if 'taxon_group' in df.columns and 'conservation_score' in df.columns:
            required_columns = ['taxon_group', 'conservation_score']
        else:
            required_columns = REQUIRED_COLUMNS
    
    logger.info("🔍 Validating input data...")
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        available_columns = list(df.columns)
        logger.error(f"❌ Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {available_columns}")
        logger.info(f"Required columns: {required_columns}")
        logger.info("ℹ️ Loading demo data instead.")
        return get_demo_data()
    
    # Create a copy to avoid modifying the original DataFrame
    validated_df = df.copy()
    
    # Validate and clean data types
    try:
        # Convert position to numeric (int64)
        if 'position' in validated_df.columns:
            validated_df['position'] = pd.to_numeric(validated_df['position'], errors='coerce')
            pos_na_count = validated_df['position'].isna().sum()
            if pos_na_count > 0:
                logger.warning(f"⚠️ {pos_na_count} invalid position values converted to NaN")
        
        # Convert conservation_score to numeric (float64)
        if 'conservation_score' in validated_df.columns:
            validated_df['conservation_score'] = pd.to_numeric(validated_df['conservation_score'], errors='coerce')
            score_na_count = validated_df['conservation_score'].isna().sum()
            if score_na_count > 0:
                logger.warning(f"⚠️ {score_na_count} invalid conservation score values converted to NaN")
        
        # Convert taxon_group to categorical
        if 'taxon_group' in validated_df.columns:
            validated_df['taxon_group'] = validated_df['taxon_group'].astype('category')
            unique_taxa = validated_df['taxon_group'].cat.categories.tolist()
            logger.info(f"📋 Taxonomic groups found: {unique_taxa}")
        
        # Convert gene and species to string (object)
        for col in ['gene', 'species']:
            if col in validated_df.columns:
                validated_df[col] = validated_df[col].astype('string')
        
        # Check for completely empty rows
        empty_rows = validated_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            validated_df = validated_df.dropna(how='all')
            logger.warning(f"⚠️ Removed {empty_rows} completely empty rows")
        
        # Check for critical missing data
        critical_na_count = validated_df[['taxon_group', 'conservation_score']].isna().any(axis=1).sum()
        if critical_na_count > 0:
            logger.warning(f"⚠️ {critical_na_count} rows have missing taxon_group or conservation_score data")
        
        # Final data summary
        final_row_count = len(validated_df)
        logger.info(f"✅ Validation complete: {final_row_count} rows ready for analysis")
        
        # Display basic statistics
        if 'conservation_score' in validated_df.columns:
            score_stats = validated_df['conservation_score'].describe()
            logger.info(f"📊 Conservation score statistics:")
            logger.info(f"   Mean: {score_stats['mean']:.3f}")
            logger.info(f"   Min: {score_stats['min']:.3f}")
            logger.info(f"   Max: {score_stats['max']:.3f}")
            logger.info(f"   Valid scores: {score_stats['count']}")
        
        return validated_df
        
    except Exception as e:
        raise ValueError(f"❌ Data validation failed: {str(e)}")


def _apply_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply column name mapping to standardize column names.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    
    # Create a copy to avoid modifying the original DataFrame
    mapped_df = df.copy()
    
    # Apply column mapping
    columns_renamed = {}
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in mapped_df.columns:
            mapped_df = mapped_df.rename(columns={old_name: new_name})
            columns_renamed[old_name] = new_name
    
    if columns_renamed:
        logger.info(f"📝 Column mapping applied: {columns_renamed}")
        print(f"📝 Column mapping applied: {columns_renamed}")
    
    return mapped_df


# Test function for development
def test_parser():
    """Test function for development purposes."""
    try:
        # Test with the example file
        test_file = "data/example_conservation_scores.csv"
        if os.path.exists(test_file):
            df = parse_input(test_file)
            validated_df = validate_input(df)
            print("✅ Test passed!")
            print(f"Shape: {validated_df.shape}")
            print(f"Columns: {list(validated_df.columns)}")
            print(f"Data types:\n{validated_df.dtypes}")
        else:
            print(f"❌ Test file not found: {test_file}")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_parser()
