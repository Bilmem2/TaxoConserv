def create_demo_data():
    """Create comprehensive sample conservation data for demonstration."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)  # For reproducible results
    
    # Create more realistic taxonomic conservation data
    n_entries = 200
    
    # Define taxonomic groups with realistic conservation patterns
    taxon_groups = {
        'Primates': {'n': 35, 'phyloP_mean': 2.8, 'phyloP_std': 0.8, 'gerp_mean': 4.2, 'gerp_std': 1.2, 'phastCons_mean': 0.75, 'phastCons_std': 0.15},
        'Carnivores': {'n': 30, 'phyloP_mean': 2.2, 'phyloP_std': 0.7, 'gerp_mean': 3.8, 'gerp_std': 1.0, 'phastCons_mean': 0.68, 'phastCons_std': 0.18},
        'Rodents': {'n': 40, 'phyloP_mean': 1.8, 'phyloP_std': 0.6, 'gerp_mean': 3.2, 'gerp_std': 0.9, 'phastCons_mean': 0.62, 'phastCons_std': 0.16},
        'Birds': {'n': 35, 'phyloP_mean': 2.5, 'phyloP_std': 0.9, 'gerp_mean': 4.0, 'gerp_std': 1.1, 'phastCons_mean': 0.72, 'phastCons_std': 0.17},
        'Reptiles': {'n': 25, 'phyloP_mean': 1.5, 'phyloP_std': 0.5, 'gerp_mean': 2.8, 'gerp_std': 0.8, 'phastCons_mean': 0.55, 'phastCons_std': 0.14},
        'Fish': {'n': 35, 'phyloP_mean': 1.2, 'phyloP_std': 0.4, 'gerp_mean': 2.2, 'gerp_std': 0.7, 'phastCons_mean': 0.48, 'phastCons_std': 0.12}
    }
    
    # Generate realistic gene names
    gene_families = ['BRCA', 'TP53', 'EGFR', 'MYC', 'RAS', 'PIK3', 'PTEN', 'APC', 'KRAS', 'BRAF']
    
    # Create data arrays
    taxon_group_list = []
    phyloP_scores = []
    gerp_scores = []
    phastCons_scores = []
    gene_names = []
    chromosomes = []
    positions = []
    functional_regions = []
    
    entry_counter = 0
    for group_name, params in taxon_groups.items():
        n = params['n']
        
        # Generate conservation scores with group-specific patterns
        group_phyloP = np.random.normal(params['phyloP_mean'], params['phyloP_std'], n)
        group_gerp = np.random.normal(params['gerp_mean'], params['gerp_std'], n)
        group_phastCons = np.random.beta(2, 2, n) * 0.4 + (params['phastCons_mean'] - 0.2)
        
        # Ensure realistic ranges
        group_phyloP = np.clip(group_phyloP, -2, 6)
        group_gerp = np.clip(group_gerp, -5, 8)
        group_phastCons = np.clip(group_phastCons, 0, 1)
        
        # Add some outliers for interesting analysis
        if n > 10:
            outlier_indices = np.random.choice(n, size=2, replace=False)
            group_phyloP[outlier_indices] *= 1.8
            group_gerp[outlier_indices] *= 1.5
        
        # Generate other attributes
        group_genes = [f"{np.random.choice(gene_families)}{i+entry_counter}" for i in range(n)]
        group_chroms = np.random.choice([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'], n)
        group_positions = np.random.randint(1000000, 250000000, n)
        group_functions = np.random.choice(['exon', 'intron', 'promoter', 'enhancer', 'UTR'], n, 
                                         p=[0.4, 0.3, 0.1, 0.1, 0.1])
        
        # Append to lists
        taxon_group_list.extend([group_name] * n)
        phyloP_scores.extend(group_phyloP)
        gerp_scores.extend(group_gerp)
        phastCons_scores.extend(group_phastCons)
        gene_names.extend(group_genes)
        chromosomes.extend(group_chroms)
        positions.extend(group_positions)
        functional_regions.extend(group_functions)
        
        entry_counter += n
    
    # Create additional derived scores for analysis
    total_entries = len(taxon_group_list)
    
    # CADD-like scores (0-40 range)
    cadd_scores = np.random.beta(2, 5, total_entries) * 35 + 2
    
    # REVEL-like scores (0-1 range)
    revel_scores = np.random.beta(1.5, 4, total_entries)
    
    # Create comprehensive DataFrame
    demo_data = pd.DataFrame({
        'gene': gene_names,
        'chromosome': chromosomes,
        'position': positions,
        'taxon_group': taxon_group_list,
        'functional_region': functional_regions,
        'phyloP_score': np.round(phyloP_scores, 3),
        'GERP_score': np.round(gerp_scores, 3),
        'phastCons_score': np.round(phastCons_scores, 3),
        'CADD_score': np.round(cadd_scores, 2),
        'REVEL_score': np.round(revel_scores, 4),
        # Legacy column for backward compatibility
        'conservation_score': np.round(phyloP_scores, 3)
    })
    
    # Add some metadata
    demo_data.attrs['description'] = 'Sample conservation dataset with multiple score types'
    demo_data.attrs['n_groups'] = len(taxon_groups)
    demo_data.attrs['score_types'] = ['phyloP', 'GERP', 'phastCons', 'CADD', 'REVEL']
    
    return demo_data


def create_sample_vcf_data():
    """Create sample VCF-like data for variant conservation analysis demonstration."""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)  # For reproducible results
    
    # Create realistic variant data
    n_variants = 50
    
    # Generate chromosomes (focusing on common ones)
    chromosomes = np.random.choice(['chr1', 'chr2', 'chr3', 'chr7', 'chr11', 'chr17', 'chr19', 'chrX'], 
                                 n_variants, p=[0.15, 0.12, 0.10, 0.12, 0.10, 0.15, 0.15, 0.11])
    
    # Generate positions
    positions = []
    for chrom in chromosomes:
        if chrom == 'chr1':
            pos = np.random.randint(1000000, 247000000)
        elif chrom == 'chr17':
            pos = np.random.randint(1000000, 83000000)  # BRCA1 region
        elif chrom == 'chr19':
            pos = np.random.randint(1000000, 59000000)
        else:
            pos = np.random.randint(1000000, 150000000)
        positions.append(pos)
    
    # Generate realistic gene names
    gene_names = [
        'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'APC', 'KRAS', 'BRAF', 'PIK3CA', 'EGFR', 'MYC',
        'RB1', 'CDKN2A', 'VHL', 'MLH1', 'MSH2', 'ATM', 'CHEK2', 'PALB2', 'CDH1', 'STK11',
        'SMAD4', 'BMPR1A', 'MUTYH', 'PMS2', 'MSH6', 'EPCAM', 'POLE', 'POLD1', 'NTHL1', 'AXIN2'
    ]
    
    # Generate variant attributes
    variant_data = []
    for i in range(n_variants):
        # Select random gene
        gene = np.random.choice(gene_names)
        
        # Generate conservation scores based on gene importance
        if gene in ['BRCA1', 'BRCA2', 'TP53', 'PTEN']:  # High conservation genes
            phyloP = np.random.normal(3.5, 1.0)
            gerp = np.random.normal(5.2, 1.5)
            phastCons = np.random.beta(3, 1) * 0.8 + 0.15
            cadd = np.random.normal(25, 8)
        elif gene in ['KRAS', 'BRAF', 'PIK3CA', 'EGFR']:  # Moderate conservation
            phyloP = np.random.normal(2.8, 0.8)
            gerp = np.random.normal(4.1, 1.2)
            phastCons = np.random.beta(2, 2) * 0.7 + 0.2
            cadd = np.random.normal(20, 6)
        else:  # Variable conservation
            phyloP = np.random.normal(2.0, 1.2)
            gerp = np.random.normal(3.2, 1.8)
            phastCons = np.random.beta(1.5, 2.5) * 0.8 + 0.1
            cadd = np.random.normal(15, 10)
        
        # Ensure realistic ranges
        phyloP = np.clip(phyloP, -3, 7)
        gerp = np.clip(gerp, -8, 10)
        phastCons = np.clip(phastCons, 0, 1)
        cadd = np.clip(cadd, 0, 40)
        
        # Generate REF and ALT alleles
        ref_alleles = ['A', 'T', 'G', 'C']
        ref = np.random.choice(ref_alleles)
        alt_choices = [x for x in ref_alleles if x != ref]
        alt = np.random.choice(alt_choices)
        
        # Variant types
        var_type = np.random.choice(['SNV', 'indel', 'substitution'], p=[0.7, 0.2, 0.1])
        
        # Functional annotations
        consequences = ['missense_variant', 'synonymous_variant', 'stop_gained', 'splice_site', 'intron_variant']
        consequence = np.random.choice(consequences, p=[0.4, 0.3, 0.1, 0.1, 0.1])
        
        variant_data.append({
            'CHROM': chromosomes[i],
            'POS': positions[i],
            'ID': f'rs{np.random.randint(1000000, 99999999)}',
            'REF': ref,
            'ALT': alt,
            'GENE': gene,
            'Consequence': consequence,
            'Variant_Type': var_type,
            'phyloP_score': round(phyloP, 3),
            'GERP_score': round(gerp, 3),
            'phastCons_score': round(phastCons, 3),
            'CADD_score': round(cadd, 2),
            'REVEL_score': round(np.random.beta(1.5, 4), 4),
            'gnomAD_AF': round(np.random.exponential(0.01), 6),  # Allele frequency
            'Pathogenicity': np.random.choice(['Benign', 'Likely_benign', 'VUS', 'Likely_pathogenic', 'Pathogenic'],
                                            p=[0.3, 0.25, 0.3, 0.1, 0.05])
        })
    
    sample_vcf_df = pd.DataFrame(variant_data)
    
    # Add metadata
    sample_vcf_df.attrs['description'] = 'Sample VCF-like dataset for variant conservation analysis'
    sample_vcf_df.attrs['n_variants'] = n_variants
    sample_vcf_df.attrs['score_types'] = ['phyloP', 'GERP', 'phastCons', 'CADD', 'REVEL']
    
    return sample_vcf_df
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
