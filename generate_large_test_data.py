#!/usr/bin/env python3
"""
Large Test Dataset Generator for TaxoConserv
Generates realistic conservation score data for performance testing
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_large_conservation_dataset(n_rows=50000, output_file="taxoconserv_large_test.csv"):
    """
    Generate a large dataset with realistic conservation scores
    
    Parameters:
    - n_rows: Number of rows to generate (default: 50000)
    - output_file: Output CSV filename
    """
    
    print(f"🧬 Generating large conservation dataset with {n_rows:,} rows...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic taxonomic groups with different sizes
    taxonomic_groups = {
        'Mammals': 0.25,
        'Birds': 0.20,
        'Reptiles': 0.15,
        'Amphibians': 0.10,
        'Fish': 0.15,
        'Invertebrates': 0.10,
        'Plants': 0.05
    }
    
    # Conservation score types with realistic ranges
    score_configs = {
        'PhyloP_Score': {'mean': 0.1, 'std': 0.8, 'range': (-20, 20)},
        'PhastCons_Score': {'mean': 0.3, 'std': 0.25, 'range': (0, 1)},
        'GERP_Score': {'mean': 0.5, 'std': 2.0, 'range': (-12, 6)},
        'CADD_Score': {'mean': 15, 'std': 8, 'range': (0, 40)},
        'REVEL_Score': {'mean': 0.4, 'std': 0.25, 'range': (0, 1)},
        'SIFT_Score': {'mean': 0.3, 'std': 0.25, 'range': (0, 1)}
    }
    
    # Generate species names
    genus_prefixes = ['Homo', 'Pan', 'Mus', 'Drosophila', 'Caenorhabditis', 'Arabidopsis', 
                     'Rattus', 'Gallus', 'Danio', 'Xenopus', 'Canis', 'Felis', 'Bos', 
                     'Sus', 'Equus', 'Ovis', 'Macaca', 'Gorilla', 'Pongo', 'Chlorocebus']
    
    species_suffixes = ['sapiens', 'troglodytes', 'musculus', 'melanogaster', 'elegans',
                       'thaliana', 'norvegicus', 'gallus', 'rerio', 'laevis', 'lupus',
                       'catus', 'taurus', 'scrofa', 'caballus', 'aries', 'mulatta',
                       'gorilla', 'abelii', 'sabaeus', 'domesticus', 'familiaris']
    
    # Chromosome names
    chromosomes = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10',
                  'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20',
                  'chr21', 'chr22', 'chrX', 'chrY', 'chrM']
    
    # Initialize data container
    data = []
    
    print("📊 Generating data rows...")
    
    for i in range(n_rows):
        if i % 10000 == 0:
            print(f"  Progress: {i:,}/{n_rows:,} ({i/n_rows*100:.1f}%)")
        
        # Select taxonomic group based on weights
        group = np.random.choice(list(taxonomic_groups.keys()), 
                               p=list(taxonomic_groups.values()))
        
        # Generate species name
        genus = random.choice(genus_prefixes)
        species = random.choice(species_suffixes)
        species_name = f"{genus} {species}"
        
        # Generate genomic coordinates
        chromosome = random.choice(chromosomes)
        position = np.random.randint(1000, 300000000)  # Realistic genomic positions
        
        # Generate conservation scores with group-specific characteristics
        scores = {}
        
        # Add group-specific bias to make groups statistically different
        group_bias = {
            'Mammals': 0.3,
            'Birds': 0.2,
            'Reptiles': 0.1,
            'Amphibians': 0.0,
            'Fish': -0.1,
            'Invertebrates': -0.2,
            'Plants': -0.3
        }
        
        bias = group_bias.get(group, 0)
        
        for score_name, config in score_configs.items():
            # Generate score with group bias
            score = np.random.normal(config['mean'] + bias, config['std'])
            
            # Clip to realistic range
            score = np.clip(score, config['range'][0], config['range'][1])
            
            # Add some missing values (5% chance)
            if np.random.random() < 0.05:
                score = np.nan
            
            scores[score_name] = score
        
        # Generate functional annotation
        functions = ['Coding', 'Non-coding', 'Regulatory', 'Intergenic', 'UTR', 'Intron', 'Exon']
        function = random.choice(functions)
        
        # Generate conservation class based on primary score
        primary_score = scores.get('PhyloP_Score', 0)
        if pd.isna(primary_score):
            conservation_class = 'Unknown'
        elif primary_score > 2:
            conservation_class = 'Highly Conserved'
        elif primary_score > 0.5:
            conservation_class = 'Moderately Conserved'
        elif primary_score > -0.5:
            conservation_class = 'Neutral'
        else:
            conservation_class = 'Not Conserved'
        
        # Create row
        row = {
            'ID': f"variant_{i+1:06d}",
            'Species': species_name,
            'Taxonomic_Group': group,
            'Chromosome': chromosome,
            'Position': position,
            'Function': function,
            'Conservation_Class': conservation_class,
            **scores  # Add all conservation scores
        }
        
        data.append(row)
    
    print("💾 Creating DataFrame...")
    df = pd.DataFrame(data)
    
    # Add some additional calculated columns
    print("🧮 Adding calculated columns...")
    
    # Average conservation score
    score_columns = list(score_configs.keys())
    df['Average_Conservation'] = df[score_columns].mean(axis=1, skipna=True)
    
    # Conservation rank within group
    df['Group_Rank'] = df.groupby('Taxonomic_Group')['Average_Conservation'].rank(method='dense', na_option='bottom')
    
    # Add timestamp
    df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📈 Dataset Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  Taxonomic groups: {df['Taxonomic_Group'].value_counts().to_dict()}")
    
    # Show missing data summary
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(1)
    print(f"\n📊 Missing Data Summary:")
    for col, count in missing_summary.items():
        if count > 0:
            print(f"  {col}: {count:,} ({missing_pct[col]}%)")
    
    # Save to file
    print(f"\n💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # File size
    import os
    file_size = os.path.getsize(output_file) / 1024**2
    print(f"✅ Saved! File size: {file_size:.1f} MB")
    
    # Performance testing hints
    print(f"\n🚀 Performance Testing Suggestions:")
    print(f"  1. Test with different group columns: 'Taxonomic_Group', 'Conservation_Class', 'Function'")
    print(f"  2. Test with different score columns: {', '.join(score_columns[:3])}")
    print(f"  3. Monitor memory usage and cache efficiency")
    print(f"  4. Test statistical analysis performance with {len(df):,} rows")
    print(f"  5. Test export functionality with large datasets")
    
    return df

def generate_multiple_test_files():
    """Generate test files of different sizes"""
    sizes = [
        (10000, "taxoconserv_medium_test.csv"),
        (50000, "taxoconserv_large_test.csv"),
        (100000, "taxoconserv_xlarge_test.csv")
    ]
    
    for size, filename in sizes:
        print(f"\n{'='*60}")
        generate_large_conservation_dataset(size, filename)

if __name__ == "__main__":
    print("🧬 TaxoConserv Large Dataset Generator")
    print("=====================================")
    
    # Ask user for size preference
    print("\nSelect dataset size:")
    print("1. Medium (10K rows) - Quick testing")
    print("2. Large (50K rows) - Performance testing")
    print("3. Extra Large (100K rows) - Stress testing")
    print("4. All sizes")
    
    choice = input("\nEnter choice (1-4) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        generate_large_conservation_dataset(10000, "taxoconserv_medium_test.csv")
    elif choice == "2":
        generate_large_conservation_dataset(50000, "taxoconserv_large_test.csv")
    elif choice == "3":
        generate_large_conservation_dataset(100000, "taxoconserv_xlarge_test.csv")
    elif choice == "4":
        generate_multiple_test_files()
    else:
        print("Invalid choice, generating default large dataset...")
        generate_large_conservation_dataset(50000, "taxoconserv_large_test.csv")
    
    print(f"\n🎉 Test data generation completed!")
    print(f"📁 Files are ready for testing in TaxoConserv application")
