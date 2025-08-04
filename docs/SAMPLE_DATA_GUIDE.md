# TaxoConserv Sample Data Guide

This guide explains how to use the built-in sample datasets in TaxoConserv for testing and demonstration purposes.

## Overview

TaxoConserv provides two types of sample datasets:

1. **Taxonomic Conservation Dataset** - For analyzing conservation patterns across taxonomic groups
2. **Variant Conservation Dataset** - For analyzing conservation scores of individual genetic variants

## 🧬 Taxonomic Conservation Sample Data

### Features
- **200 entries** across 6 major taxonomic groups
- **Multiple conservation scores**: phyloP, GERP, phastCons, CADD, REVEL
- **Realistic patterns**: Different conservation levels per taxonomic group
- **Gene information**: Includes gene names, chromosomes, and positions
- **Outliers included**: For testing statistical robustness

### Taxonomic Groups
- **Primates** (35 entries) - Highest conservation scores
- **Carnivores** (30 entries) - High conservation
- **Rodents** (40 entries) - Moderate conservation
- **Birds** (35 entries) - High conservation
- **Reptiles** (25 entries) - Lower conservation
- **Fish** (35 entries) - Lowest conservation

### Conservation Score Ranges
- **phyloP**: -2.0 to 6.0 (higher = more conserved)
- **GERP**: -5.0 to 8.0 (higher = more conserved)
- **phastCons**: 0.0 to 1.0 (higher = more conserved)
- **CADD**: 0 to 40 (higher = more deleterious)
- **REVEL**: 0.0 to 1.0 (higher = more pathogenic)

### How to Use
1. Go to the **Taxonomic Conservation Analysis** tab
2. Click **"🧪 Load Sample Data"** in the sidebar
3. Configure analysis parameters:
   - **Grouping Column**: `taxon_group`
   - **Score Column**: Choose from `phyloP_score`, `GERP_score`, `phastCons_score`
4. Click **"▶️ Run Analysis"**

### Expected Results
- **Significant differences** between taxonomic groups (Kruskal-Wallis p < 0.05)
- **Primates and Birds** show highest conservation
- **Fish and Reptiles** show lowest conservation
- **Clear separation** in box plots and violin plots

## 🔬 Variant Conservation Sample Data

### Features
- **50 clinically relevant variants**
- **High-impact genes**: BRCA1, BRCA2, TP53, PTEN, KRAS, BRAF, etc.
- **Multiple annotations**: Gene, consequence, pathogenicity predictions
- **Conservation scores**: phyloP, GERP, phastCons, CADD, REVEL
- **Realistic variant types**: SNVs, indels, substitutions

### Variant Categories
- **Pathogenic/Likely Pathogenic** (~15%) - High conservation scores
- **Benign/Likely Benign** (~55%) - Variable conservation scores
- **VUS (Variants of Uncertain Significance)** (~30%) - Mixed conservation patterns

### How to Use
1. Go to the **Variant Conservation Analysis** tab
2. Click **"🧪 Load Sample VCF Data"**
3. Apply filters if desired:
   - Filter by specific genes (e.g., BRCA1, TP53)
   - Filter by pathogenicity classification
4. Click **"🔍 Analyze Conservation Patterns"**

### Expected Results
- **Pathogenic variants** tend to have higher conservation scores
- **Gene-specific patterns** (e.g., BRCA1/2 variants highly conserved)
- **Score correlations** between different conservation metrics
- **Clear distributions** in histograms and box plots

## 🎯 Analysis Recommendations

### For Taxonomic Analysis
1. **Start with phyloP scores** - Most widely used conservation metric
2. **Try different plot types**:
   - **Box plots** - Good for comparing groups
   - **Violin plots** - Shows distribution shapes
   - **Histograms** - Overall score distributions
3. **Enable multi-score analysis** to compare different conservation metrics
4. **Use advanced statistics** for detailed comparisons

### For Variant Analysis
1. **Filter by high-impact genes** (BRCA1, TP53, PTEN) for clearest patterns
2. **Compare pathogenic vs benign variants** to see conservation differences
3. **Examine multiple conservation scores** - they often provide complementary information
4. **Look for outliers** - variants with unexpected conservation patterns

## 📊 Interpretation Guidelines

### Conservation Score Interpretation
- **phyloP > 2.0**: Highly conserved, likely functional
- **GERP > 4.0**: Strong evolutionary constraint
- **phastCons > 0.7**: High probability of conservation
- **CADD > 20**: Likely deleterious if variant
- **REVEL > 0.5**: Likely pathogenic if variant

### Statistical Significance
- **p < 0.05**: Significant difference between groups
- **Effect size (η²)**:
  - Small: 0.01-0.06
  - Medium: 0.06-0.14
  - Large: > 0.14

## 🔍 Troubleshooting

### Common Issues
1. **No significant differences**: Try different conservation scores or grouping methods
2. **Too few data points**: Increase sample size or combine smaller groups
3. **Outliers affecting results**: Use robust statistical methods or filter extreme values
4. **Missing conservation data**: Ensure proper column selection and data format

### Tips for Best Results
- **Use recommended plot types** (auto-detected based on data)
- **Enable interactive plots** for detailed exploration
- **Export results** for external analysis
- **Try different color palettes** for better visualization

## 📝 Citation and Usage

When using TaxoConserv sample data in publications or presentations:

```
Sample conservation datasets generated by TaxoConserv v1.0
(https://github.com/Bilmem2/TaxoConserv)
Simulated data based on realistic conservation score distributions
```

## 🆘 Support

If you encounter issues with sample data:
1. Check the console for error messages
2. Try resetting the interface (🔄 Reset button)
3. Ensure all required modules are installed
4. Report bugs at: https://github.com/Bilmem2/TaxoConserv/issues

---

*This guide covers sample data usage as of TaxoConserv v1.0. For updates and additional features, see the main documentation.*
