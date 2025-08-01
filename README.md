# TaxoConserv

**A computational platform for statistical analysis and visualization of evolutionary conservation scores**

Execute the test suite:
```bash
python -m pytest tests/
```

## Documentation

### Reference Materials
- **Web Application**: https://taxoconserv.streamlit.app/
- **User Guide**: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- **Web Interface Reference**: [`docs/WEB_INTERFACE_GUIDE.md`](docs/WEB_INTERFACE_GUIDE.md)
- **Sample Data Guide**: [`docs/SAMPLE_DATA_GUIDE.md`](docs/SAMPLE_DATA_GUIDE.md)

### Technical Documentation
- **Testing Framework**: [`tests/README_TEST_PLAN.md`](tests/README_TEST_PLAN.md)
- **Performance Considerations**: [`tests/README_PERFORMANCE_SECURITY.md`](tests/README_PERFORMANCE_SECURITY.md)nary analysis through two complementary approaches:
- **Taxonomic Conservation Analysis**: Statistical comparison of conservation metrics across taxonomic classifications
- **Variant Conservation Analysis**: Individual variant assessment from VCF files with conservation score annotation

The software implements established conservation metrics (PhyloP, GERP, phastCons) with appropriate statistical methods for hypothesis testing and data visualization.

**Web Interface**: https://taxoconserv.streamlit.app/

## Installation

For command-line usage:
```bash
git clone https://github.com/Bilmem2/TaxoConserv.git
cd TaxoConserv
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)
Access the live application at https://taxoconserv.streamlit.app/

1. Select analysis mode (Taxonomic or Variant Conservation)
2. Upload data files or load provided sample datasets
3. Configure analysis parameters using the sidebar controls
4. Execute analysis and review statistical results
5. Export findings in preferred format

### Command Line Interface
For programmatic use or batch processing:
```bash
python taxoconserv.py --input data.csv --score phyloP --group taxon
```

## Methods and Features

### Statistical Analysis
- Non-parametric testing (Kruskal-Wallis H-test)
- Post-hoc pairwise comparisons with multiple testing correction
- Effect size calculations (eta-squared, Cohen's d)
- Descriptive statistics and distribution analysis

### Supported Conservation Metrics
- PhyloP: Phylogenetic p-values for conservation/acceleration
- GERP: Genomic Evolutionary Rate Profiling scores
- phastCons: Hidden Markov Model-based conservation probabilities

### Visualization Methods
- Box plots and violin plots for distribution comparison
- Scatter plots for correlation analysis
- Heatmaps for score matrices
- Statistical summary tables

### Input/Output Capabilities
- CSV/TSV file parsing for taxonomic data
- VCF format support (including compressed files)
- Multiple export formats (CSV, JSON, PNG, HTML)
- Sample dataset generation for testing

### Analysis Modes

**Taxonomic Conservation Analysis**
- Comparative analysis across user-defined taxonomic groups
- Statistical hypothesis testing between groups
- Distribution visualization and summary statistics

**Variant Conservation Analysis**
- VCF file processing with conservation score extraction
- Per-variant conservation assessment
- Genomic coordinate-based analysis

## Usage

### Web Interface
1. Navigate to https://taxoconserv.streamlit.app/
2. Select analysis mode (Taxonomic or Variant Conservation)
3. Upload data files or load provided sample datasets
4. Configure analysis parameters using the sidebar controls
5. Execute analysis and review statistical results
6. Export findings in preferred format

### Command Line Usage
For programmatic access or integration into analysis pipelines:
```bash
python taxoconserv.py --input your_data.csv --score phyloP --group taxon_group
```

## Input Data Requirements

### Taxonomic Conservation Analysis
Input files should be in CSV or TSV format containing:

```csv
position,phyloP,taxon_group,gene
123456,2.5,primate,BRCA1
234567,-0.8,mammal,TP53
345678,1.2,vertebrate,EGFR
```

**Required fields:**
- Conservation score column (continuous numerical values)
- Taxonomic group column (categorical classification labels)

### Variant Conservation Analysis
Standard VCF format files (.vcf or .vcf.gz compression):

```vcf
##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	1000	.	A	G	60	PASS	PhyloP=2.5;GERP=1.8
chr2	2000	.	C	T	90	PASS	phastCons=0.95;PhyloP=3.2
```

**Supported conservation annotations in INFO field:**
- PhyloP scores: PhyloP, phyloP, PHYLOP
- GERP scores: GERP, gerp, GERP_RS  
- phastCons scores: phastCons, PhastCons, PHASTCONS

## Dependencies

### Core Requirements
- Python 3.8 or higher
- pandas, numpy, scipy (numerical computing and statistics)
- matplotlib, seaborn, plotly (data visualization)
- streamlit (web application framework)

### Additional Libraries
- PyVCF3 (VCF file parsing)
- statsmodels (statistical modeling)
- duckdb (query optimization)
- requests (HTTP functionality)

### Installation Commands
For command-line usage:
```bash
pip install -r requirements.txt
```

For minimal installation (core functionality only):
```bash
pip install pandas numpy scipy matplotlib
```

## License

Apache License 2.0

Copyright (c) 2025 Can Sevilmiş. All rights reserved.

Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

## Repository Structure

```
TaxoConserv/
├── taxoconserv.py              # Command-line interface
├── web_taxoconserv.py          # Web application interface
├── src/                        # Core analysis modules
│   ├── analysis.py             # Statistical analysis implementation
│   ├── variant_analysis.py     # VCF processing and variant analysis
│   ├── visualization.py       # Plotting functions
│   ├── input_parser.py         # Data parsing and validation
│   ├── taxon_grouping.py       # Taxonomic classification utilities
│   ├── advanced_statistics.py  # Extended statistical methods
│   └── performance_optimizer.py # Computational optimizations
├── data/                       # Example datasets
│   ├── testdata_standard.csv   # Standard taxonomic test data
│   ├── test_variants.vcf       # Sample VCF file
│   └── example_conservation_scores.csv
├── tests/                      # Test suite
├── docs/                       # Documentation
│   ├── USER_GUIDE.md           # User documentation
│   ├── WEB_INTERFACE_GUIDE.md  # Web interface reference
│   └── SAMPLE_DATA_GUIDE.md    # Sample data descriptions
└── requirements*.txt           # Python dependencies
```

## Testing

```bash
python -m pytest tests/
```

## Citation

When using TaxoConserv in research publications, please cite:

```
TaxoConserv: A Computational Platform for Statistical Analysis and Visualization 
of Evolutionary Conservation Scores Across Taxonomic Classifications
Author: Can Sevilmiş
Year: 2025
URL: https://github.com/Bilmem2/TaxoConserv
DOI: https://zenodo.org/records/16683583
```
