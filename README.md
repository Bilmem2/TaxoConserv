# TaxoConserv

**A comprehensive web-based platform for statistical analysis and visualization of evolutionary conservation scores across taxonomic groups.**

TaxoConserv enables researchers to analyze conservation metrics (PhyloP, GERP, phastCons) with robust statistical methods, interactive visualizations, and comparative analysis between different taxonomic classifications.

🌐 **Access from Here** https://taxoconserv.streamlit.app/

## Installation

```bash
git clone https://github.com/Bilmem2/TaxoConserv.git
cd TaxoConserv
pip install -r requirements.txt
```

## Usage

### Command Line
```bash
python taxoconserv.py --input data.csv --score phyloP --group taxon
```

### Web Interface (Recommended)
```bash
streamlit run web_taxoconserv.py
```
Open browser to `http://localhost:8501`

**Or use the live demo:** https://taxoconserv.streamlit.app/

## Features

- 🌐 **Web Interface**: 24/7 accessible at https://taxoconserv.streamlit.app/
- 📊 **Statistical Analysis**: Kruskal-Wallis, post-hoc tests, effect sizes
- 📈 **Multiple Visualizations**: Boxplots, violin plots, heatmaps, correlation matrices
- 📱 **Mobile Responsive**: Works on desktop, tablet, and mobile devices
- 🔄 **Interactive Interface**: Real-time analysis and visualization
- 📥 **Data Export**: CSV, JSON, PNG, HTML formats
- 🧬 **Conservation Score Detection**: Auto-detects PhyloP, GERP, phastCons scores

## Quick Start

1. **Try the web application:** https://taxoconserv.streamlit.app/
2. **Load sample data** using the "🧪 Load Sample Data" button
3. **Configure analysis** settings in the sidebar
4. **Run analysis** and view results

## Data Format

CSV or TSV files with:
- Conservation score column (numeric)
- Taxonomic group column (categorical)

Example:
```csv
position,phyloP,taxon_group,gene
123456,2.5,primate,BRCA1
234567,-0.8,mammal,TP53
```

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- matplotlib, seaborn, plotly
- streamlit (for web interface)

## License

Apache License 2.0

Copyright (c) 2025 Can Sevilmiş. All rights reserved.

Licensed under the Apache License, Version 2.0. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

## Structure

```
TaxoConserv/
├── taxoconserv.py          # Main CLI script
├── web_taxoconserv.py      # Web interface
├── src/                    # Core modules
├── data/                   # Example datasets
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Testing

```bash
python -m pytest tests/
```

## Documentation

- 🌐 **Web Interface**: https://taxoconserv.streamlit.app/
- 📋 **Web Interface Guide**: `WEB_INTERFACE_GUIDE.md`

## Citation

If you use TaxoConserv in your research, please cite:

```
TaxoConserv: A Web-Based Platform for Statistical Analysis and Visualization of Evolutionary Conservation Scores Across Taxonomic Classifications
Author: Can Sevilmiş
Year: 2025
URL: https://github.com/Bilmem2/TaxoConserv
DOI: https://zenodo.org/records/16683583
```
