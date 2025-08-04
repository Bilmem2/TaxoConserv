# TaxoConserv

🧬 **Computational Platform for Evolutionary Conservation Analysis**

TaxoConserv is a bioinformatics tool designed for statistical analysis and visualization of evolutionary conservation scores. This project supports researchers and bioinformaticians in understanding conservation patterns across species.

## Features

### Core Analysis Modes
- **📊 Taxonomic Conservation Analysis**: Statistical comparison across taxonomic classifications
- **🧪 Variant Conservation Analysis**: Individual variant assessment from VCF files
- **📈 Multi-Score Analysis**: Simultaneous analysis of multiple conservation metrics

### Capabilities
- **⚡ Performance Optimized**: Data sampling for large datasets (50K+ rows)
- **📋 Statistical Testing**: Kruskal-Wallis, Mann-Whitney U, post-hoc comparisons
- **🎨 Visualizations**: Box plots, violin plots, heatmaps, correlation matrices
- **📁 Flexible Input**: CSV/TSV, VCF files with automatic format detection
- **💾 Export Options**: CSV, JSON, PNG, HTML formats

### Conservation Metrics Supported
- PhyloP scores
- GERP scores  
- phastCons scores
- Custom conservation metrics

## 🌐 Quick Start

### Web Interface (Recommended)
Access the interactive web application:
**🔗 https://taxoconserv.streamlit.app/**

### Command Line Interface
```bash
# Clone the repository
git clone https://github.com/Bilmem2/TaxoConserv.git
cd TaxoConserv

# Install dependencies
pip install -r requirements.txt

# Run taxonomic analysis
python taxoconserv.py --input data.csv --score phyloP --group taxon

# Run web interface locally
python -m streamlit run web_taxoconserv.py
```

## 📁 Project Structure

```
TaxoConserv/
├── src/                          # Core analysis modules
│   ├── analysis.py              # Statistical analysis engine
│   ├── visualization.py         # Plotting and visualization
│   ├── performance_optimizer.py # Performance optimization
│   └── ...
├── data/                        # Example datasets
├── test_datasets/              # Comprehensive test data
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── taxoconserv.py             # CLI application
├── web_taxoconserv.py         # Web interface
└── requirements.txt           # Dependencies
```

## 📊 Data Requirements

### Taxonomic Analysis
- **Format**: CSV/TSV files
- **Required columns**: Conservation score column + taxonomic group column
- **Example**: `species,phyloP_score` or `taxon_group,conservation_value`

### Variant Analysis  
- **Format**: VCF files
- **Required**: Conservation scores in INFO field
- **Supported**: Standard VCF format with custom INFO tags

## 🔧 Performance Features

- **Smart Sampling**: Automatic data sampling for datasets >10K rows
- **Memory Optimization**: Efficient processing of large files
- **Progress Tracking**: Real-time analysis progress indicators
- **Plot Optimization**: Automatic plot type selection for optimal performance

## 📚 Documentation

- **[User Guide](docs/USER_GUIDE.md)**: Comprehensive usage instructions
- **[Web Interface Guide](docs/WEB_INTERFACE_GUIDE.md)**: Web app tutorial  
- **[Sample Data Guide](docs/SAMPLE_DATA_GUIDE.md)**: Data format examples
- **[ACMG Analysis](docs/ACMG_ANALYSIS.md)**: Clinical variant interpretation

## 🧪 Test Datasets

The `test_datasets/` directory includes test data:
- Small datasets (200-1K rows) for quick testing
- Medium datasets (10K rows) for performance testing  
- Large datasets (50K+ rows) for stress testing
- Edge cases and validation scenarios

## 🛠️ Requirements

- **Python**: 3.8+
- **Core Libraries**: pandas, numpy, scipy, matplotlib, plotly, streamlit
- **Optional**: duckdb (for large dataset optimization)

## 📄 License

```
Apache License 2.0
Copyright (c) 2025 Can Sevilmiş

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 📧 Contact

**Can Sevilmiş** - Student Project  
🔗 GitHub: [Bilmem2/TaxoConserv](https://github.com/Bilmem2/TaxoConserv)

---

*TaxoConserv v2.0.0 - Evolutionary Conservation Analysis Platform*
