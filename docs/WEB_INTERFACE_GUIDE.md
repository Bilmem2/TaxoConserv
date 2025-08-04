# TaxoConserv Web Interface Guide

## Overview

TaxoConserv provides statistical analysis and visualization of taxonomic conservation scores. The web interface enables comparison of conservation scores across taxonomic groups with statistical testing and visualization capabilities.

## 📋 Features

| Feature         | Description                                                      |
|-----------------|------------------------------------------------------------------|
| **Data Input**  | Upload CSV/TSV files or use sample data. Column selection interface. |
| **Conservation Score Detection** | Automatic detection of PhyloP, phastCons, GERP++, SiPhy columns. |
| **Advanced Grouping** | Dynamic filtering, hierarchical grouping, statistical clustering, custom mapping. |
| **Multi-Score Analysis** | Correlation analysis, pairwise comparisons, score relationships. |
| **Advanced Statistics** | Assumption testing, parametric/non-parametric tests, effect sizes. |
| **Advanced Visualizations** | Ridgeline plots, raincloud plots, 3D scatter, forest plots. |
| **Statistics**  | Kruskal-Wallis test, summary statistics (mean, median, std, min, max). |
| **Visualization**| Boxplot, violin plot, heatmap, interactive Plotly charts. |
| **Export**      | Results as CSV/JSON, plots as PNG/HTML. |

## Usage

1. **📁 Upload Data:** Upload CSV/TSV file or load sample dataset.
2. **⚙️ Select Columns:** Choose grouping and score columns.
3. **🔧 Configure Analysis:** Enable advanced options if needed:
   - Advanced Group Selection for filtering and clustering
   - Multi-Score Analysis for comparing multiple scores
   - Advanced Statistics for comprehensive testing
   - Advanced Visualizations for specialized plots
4. **▶️ Run Analysis:** Click "Run Analysis" to generate results.
5. **💾 Export Results:** Download analysis results and visualizations.

## 📊 Data Format

| taxon_group | phyloP_score | phastCons_score | GERP_score | family   | genus   | species    |
|-------------|-------------|----------------|------------|----------|---------|------------|
| Mammals     | 2.5         | 0.85           | 1.8        | Mammalia | Canis   | Species_1  |
| Birds       | 1.8         | 0.92           | 1.2        | Aves     | Corvus  | Species_2  |
| Reptiles    | 0.5         | 0.67           | -0.5       | Reptilia | Lacerta | Species_3  |

**Supported Conservation Scores:**

- **PhyloP**: `phyloP`, `phyloP_score`, `phylop`, `phylop_score` (range: -10 to 10)
- **PhastCons**: `phastCons`, `phastCons_score`, `phastcons`, `phastcons_score` (range: 0 to 1)
- **GERP**: `GERP`, `GERP++`, `GERP_score`, `gerp`, `gerp_score` (range: -10 to 10)
- **SiPhy**: `SiPhy`, `SiPhy_score`, `siphy`, `siphy_score` (range: 0 to 1)
- **Generic**: `conservation_score`, `conservation`, `score` (numeric values)

## Advanced Features

### 🎛️ Advanced Group Selection
- Dynamic filtering by group properties
- Hierarchical taxonomic classifications
- Statistical clustering methods
- Custom group mapping and merging

### 🔬 Multi-Score Analysis
- Automatic conservation score detection
- Correlation analysis (Pearson, Spearman, Kendall)
- Pairwise score comparisons
- Interactive correlation matrices

### 📈 Advanced Statistics
- Normality testing (Shapiro-Wilk, D'Agostino)
- Parametric tests (ANOVA with Tukey HSD)
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney U)
- Effect size calculations

### 📉 Advanced Visualizations
- Ridgeline and raincloud plots
- 3D scatter plots for multi-dimensional data
- Forest plots for effect sizes
- Q-Q plots and diagnostic visualizations

## 🧬 Applications

- Conservation genomics
- Comparative genomics
- Evolutionary biology
- Population genetics

## 📊 Statistical Methods

**Input Requirements:**
- Numeric conservation score column
- Categorical taxonomic group column
- Optional hierarchical columns (family, genus, species)

**Conservation Score Detection:**
- Automatic detection of common score types
- Score prioritization in column selection
- Type-specific value range descriptions

**Statistical Testing:**
- Kruskal-Wallis test for multi-group comparisons
- Mann-Whitney U test for two-group comparisons
- Post-hoc pairwise comparisons for significant results
- Summary statistics for all groups

**Visualization:**
- Distribution plots (boxplot, violin, swarm)
- Group-wise heatmaps
- Interactive exploration and export

## ❗ Troubleshooting

**📁 File Upload Issues:**
- Ensure CSV/TSV format with proper column headers
- Score columns must be numeric, group columns categorical
- Large files may require processing time

**⚙️ Analysis Issues:**
- Small groups may generate warnings; consider merging
- Missing values prevent analysis
- Multi-score analysis requires 2+ numeric columns

**📊 Visualization Issues:**
- Select appropriate plot types for your data
- Interactive plots require modern browser versions
- PNG downloads require kaleido package

**💾 Export Issues:**
- Allow downloads in browser settings
- Check pop-up blockers

## 📚 References

Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(260), 583-621.
