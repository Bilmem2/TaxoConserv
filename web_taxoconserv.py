#!/usr/bin/env python3
"""
TaxoConserv - Clinical Variant Conservation Analysis Platform
Web Interface Module

Copyright 2025 Can Sevilmiş

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

DISCLAIMER:
This software is provided "AS IS" and any express or implied warranties,
including, but not limited to, the implied warranties of merchantability
and fitness for a particular purpose are disclaimed. In no event shall
the copyright holder or contributors be liable for any direct, indirect,
incidental, special, exemplary, or consequential damages (including, but
not limited to, procurement of substitute goods or services; loss of use,
data, or profits; or business interruption) however caused and on any
theory of liability, whether in contract, strict liability, or tort
(including negligence or otherwise) arising in any way out of the use
of this software, even if advised of the possibility of such damage.

For research and educational purposes only. Not intended for clinical or
diagnostic use. Users are responsible for validating results and ensuring
compliance with applicable regulations and ethical guidelines.

Usage:
    streamlit run web_taxoconserv.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit.components.v1 as components
import io
import sys
import os
import json
import base64
import time
import traceback
from pathlib import Path
from datetime import datetime
from scipy.stats import kruskal

# Import version info from main module
sys.path.insert(0, str(Path(__file__).parent))

# Add src directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

    # Try to import custom modules
try:
    # Performance optimization system
    from src.performance_optimizer import (
        cache, performance_timer, streamlit_cache_data,
        cached_statistical_analysis, cached_data_processing,
        cached_conservation_score_detection, DataOptimizer,
        performance_monitor, optimize_streamlit_config,
        lazy_loader
    )
    
    # DuckDB integration for large datasets
    from src.duckdb_integration import (
        duckdb_processor, get_fast_group_statistics,
        get_fast_outlier_detection, optimize_large_dataset, 
        DUCKDB_AVAILABLE
    )
    
    # Conservation Score Detection and Description Dictionary
    from src.score_utils import (
        CONSERVATION_SCORE_PATTERNS,
        detect_conservation_scores,
        get_score_description,
        prioritize_conservation_scores
    )
    
    # Demo veri fonksiyonu
    from src.input_parser import create_demo_data
    
    # Statistical analysis fonksiyonları
    from src.analysis import perform_statistical_analysis
    
    # Advanced plot recommendations
    from src.score_plot_mapping import score_plot_mapper, get_enhanced_plot_info
    
    # Multi-score analysis system
    from src.multi_score_analysis import create_multi_score_analysis_ui, MultiScoreAnalyzer
    
    # Advanced statistics system (simplified usage)
    from src.advanced_statistics import create_advanced_statistics_ui, AdvancedStatisticalAnalyzer    # Try importing optional packages  
    sp = None  # Default to None for optional dependency
    try:
        # Suppress Pylance import warnings for optional package
        exec("import scikit_posthocs as sp; globals()['sp'] = sp")
    except ImportError:
        pass  # sp remains None - graceful degradation
    
    MODULES_AVAILABLE = True
except Exception as e:
    MODULES_AVAILABLE = False
    st.error(f"⚠️ Module import warning: {e}")
    # Create fallback functions if imports fail
    def detect_conservation_scores(data):
        return {}
    def prioritize_conservation_scores(score_options, detected_scores):
        return score_options
    def create_demo_data():
        return pd.DataFrame()

# Smart column suggestion functions
def suggest_grouping_column(data):
    """Suggest the best column for taxonomic grouping based on column names and data patterns."""
    if data is None or data.empty:
        return None
    
    # Priority keywords for grouping columns (in order of preference)
    grouping_keywords = [
        # Taxonomic terms (highest priority)
        ['taxon', 'taxonomy', 'taxonomic', 'group', 'clade'],
        ['species', 'genus', 'family', 'order', 'class', 'phylum'],
        ['organism', 'taxa', 'lineage', 'phylo'],
        # General grouping terms
        ['category', 'type', 'class', 'cluster', 'classification'],
        ['population', 'sample', 'cohort', 'batch'],
        # ID columns (lower priority but still useful)
        ['id', 'identifier', 'name', 'label']
    ]
    
    non_numeric_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
    
    # Score columns based on keyword matches and data characteristics
    scored_columns = []
    
    for col in non_numeric_cols:
        col_lower = col.lower()
        score = 0
        priority_level = 10  # Higher priority gets lower number
        
        # Check keyword matches with different priority levels
        for level, keyword_group in enumerate(grouping_keywords):
            for keyword in keyword_group:
                if keyword in col_lower:
                    score += 100 - (level * 10)  # Higher score for higher priority keywords
                    priority_level = min(priority_level, level)
                    break
        
        # Analyze data characteristics
        unique_values = data[col].nunique()
        total_values = len(data[col].dropna())
        
        if total_values > 0:
            # Prefer columns with reasonable number of groups (2-20)
            if 2 <= unique_values <= 20:
                score += 50
            elif unique_values > 20 and unique_values < total_values * 0.8:
                score += 30  # Many groups but not too many
            elif unique_values == 1:
                score -= 100  # Single value columns are useless
            
            # Prefer columns with good coverage (few missing values)
            coverage = total_values / len(data)
            score += int(coverage * 20)
            
            # Prefer columns with string/categorical data
            if data[col].dtype == 'object':
                score += 10
        
        scored_columns.append((col, score, priority_level, unique_values))
    
    # Sort by score (descending) and priority level (ascending)
    scored_columns.sort(key=lambda x: (-x[1], x[2]))
    
    return scored_columns[0][0] if scored_columns else (non_numeric_cols[0] if non_numeric_cols else None)

def suggest_conservation_score_column(data):
    """Suggest the best column for conservation scores based on column names and data patterns."""
    if data is None or data.empty:
        return None
    
    # Priority keywords for conservation score columns
    conservation_keywords = [
        # Conservation-specific scores (highest priority)
        ['phylop', 'phylo_p', 'phastcons', 'phast_cons', 'gerp', 'siphy'],
        ['conservation', 'conserved', 'conserve', 'evolutionary'],
        # Pathogenicity and functional scores
        ['cadd', 'revel', 'polyphen', 'sift', 'fathmm', 'metasvm', 'metalr'],
        ['pathogenic', 'pathogenicity', 'deleteriousness', 'deleterious'],
        ['functional', 'impact', 'effect', 'damage', 'harmful'],
        # General scoring terms
        ['score', 'value', 'metric', 'rating', 'index'],
        # Statistical/probability terms
        ['probability', 'likelihood', 'confidence', 'pvalue', 'p_value']
    ]
    
    numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    
    # Score columns based on keyword matches and data characteristics
    scored_columns = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        score = 0
        priority_level = 10
        
        # Check keyword matches with different priority levels
        for level, keyword_group in enumerate(conservation_keywords):
            for keyword in keyword_group:
                if keyword in col_lower:
                    score += 100 - (level * 10)
                    priority_level = min(priority_level, level)
                    break
        
        # Analyze data characteristics
        col_data = data[col].dropna()
        if len(col_data) > 0:
            # Prefer columns with reasonable value ranges
            data_range = col_data.max() - col_data.min()
            if data_range > 0:
                score += 10
            
            # Prefer columns with good coverage (few missing values)
            coverage = len(col_data) / len(data)
            score += int(coverage * 30)
            
            # Prefer floating point numbers (more likely to be scores)
            if data[col].dtype in ['float64', 'float32']:
                score += 15
            
            # Check if values are in typical score ranges
            min_val, max_val = col_data.min(), col_data.max()
            
            # Common conservation score ranges
            if -10 <= min_val and max_val <= 10:  # phyloP-like
                score += 25
            elif 0 <= min_val and max_val <= 1:   # probability/normalized scores
                score += 20
            elif 0 <= min_val and max_val <= 100: # percentage-like or CADD
                score += 15
        
        scored_columns.append((col, score, priority_level))
    
    # Sort by score (descending) and priority level (ascending)
    scored_columns.sort(key=lambda x: (-x[1], x[2]))
    
    return scored_columns[0][0] if scored_columns else (numeric_cols[0] if numeric_cols else None)

def get_column_suggestions(data):
    """Get suggestions for both grouping and score columns with confidence indicators."""
    if data is None or data.empty:
        return None, None, {}, {}
    
    # Get suggestions
    suggested_group = suggest_grouping_column(data)
    suggested_score = suggest_conservation_score_column(data)
    
    # Calculate confidence for grouping column
    group_confidence = {}
    if suggested_group:
        non_numeric_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        
        for col in non_numeric_cols[:3]:  # Top 3 candidates
            col_lower = col.lower()
            confidence = 0
            
            # Keyword-based confidence
            grouping_keywords_flat = ['taxon', 'group', 'species', 'genus', 'family', 'category', 'type']
            for keyword in grouping_keywords_flat:
                if keyword in col_lower:
                    confidence += 30
            
            # Data-based confidence
            unique_ratio = data[col].nunique() / len(data[col].dropna()) if len(data[col].dropna()) > 0 else 0
            if 0.1 <= unique_ratio <= 0.8:  # Good grouping ratio
                confidence += 40
                
            coverage = len(data[col].dropna()) / len(data)
            confidence += int(coverage * 30)
            
            group_confidence[col] = min(confidence, 100)
    
    # Calculate confidence for score column
    score_confidence = {}
    if suggested_score:
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        for col in numeric_cols[:3]:  # Top 3 candidates
            col_lower = col.lower()
            confidence = 0
            
            # Keyword-based confidence
            score_keywords_flat = ['phylop', 'gerp', 'cadd', 'conservation', 'score', 'value']
            for keyword in score_keywords_flat:
                if keyword in col_lower:
                    confidence += 40
            
            # Data-based confidence
            coverage = len(data[col].dropna()) / len(data)
            confidence += int(coverage * 30)
            
            # Range-based confidence
            col_data = data[col].dropna()
            if len(col_data) > 0:
                min_val, max_val = col_data.min(), col_data.max()
                if -10 <= min_val and max_val <= 10:
                    confidence += 30
                elif 0 <= min_val and max_val <= 1:
                    confidence += 25
                elif 0 <= min_val and max_val <= 100:
                    confidence += 20
            
            score_confidence[col] = min(confidence, 100)
    
    return suggested_group, suggested_score, group_confidence, score_confidence

# Initialize performance optimization
try:
    optimize_streamlit_config()
except:
    pass  # Graceful fallback if optimization fails
    def perform_statistical_analysis(data, score_column, taxon_column):
        return {
            'h_statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'n_groups': 0,
            'stats_summary': pd.DataFrame()
        }



@performance_timer
def create_local_visualization(data, score_column, taxon_column, plot_type, color_palette, interactive=False):
    """Create visualization based on selected parameters."""
    try:
        if interactive:
            fig = None
            # Detect Streamlit theme
            theme = st.get_option("theme.base") if hasattr(st, "get_option") else None
            # Choose Plotly template and font color based on theme
            if theme == "dark":
                plotly_template = "plotly_dark"
                font_color = "#eaeaea"
            else:
                plotly_template = "plotly_white"
                font_color = "#222"
            # Create interactive plot with Plotly
            if plot_type == "boxplot":
                fig = px.box(data, x=taxon_column, y=score_column, title=f"Conservation Score Distribution by Taxonomic Group", color=taxon_column, color_discrete_sequence=px.colors.qualitative.Set3)
            elif plot_type == "violin":
                fig = px.violin(data, x=taxon_column, y=score_column, title=f"Conservation Score Distribution by Taxonomic Group", color=taxon_column, color_discrete_sequence=px.colors.qualitative.Set3, box=True)
            elif plot_type == "swarm":
                fig = px.strip(data, x=taxon_column, y=score_column, title=f"Conservation Score Distribution by Taxonomic Group", color=taxon_column, color_discrete_sequence=px.colors.qualitative.Set3)
            elif plot_type == "histogram":
                fig = px.histogram(data, x=score_column, title=f"Distribution of {score_column.replace('_', ' ').title()}", nbins=20, color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(xaxis_title=score_column.replace('_', ' ').title(), yaxis_title="Frequency")
            elif plot_type == "heatmap":
                data = data.copy()
                data['row_group'] = (data.reset_index().index // 10)
                pivot_data = data.pivot_table(values=score_column, index='row_group', columns=taxon_column, aggfunc='mean')
                plotly_supported_scales = [
                    "viridis", "plasma", "cividis", "magma", "inferno", "turbo", "blues", "reds", "greens", "picnic", "rainbow", "jet", "hot", "icefire", "twilight"
                ]
                plotly_palette = color_palette.lower()
                if plotly_palette not in plotly_supported_scales:
                    plotly_palette = "viridis"
                fig = px.imshow(pivot_data, title="Conservation Score Heatmap", color_continuous_scale=plotly_palette)
            elif plot_type == "barplot":
                group_means = data.groupby(taxon_column)[score_column].mean().sort_values(ascending=False)
                fig = px.bar(x=group_means.index, y=group_means.values, title=f"Mean {score_column.replace('_', ' ').title()} by {taxon_column.replace('_', ' ').title()}", color=group_means.index, color_discrete_sequence=px.colors.qualitative.Set3, text=group_means.values.round(3))
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_title=taxon_column.replace('_', ' ').title(), yaxis_title=f"Mean {score_column.replace('_', ' ').title()}")
            elif plot_type == "kde":
                fig = px.histogram(data, x=score_column, color=taxon_column, marginal="violin", nbins=20, title=f"KDE Plot of {score_column.replace('_', ' ').title()} by Group", template=plotly_template, color_discrete_sequence=px.colors.qualitative.Set3)
            elif plot_type == "pairplot":
                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                if len(num_cols) < 2:
                    return None, None
                fig = px.scatter_matrix(data, dimensions=num_cols, title="Pairplot of Numeric Features", template=plotly_template)
                
            if fig is not None:
                # Başlık metni her plot türü için sabit ve doğru şekilde atanıyor
                if plot_type == "boxplot" or plot_type == "violin" or plot_type == "swarm":
                    plot_title = "Conservation Score Distribution by Taxonomic Group"
                elif plot_type == "kde":
                    plot_title = f"KDE Plot of {score_column.replace('_', ' ').title()} by Group"
                elif plot_type == "correlation":
                    plot_title = "Correlation Matrix"
                elif plot_type == "pairplot":
                    plot_title = "Pairplot of Numeric Features"
                elif plot_type == "barplot":
                    plot_title = f"Mean {score_column.replace('_', ' ').title()} by {taxon_column.replace('_', ' ').title()}"
                elif plot_type == "heatmap":
                    plot_title = "Conservation Score Heatmap"
                elif plot_type == "histogram":
                    plot_title = f"Distribution of {score_column.replace('_', ' ').title()}"
                else:
                    plot_title = "Conservation Score Distribution"
                fig.update_layout(
                    template=plotly_template,
                    font=dict(size=14, color=font_color),
                    title={
                        'text': plot_title,
                        'font': {'size': 20, 'color': font_color, 'family': 'Arial Black', 'weight': 'bold'},
                        'x': 0.5,
                        'xanchor': 'center',
                        'y': 0.95,
                        'yanchor': 'top'
                    },
                    title_font_color=font_color,
                    title_font_size=20,
                    width=800, height=600
                )
                # Force title color by updating layout again to override template
                fig.update_layout(title_font=dict(color=font_color, size=20, family='Arial Black'))
                # Ensure exported HTML uses the same template and colors
                fig.write_html = lambda file, **kwargs: pio.write_html(fig, file, include_plotlyjs=True, full_html=True, config={'displayModeBar': True})
                return fig, "interactive"
            return None, None
        else:
            # Create static plot with matplotlib/seaborn
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Set style
            sns.set_style("whitegrid")
            plt.rcParams['font.size'] = 12
            
            if plot_type == "boxplot":
                sns.boxplot(data=data, x=taxon_column, y=score_column, hue=taxon_column, palette=color_palette, ax=ax, legend=False)
            elif plot_type == "histogram":
                ax.hist(data[score_column].dropna(), bins=20, color='skyblue', edgecolor='black', alpha=0.8)
                ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=14)
                ax.set_ylabel("Frequency", fontsize=14)
                ax.set_title(f"Distribution of {score_column.replace('_', ' ').title()}", fontsize=16, fontweight='bold', pad=20)
                
            elif plot_type == "violin":
                sns.violinplot(data=data, x=taxon_column, y=score_column, hue=taxon_column,
                              palette=color_palette, ax=ax, legend=False)
            elif plot_type == "swarm":
                sns.swarmplot(data=data, x=taxon_column, y=score_column, hue=taxon_column,
                             palette=color_palette, ax=ax, legend=False)
            elif plot_type == "kde":
                # KDE plot for each group
                unique_groups = data[taxon_column].dropna().unique()
                for group in unique_groups:
                    subset = data[data[taxon_column] == group][score_column].dropna()
                    if len(subset) > 1:
                        sns.kdeplot(subset, label=str(group), ax=ax)
                ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=14)
                ax.set_ylabel("Density", fontsize=14)
                ax.set_title(f"KDE Plot of {score_column.replace('_', ' ').title()} by Group", fontsize=16, fontweight='bold', pad=20)
                ax.legend(title=taxon_column.replace('_', ' ').title())
            elif plot_type == "heatmap":
                # Create correlation matrix
                pivot_data = data.pivot_table(values=score_column, 
                                            index=data.index // 10,
                                            columns=taxon_column, 
                                            aggfunc='mean')
                sns.heatmap(pivot_data, annot=True, cmap=color_palette, ax=ax)
            elif plot_type == "barplot":
                # Create bar plot
                group_means = data.groupby(taxon_column)[score_column].mean().sort_values(ascending=False)
                bars = ax.bar(range(len(group_means)), group_means.values, 
                             color=sns.color_palette(color_palette, len(group_means)))
                ax.set_xlabel(taxon_column.replace('_', ' ').title(), fontsize=14)
                ax.set_ylabel(f"Mean {score_column.replace('_', ' ').title()}", fontsize=14)
                ax.set_title(f"Mean {score_column.replace('_', ' ').title()} by {taxon_column.replace('_', ' ').title()}", 
                           fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks(range(len(group_means)))
                ax.set_xticklabels(group_means.index, rotation=45, ha='right')
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, group_means.values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            elif plot_type == "pairplot":
                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                if len(num_cols) < 2:
                    return None, None
                # Seaborn pairplot returns its own fig
                fig = sns.pairplot(data[num_cols])
                plt.tight_layout()
                return fig.fig, "static"
            
            # Set common layout for all plots (except histogram, kde, density, pairplot)
            if plot_type not in ["histogram", "kde", "density", "pairplot"]:
                plt.title("Conservation Score Distribution by Taxonomic Group", 
                         fontsize=16, fontweight='bold', pad=20)
                plt.xlabel(taxon_column.replace('_', ' ').title(), fontsize=14)
                plt.ylabel(score_column.replace('_', ' ').title(), fontsize=14)
                plt.xticks(rotation=45)
            plt.tight_layout()
            return fig, "static"
            
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None, None

def main():
    # Suppress tornado websocket warnings
    import logging
    logging.getLogger('tornado.websocket').setLevel(logging.ERROR)
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    
    # Configure Streamlit page - MUST be first Streamlit command
    st.set_page_config(
        page_title="TaxoConserv - Conservation Score Analysis",
        page_icon="🌿",
        layout="centered",  # Centered layout for better readability
        initial_sidebar_state="expanded"
    )
    
    # Academic-focused welcome header - simplified HTML
    st.markdown("""
<div style="background: linear-gradient(135deg, #6e9c6b, #9fb97f, #a8c98a); color: white; border-radius: 15px; padding: 2rem; margin-bottom: 2rem;">
  <h1 style="font-size: 2.2rem; margin-bottom: 0.8rem; color: white;">🌿 TaxoConserv</h1>
  
  <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem;">
    <strong>Conservation Score Analysis Platform</strong><br>
    Analyze evolutionary conservation scores across taxonomic groups with statistical methods and visualizations.
  </div>
  
  <div style="margin-bottom: 1.5rem;">
    <a href="https://github.com/Bilmem2/TaxoConserv" target="_blank" style="color: white; background: rgba(255,255,255,0.15); padding: 0.6rem 1.2rem; border-radius: 8px; text-decoration: none; margin-right: 1rem;">
      📖 Documentation
    </a>
    <span style="background: rgba(255,255,255,0.1); padding: 0.6rem 1.2rem; border-radius: 8px;">
      v2.1.0
    </span>
  </div>
  
  <div style="background: rgba(255,255,255,0.12); border-radius: 12px; padding: 1.5rem;">
    <div style="font-weight: 600; margin-bottom: 0.8rem;">Quick Start</div>
    <ol style="margin: 0; padding-left: 1.2rem;">
      <li><strong>Upload</strong> conservation data file (CSV/TSV) or load sample dataset</li>
      <li><strong>Configure</strong> analysis by selecting score and grouping columns</li>
      <li><strong>Customize</strong> visualization and statistical options</li>
      <li><strong>Run Analysis</strong> to generate results and plots</li>
    </ol>
  </div>
</div>
""", unsafe_allow_html=True)

    # Apply dynamic layout based on user preference
    layout_mode = st.session_state.get('layout_mode', 'Centered (Optimized)')
    
    if layout_mode == "Wide (Full Screen)":
        st.markdown("""
        <style>
        .block-container {max-width: 95% !important; padding: 1rem 2rem;}
        </style>
        """, unsafe_allow_html=True)
    elif layout_mode == "Compact (Minimal)":
        st.markdown("""
        <style>
        .block-container {max-width: 900px !important; padding: 0.5rem 1rem;}
        .stColumns {gap: 0.5rem;}
        </style>
        """, unsafe_allow_html=True)
    else:  # Centered (Optimized) - Default
        st.markdown("""
        <style>
        .block-container {max-width: 1200px !important; padding: 1rem 1.5rem;}
        </style>
        """, unsafe_allow_html=True)

    # Tab sistemi ekleme - Conservation analysis ve Variant analysis
    tab1, tab2 = st.tabs(["🧬 Taxonomic Conservation Analysis", "🔬 Variant Conservation Analysis"])
    
    with tab1:
        # Mevcut conservation analysis interface
        run_taxonomic_analysis()
    
    with tab2:
        # Yeni variant analysis interface
        run_variant_analysis()

def run_taxonomic_analysis():
    """Taxonomic conservation analysis interface"""
    st.subheader("🧬 Taxonomic Conservation Analysis")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Data Input Section
    st.sidebar.subheader("📁 Data Input")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Conservation Data",
        type=['csv', 'tsv'],
        help="Upload a CSV or TSV file containing conservation scores and taxonomic groups",
        key="main_file_uploader"
    )
    
    # Sample data button
    if st.sidebar.button(
        "🧪 Load Sample Data",
        help="Load built-in conservation dataset for testing (phyloP, GERP, phastCons scores across taxonomic groups)",
        use_container_width=True,
        type="secondary",
        key="taxonomic_sample_data_button"
    ):
        st.session_state['demo_loaded'] = True
        st.rerun()
    
    # Add sample data info
    with st.sidebar.expander("ℹ️ About Sample Data", expanded=False):
        st.markdown("""
        **Taxonomic Conservation Dataset**
        - **Scores**: phyloP, GERP, phastCons, CADD, REVEL
        - **Groups**: Primates, Carnivores, Rodents, Birds, Reptiles, Fish
        - **Genes**: BRCA, TP53, EGFR, MYC, KRAS, etc.
        - **Size**: 200 entries across 6 taxonomic groups
        - **Features**: Realistic conservation patterns, outliers, multiple score types
        - **Use**: Testing & demonstration of taxonomic analysis
        """)
    
    # Add legal/about information
    with st.sidebar.expander("⚖️ Legal & About", expanded=False):
        st.markdown("""
        **TaxoConserv v2.1.0**
        
        **Copyright © 2025 Can Sevilmiş**
        
        Licensed under Apache License 2.0
        
        **DISCLAIMER**: This software is provided "AS IS" without warranty. 
        For research and educational purposes only. Not intended for clinical 
        or diagnostic use. Users are responsible for validating results.
        
        **GitHub**: [TaxoConserv Repository](https://github.com/Bilmem2/TaxoConserv)
        """)
    
    # Reset button with improved functionality
    if st.sidebar.button("🔄 Reset", help="Clear all data, files and restart from scratch", key="taxonomic_reset_button"):
        # Clear all session state completely (including uploaded files)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Clear Streamlit cache
        try:
            st.cache_data.clear()
        except:
            pass
        
        # Force all file uploaders to reset by clearing their specific keys
        file_uploader_keys = ["main_file_uploader", "vcf_file_uploader", "conservation_db_uploader"]
        for file_key in file_uploader_keys:
            if file_key in st.session_state:
                del st.session_state[file_key]
            
        # Show success message and force complete reload
        st.sidebar.success("✅ Application reset completely! All files and data cleared.")
        st.rerun()
    
    # Load data
    data = None
    
    # Demo data loading
    if st.session_state.get('demo_loaded', False):
        try:
            from src.input_parser import create_demo_data
            data = create_demo_data()
            st.sidebar.success("✅ Demo data loaded successfully!")
            
            # Use smart suggestions for demo data
            suggested_group, suggested_score, _, _ = get_column_suggestions(data)
            
            # Set smart defaults or fallback to hardcoded ones
            st.session_state['group_column'] = suggested_group if suggested_group else 'taxon_group'
            st.session_state['score_column'] = suggested_score if suggested_score else 'phyloP_score'
            
        except Exception as e:
            st.error(f"Error loading demo data: {e}")
    
    # File upload handling
    elif uploaded_file is not None:
        try:
            # Reset file pointer to ensure proper reading
            uploaded_file.seek(0)
            
            # Read file based on extension with proper encoding
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file, encoding='utf-8')
            elif uploaded_file.name.endswith('.tsv'):
                data = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
            else:
                st.sidebar.error("❌ Unsupported file type! Only CSV or TSV allowed.")
                return
                
        except UnicodeDecodeError:
            try:
                # Try with different encoding if UTF-8 fails
                uploaded_file.seek(0)
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, encoding='latin-1')
                else:
                    data = pd.read_csv(uploaded_file, sep='\t', encoding='latin-1')
            except Exception as e:
                st.sidebar.error(f"❌ Encoding error: {e}")
                return
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {e}")
            return
            
        try:
            # Check if data was loaded successfully
            if data is not None and not data.empty and len(data.columns) > 0:
                st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                st.session_state['file_loaded'] = True
                
                # Use smart suggestions for uploaded data
                suggested_group, suggested_score, _, _ = get_column_suggestions(data)
                
                # Set smart defaults with fallbacks
                if suggested_group:
                    st.session_state['group_column'] = suggested_group
                elif 'taxon_group' in data.columns:
                    st.session_state['group_column'] = 'taxon_group'
                else:
                    categorical_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
                    if categorical_cols:
                        st.session_state['group_column'] = categorical_cols[0]
                
                if suggested_score:
                    st.session_state['score_column'] = suggested_score
                else:
                    # Fallback to old detection method
                    detected_scores = detect_conservation_scores(data)
                    if detected_scores:
                        num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                        prioritized_scores = prioritize_conservation_scores(num_cols, detected_scores)
                        st.session_state['score_column'] = prioritized_scores[0]
                        st.session_state['detected_conservation_scores'] = detected_scores
                    else:
                        num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                        if num_cols:
                            st.session_state['score_column'] = num_cols[0]
            else:
                st.sidebar.error("❌ File appears to be empty or has no readable columns.")
                data = None
                
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file data: {e}")
            data = None
    
    # Main interface - Welcome or Analysis
    if data is None:
        # Welcome interface
        st.markdown("""
        ### Welcome to Taxonomic Conservation Analysis
        
        This tool analyzes evolutionary conservation scores across different taxonomic groups.
        
        **Getting Started:**
        1. **Upload your data** using the sidebar file uploader, or
        2. **Load sample data** to explore the interface with realistic conservation data
        3. **Configure analysis** parameters in the sidebar
        4. **Run analysis** to generate results and visualizations
        
        **Data Format:**
        Your CSV/TSV file should contain:
        - **Conservation scores** (numeric columns like phyloP, GERP, phastCons, CADD)
        - **Taxonomic groups** (categorical column like taxon_group, family, order)
        - **Gene information** (optional: gene names, chromosomes, positions)
        
        **Example:**
        ```
        gene,chromosome,position,phyloP_score,taxon_group
        BRCA1,chr17,43094464,3.245,Primates
        TP53,chr17,7676040,2.156,Carnivores
        EGFR,chr7,55181378,2.890,Birds
        ```
        """)
        
        # Quick example
        with st.expander("📖 View Sample Data Format", expanded=False):
            sample_data = pd.DataFrame({
                'gene': ['BRCA1', 'TP53', 'EGFR'],
                'chromosome': ['chr17', 'chr17', 'chr7'],
                'position': [43094464, 7676040, 55181378],
                'taxon_group': ['Primates', 'Carnivores', 'Birds'],
                'phyloP_score': [3.245, 2.156, 2.890],
                'GERP_score': [4.820, 3.567, 4.123],
                'phastCons_score': [0.892, 0.734, 0.812]
            })
            st.dataframe(sample_data, use_container_width=True)
    
    # Modern CSS styling with enhanced design
    st.markdown("""
<style>
    /* Main container improvements */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Header gradient styling */
    .main-header {
        background: linear-gradient(135deg, #4CAF50, #45a049, #2E7D32);
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    
    /* Info boxes with modern styling */
    .info-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        color: var(--text-color, #212529);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 1px solid #ffeaa7;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff9f43;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(255, 159, 67, 0.15);
        color: var(--text-color, #856404);
    }
    
    /* Stats cards with subtle elevation */
    .stats-card {
        background: var(--background-color, #ffffff);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: box-shadow 0.2s ease;
        color: var(--text-color, #212529);
    }
    
    .stats-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric containers with improved spacing */
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1.5rem 0;
        gap: 1rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        min-width: 120px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        color: var(--text-color, #495057);
    }
    
    /* Sidebar improvements */
    .sidebar .stSelectbox > label, .sidebar .stCheckbox > label {
        font-weight: 600;
        color: #2E7D32 !important;
        font-size: 0.9rem;
    }
    
    /* Progress bar with modern styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        border-radius: 4px;
    }
    
    /* File uploader with enhanced styling */
    .uploadedFile {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 2.5rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e8f5e8 100%);
        color: var(--text-color, #495057);
        transition: background 0.2s ease;
    }
    
    .uploadedFile:hover {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
    }
    
    /* Button styling improvements */
    .stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
    }
    
    /* Chart container improvements */
    .element-container {
        border-radius: 8px;
    }
    
    /* Data frame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e9ecef;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .metric-container {
            flex-direction: column;
        }
        
        .metric-box {
            margin: 0.5rem 0;
        }
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .info-box {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-left-color: #68d391;
            color: #e2e8f0;
        }
        
        .stats-card {
            background: #2d3748;
            border-color: #4a5568;
            color: #e2e8f0;
        }
        
        .metric-box {
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-color: #4a5568;
            color: #e2e8f0;
        }
    }
</style>
""", unsafe_allow_html=True)

    # All code referencing data, score_column, or group_column is strictly inside this block
    if data is not None:
        # Single unified configuration sidebar with smart suggestions
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Analysis Configuration")
        
        # Get smart column suggestions
        suggested_group, suggested_score, group_confidence, score_confidence = get_column_suggestions(data)
        
        # Group column selection with smart suggestions
        group_options = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        if not group_options:
            group_options = list(data.columns)
        
        # Determine default selection with smart suggestion
        default_group_idx = 0
        if suggested_group and suggested_group in group_options:
            default_group_idx = group_options.index(suggested_group)
        elif st.session_state.get('group_column') and st.session_state.get('group_column') in group_options:
            saved_group = st.session_state.get('group_column')
            if saved_group and saved_group in group_options:
                default_group_idx = group_options.index(saved_group)
        
        # Show suggestion info if available
        if suggested_group and suggested_group in group_confidence:
            confidence = group_confidence[suggested_group]
            confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
            st.sidebar.info(f"{confidence_color} **Suggested**: '{suggested_group}' (confidence: {confidence}%)")
        
        group_column = st.sidebar.selectbox(
            "Grouping Column",
            options=group_options,
            index=default_group_idx,
            help="Select column for taxonomic grouping. Suggestion based on column names and data patterns.",
            key="main_group_column"
        )
        
        # Score column selection with smart suggestions
        score_options = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if score_options:
            # Determine default selection with smart suggestion
            default_score_idx = 0
            if suggested_score and suggested_score in score_options:
                default_score_idx = score_options.index(suggested_score)
            elif st.session_state.get('score_column') and st.session_state.get('score_column') in score_options:
                saved_score = st.session_state.get('score_column')
                if saved_score and saved_score in score_options:
                    default_score_idx = score_options.index(saved_score)
            
            # Show suggestion info if available
            if suggested_score and suggested_score in score_confidence:
                confidence = score_confidence[suggested_score]
                confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
                st.sidebar.info(f"{confidence_color} **Suggested**: '{suggested_score}' (confidence: {confidence}%)")
            
            score_column = st.sidebar.selectbox(
                "Conservation Score Column",
                options=score_options,
                index=default_score_idx,
                help="Select conservation score to analyze. Suggestion based on column names and typical score patterns.",
                key="main_score_column"
            )
        else:
            st.sidebar.error("No numeric columns found for analysis!")
            return
        
        # Show column suggestions summary in an expander
        if group_confidence or score_confidence:
            with st.sidebar.expander("🤖 Smart Suggestions", expanded=False):
                st.markdown("**Column Recommendations:**")
                
                if group_confidence:
                    st.markdown("*Grouping Columns:*")
                    for col, conf in sorted(group_confidence.items(), key=lambda x: x[1], reverse=True)[:3]:
                        confidence_icon = "🟢" if conf >= 70 else "🟡" if conf >= 40 else "🔴"
                        st.markdown(f"• {confidence_icon} `{col}` ({conf}%)")
                
                if score_confidence:
                    st.markdown("*Score Columns:*")
                    for col, conf in sorted(score_confidence.items(), key=lambda x: x[1], reverse=True)[:3]:
                        confidence_icon = "🟢" if conf >= 70 else "🟡" if conf >= 40 else "🔴"
                        st.markdown(f"• {confidence_icon} `{col}` ({conf}%)")
                
                st.caption("🟢 High confidence (≥70%) • 🟡 Medium confidence (40-69%) • 🔴 Low confidence (<40%)")
        
        # Optional Advanced Options
        with st.sidebar.expander("🔬 Advanced Options", expanded=False):
            # Advanced Grouping Options
            hierarchy_input = st.text_input(
                "Hierarchy Columns (comma-separated)",
                value="family,genus,species",
                help="Specify columns for hierarchical grouping, in order of priority"
            )
            hierarchy = [col.strip() for col in hierarchy_input.split(",") if col.strip() in data.columns]
            custom_map_str = st.text_area(
                "Custom Group Mapping (e.g. primate:Mammals, hominid:Mammals)",
                value="primate:Mammals\nhominid:Mammals",
                help="Map synonyms or merge groups. Format: old:new, one per line"
            )
            custom_map = {}
            for line in custom_map_str.splitlines():
                if ":" in line:
                    old, new = line.split(":", 1)
                    custom_map[old.strip().lower()] = new.strip()
        
        # If no advanced options selected, set defaults
        if 'hierarchy' not in locals():
            hierarchy = None
        if 'custom_map' not in locals():
            custom_map = None
        
        # Detect conservation scores for current data
        detected_scores = detect_conservation_scores(data) if data is not None else {}
        
        # ===== PERFORMANCE OPTIMIZATION FOR LARGE DATASETS =====
        # Check dataset size and apply intelligent sampling for statistical tests
        n_rows = len(data)
        n_groups = data[group_column].nunique()
        
        # Performance thresholds
        LARGE_DATASET_THRESHOLD = 10000
        VERY_LARGE_DATASET_THRESHOLD = 50000
        STATISTICAL_SAMPLE_SIZE = 5000  # Maximum sample size for statistical tests
        
        # Create optimized dataset for statistical analysis
        statistical_data = data.copy()
        statistical_sampling_applied = False
        
        if n_rows > LARGE_DATASET_THRESHOLD:
            # For large datasets, use stratified sampling for statistical tests
            sample_size_per_group = min(STATISTICAL_SAMPLE_SIZE // n_groups, n_rows // n_groups)
            
            if sample_size_per_group < n_rows // n_groups:  # Only sample if necessary
                try:
                    # Stratified sampling to maintain group proportions
                    statistical_data = data.groupby(group_column, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), sample_size_per_group)) if len(x) > sample_size_per_group else x
                    ).reset_index(drop=True)
                    statistical_sampling_applied = True
                    
                    # Show sampling info to user
                    if n_rows > VERY_LARGE_DATASET_THRESHOLD:
                        st.sidebar.info(f"📊 **Performance Mode**: Using {len(statistical_data):,} samples (from {n_rows:,}) for statistical tests to ensure fast response times.")
                    
                except Exception as e:
                    # Fallback: simple random sampling
                    statistical_data = data.sample(min(STATISTICAL_SAMPLE_SIZE, n_rows)).reset_index(drop=True)
                    statistical_sampling_applied = True
                    st.sidebar.warning(f"⚠️ Using simple random sampling ({len(statistical_data):,} samples) for performance.")
        
        # Continue with rest of analysis based on selected mode...
        
        # Multi-score analysis expander (if multiple conservation scores detected)
        if len(detected_scores) > 1:
            with st.sidebar.expander("🔬 Multi-Score Analysis", expanded=False):
                # Show compact summary
                st.markdown(f"**{len(detected_scores)} Conservation Scores Detected:**")
                score_list = ', '.join([f"`{col}`" for col in detected_scores.keys()])
                st.markdown(f"{score_list}")
                
                # Quick correlation summary (without heavy visualization)
                conservation_cols = list(detected_scores.keys())
                if len(conservation_cols) > 1:
                    corr_matrix = data[conservation_cols].corr(method='pearson')
                    
                    # Show just the highest and lowest correlations
                    corr_values = []
                    for i in range(len(conservation_cols)):
                        for j in range(i+1, len(conservation_cols)):
                            corr_val = corr_matrix.iloc[i, j]
                            corr_values.append((conservation_cols[i], conservation_cols[j], corr_val))
                    
                    if corr_values:
                        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
                        highest_corr = corr_values[0]
                        st.markdown(f"**Strongest correlation:** {highest_corr[0]} ↔ {highest_corr[1]} (r={highest_corr[2]:.2f})")
                        
                        if len(corr_values) > 1:
                            lowest_corr = corr_values[-1]
                            st.markdown(f"**Weakest correlation:** {lowest_corr[0]} ↔ {lowest_corr[1]} (r={lowest_corr[2]:.2f})")
                    
                    st.info("💡 Enable 'Multi-Score Analysis' in Fine Settings for detailed comparison")
                else:
                    st.info("Only one conservation score detected.")
        # Concise, English summary box
        n_rows = len(data)
        n_cols = len(data.columns)
        n_groups = data[group_column].nunique()
        missing_scores = data[score_column].isnull().sum()
        missing_groups = data[group_column].isnull().sum()
        min_group_size = data[group_column].value_counts().min()
        # Define plot_type and color_palette before using in summary box
        plot_type = st.session_state.get('plot_type', 'boxplot')
        color_palette = st.session_state.get('color_palette', 'Set3')
        st.markdown(f"""
<div class='info-box' style='margin-top:0;'>
<h4>&#128203; Data Summary</h4>
<ul style='list-style-type: none; padding-left: 0;'>
<li><b>Rows:</b> {n_rows:,} {("(Large dataset)" if n_rows > 10000 else "")}</li>
<li><b>Columns:</b> {n_cols}</li>
<li><b>Groups:</b> {n_groups}</li>
<li><b>Score Column:</b> <code>{score_column}</code> ({str(data[score_column].dtype)})</li>
<li><b>Group Column:</b> <code>{group_column}</code> ({str(data[group_column].dtype)})</li>
<li><b>Smallest Group Size:</b> {min_group_size}</li>
<li><b>Missing Scores:</b> {missing_scores}</li>
<li><b>Missing Groups:</b> {missing_groups}</li>
</ul>
<div style='margin-top:0.7em;'>
<b>Note:</b> Missing data or very small groups may affect analysis results.
</div>
</div>
""", unsafe_allow_html=True)
        # Set grouping config for downstream functions
        data.attrs['group_column'] = group_column
        data.attrs['normalize'] = True
        data.attrs['custom_map'] = custom_map if custom_map else None
        data.attrs['hierarchy'] = hierarchy if hierarchy else None

        # Validation warnings in sidebar
        if not pd.api.types.is_numeric_dtype(data[score_column]):
            st.sidebar.error("⚠️ Score column must contain numeric values!")
            return
            
        # Smart data quality warnings
        data_issues = []
        if missing_scores > 0:
            data_issues.append(f"Missing scores: {missing_scores}")
        if missing_groups > 0:
            data_issues.append(f"Missing groups: {missing_groups}")
        if min_group_size < 3:
            small_groups = data[group_column].value_counts()
            small_groups = small_groups[small_groups < 3]
            # Limit display to first 5 small groups to avoid performance issues
            small_groups_list = list(small_groups.items())
            if len(small_groups_list) <= 5:
                data_issues.append(f"Small groups (n<3): {', '.join([f'{g}({n})' for g, n in small_groups_list])}")
            else:
                # Show first 3 groups and count the rest
                first_groups = ', '.join([f'{g}({n})' for g, n in small_groups_list[:3]])
                remaining_count = len(small_groups_list) - 3
                data_issues.append(f"Small groups (n<3): {first_groups} ...and {remaining_count} more")
        if n_groups < 2:
            data_issues.append("Need at least 2 groups for comparison")
            
        if data_issues:
            with st.sidebar.expander("⚠️ Data Quality Issues", expanded=False):
                # Show only the count and most critical issues
                st.warning(f"Found {len(data_issues)} data quality issue(s)")
                if len(data_issues) <= 2:
                    for issue in data_issues:
                        st.caption(f"• {issue}")
                else:
                    # Show only first 2 issues and count the rest
                    for issue in data_issues[:2]:
                        st.caption(f"• {issue}")
                    st.caption(f"• ...and {len(data_issues)-2} more issues")
                st.info("💡 Review data quality before analysis")
        # --- Enhanced automatic plot suggestion logic ---
        def suggest_plot_types(score_column: str) -> tuple[list, str]:
            """Enhanced plot suggestion using advanced mapping system"""
            try:
                plot_info = get_enhanced_plot_info(score_column)
                return plot_info['plots'], plot_info['explanation']
            except:
                # Fallback to original logic if advanced system fails
                col_lower = score_column.lower()
                if any(x in col_lower for x in ["phylop", "phastcons", "conservation"]):
                    return ["boxplot", "violin", "kde"], "These scores are continuous and best visualized with distribution plots."
                elif "gerp" in col_lower:
                    return ["histogram", "barplot"], "Gerp scores are typically summarized with histograms and barplots."
                elif any(x in col_lower for x in ["revel", "cadd"]):
                    return ["swarm", "kde"], "REVEL/CADD scores are suitable for individual and group density visualizations."
                else:
                    return ["boxplot"], "Boxplot is recommended by default for general score visualization."

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Visualization")
        
        # Get recommended plot types and tooltip
        recommended_plots, tooltip_text = suggest_plot_types(score_column)
        
        # Check if additional plot types are enabled (from Fine Settings)
        show_additional_plots = st.session_state.get('show_additional_plots', False)
        
        # Create full plot options list if additional plots are enabled
        all_plot_options = ["boxplot", "violin", "histogram", "swarm", "kde", "barplot", "scatter", "density"]
        
        if show_additional_plots:
            available_plots = all_plot_options
            plot_help_text = "All available plot types"
            st.sidebar.markdown("**Available plot types (all options):** " + ", ".join([f"`{plot}`" for plot in all_plot_options]))
        else:
            available_plots = recommended_plots
            plot_help_text = "Auto-selected based on your data type"
            st.sidebar.markdown("**Recommended plot types (auto-detected):** " + ", ".join([f"`{plot}`" for plot in recommended_plots]))
        
        # Enhanced recommendation explanation
        with st.sidebar.expander("💡 Why these plots?", expanded=False):
            st.markdown(tooltip_text)
            
            if show_additional_plots:
                st.markdown("**Additional Plot Types Enabled:**")
                st.markdown("- **boxplot**: Shows distribution quartiles and outliers")
                st.markdown("- **violin**: Shows distribution shape and density")
                st.markdown("- **histogram**: Shows frequency distribution")
                st.markdown("- **swarm**: Shows individual data points")
                st.markdown("- **kde**: Shows smooth density estimation")
                st.markdown("- **barplot**: Shows group means with error bars")
                st.markdown("- **scatter**: Shows individual points scattered")
                st.markdown("- **density**: Shows overlapping density curves")
            else:
                # Show plot-specific explanations
                try:
                    for plot in recommended_plots:
                        plot_explanation = score_plot_mapper.get_plot_explanation(score_column, plot)
                        st.markdown(f"**{plot.title()}:** {plot_explanation}")
                except:
                    pass

        # Show plot options based on additional plots setting
        plot_type = st.sidebar.selectbox(
            "📈 Plot Type",
            options=available_plots,
            index=0,
            key='plot_type',
            help=plot_help_text
        )
        
        # Analysis button
        st.sidebar.markdown("---")
        analysis_enabled = data is not None and group_column and score_column
        
        # Get show_statistics setting outside of button to make it work immediately
        show_statistics = st.session_state.get('show_statistics', True)
        
        # Show a preview of what the Statistical Results option does
        if not st.session_state.get('main_run_analysis_pressed', False):
            if show_statistics:
                st.info("📊 **Statistical Results Panel is ENABLED**\n\nAfter running analysis, you'll see:\n- Kruskal-Wallis test results\n- Group statistics\n- Post-hoc tests\n- Outlier detection\n- Normality tests")
            else:
                st.info("📈 **Plot-Only Mode is ENABLED**\n\nAfter running analysis, you'll see only the visualization in full width.")
        
        if st.sidebar.button("▶️ Run Analysis", type="primary", disabled=not analysis_enabled, key="main_run_analysis"):
            # Set flag to indicate analysis has been run
            st.session_state['main_run_analysis_pressed'] = True
            
            # ===== UNIFIED DYNAMIC PROGRESS TRACKING SYSTEM =====
            # Create a single, persistent progress container
            main_progress_container = st.container()
            
            with main_progress_container:
                # Create main progress interface
                progress_header = st.empty()
                main_progress_bar = st.progress(0)
                progress_status = st.empty()
                progress_details = st.empty()
                
                # Initialize progress
                progress_header.markdown("### 🚀 **Running Analysis...**")
                progress_status.info("🔧 **Initializing analysis configuration...**")
                progress_details.caption("Setting up parameters and validating input data...")
                main_progress_bar.progress(5)
                time.sleep(0.3)
                
                # Get plot settings from Fine Settings or use defaults
                plot_mode = st.session_state.get('fine_plot_mode', 'Interactive (Plotly)')
                color_palette = st.session_state.get('color_palette', 'Set3')
                chart_dpi = 150  # Fixed DPI value
                interactive_mode = plot_mode == "Interactive (Plotly)"
                
                # Use optimized chart size for better display
                chart_width, chart_height = 8, 6
                
                # Step 1: Data Summary (10%)
                progress_status.info("� **Generating Data Summary...**")
                progress_details.caption(f"Processing {len(data):,} rows with {data[group_column].nunique()} groups...")
                main_progress_bar.progress(10)
                time.sleep(0.2)
                
                # Step 2: Group Statistics (25%)
                progress_status.info("📈 **Calculating Group Statistics...**")
                progress_details.caption("Computing statistical summaries for each group...")
                main_progress_bar.progress(25)
                group_stats = data.groupby(group_column)[score_column].agg(['count', 'mean', 'std', 'min', 'max']).round(6)
                time.sleep(0.3)
                
                # Step 3: Statistical Analysis (45%)
                progress_status.info("🧮 **Performing Statistical Tests...**")
                if statistical_sampling_applied:
                    progress_details.caption(f"Running Kruskal-Wallis test on {len(statistical_data):,} samples for optimal performance...")
                else:
                    progress_details.caption("Running Kruskal-Wallis test and additional statistical tests...")
                main_progress_bar.progress(45)
                try:
                    from src.analysis import perform_statistical_analysis
                    # Use optimized data for statistical tests to improve performance
                    stats_results = perform_statistical_analysis(statistical_data, score_column, group_column)
                    
                    # Add sampling info to results if applied
                    if statistical_sampling_applied and stats_results:
                        stats_results['sampling_applied'] = True
                        stats_results['sample_size'] = len(statistical_data)
                        stats_results['original_size'] = n_rows
                except Exception as e:
                    stats_results = None
                time.sleep(0.4)
                
                # Step 4: Visualization Preparation (65%)
                progress_status.info("🎨 **Preparing Visualization...**")
                progress_details.caption("Setting up chart configuration and data optimization...")
                main_progress_bar.progress(65)
                
                # Performance assessment and intelligent data sampling
                n_rows = len(data)
                n_groups = data[group_column].nunique()
                
                # Define performance thresholds
                LARGE_DATASET_THRESHOLD = 10000
                VERY_LARGE_DATASET_THRESHOLD = 50000
                MAX_PLOT_POINTS = 5000
                
                # Smart data sampling for large datasets
                plot_data = data.copy()
                sampling_applied = False
                performance_warning = False
                skip_visualization = False
                
                time.sleep(0.3)
                
                # Step 5: Handle Large Datasets (75%)
                if n_rows > VERY_LARGE_DATASET_THRESHOLD:
                    progress_status.warning("⚠️ **Large Dataset Processing...**")
                    progress_details.caption(f"Optimizing visualization for {n_rows:,} rows...")
                    main_progress_bar.progress(75)
                    performance_warning = True
                elif n_rows > LARGE_DATASET_THRESHOLD:
                    progress_status.info("📊 **Medium Dataset Processing...**")
                    progress_details.caption(f"Optimizing performance for {n_rows:,} rows...")
                    main_progress_bar.progress(75)
                else:
                    progress_status.info("✅ **Standard Dataset Processing...**")
                    progress_details.caption(f"Processing {n_rows:,} rows with standard optimization...")
                    main_progress_bar.progress(75)
                
                time.sleep(0.2)
                
                # Step 6: Creating Visualization (90%)
                if not skip_visualization:
                    progress_status.info("🎨 **Creating Visualization...**")
                    progress_details.caption(f"Rendering {plot_type} chart...")
                    main_progress_bar.progress(90)
                
                # Step 7: Finalization (100%)
                progress_status.success("✅ **Analysis Complete!**")
                progress_details.caption("All analysis steps completed successfully.")
                main_progress_bar.progress(100)
                time.sleep(0.5)
                
                # Clear progress interface
                progress_header.empty()
                main_progress_bar.empty()
                progress_status.empty()
                progress_details.empty()
            
            # ===== ACTUAL ANALYSIS RESULTS =====
            # Run the actual analysis
            st.subheader("📊 Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(data))
            with col2:
                st.metric("Groups", data[group_column].nunique())
            with col3:
                st.metric("Score Range", f"{data[score_column].min():.2f} - {data[score_column].max():.2f}")
            with col4:
                st.metric("Missing Values", data[score_column].isnull().sum())
            
            # Group statistics 
            st.subheader("📈 Group Statistics")
            st.dataframe(group_stats, use_container_width=True)
            
            # Statistical analysis
            st.subheader("🧮 Statistical Analysis")
            try:
                if stats_results:
                    # Show sampling info if applied
                    if stats_results.get('sampling_applied', False):
                        st.info(f"⚡ **Performance Optimization**: Statistical tests performed on {stats_results.get('sample_size', 'N/A'):,} stratified samples (from {stats_results.get('original_size', 'N/A'):,} total rows) for faster computation.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kruskal-Wallis H", f"{float(stats_results['h_statistic']):.6f}")
                        st.metric("p-value", f"{float(stats_results['p_value']):.8f}")
                    with col2:
                        significance = "✅ Significant" if stats_results['significant'] else "❌ Not Significant"
                        st.metric("Result", significance)
                        st.metric("Groups Compared", stats_results['n_groups'])
                else:
                    st.warning("Statistical analysis could not be performed.")
                        
            except Exception as e:
                st.warning(f"Statistical analysis not available: {e}")
            
            # Visualization with Smart Performance Optimization
            st.subheader("🎨 Visualization")
            
            # Performance assessment and intelligent data sampling
            n_rows = len(data)
            n_groups = data[group_column].nunique()
            
            # Define performance thresholds
            LARGE_DATASET_THRESHOLD = 10000
            VERY_LARGE_DATASET_THRESHOLD = 50000
            MAX_PLOT_POINTS = 5000
            
            # Smart data sampling for large datasets
            plot_data = data.copy()
            sampling_applied = False
            performance_warning = False
            
            if n_rows > VERY_LARGE_DATASET_THRESHOLD:
                # For very large datasets (>50k), show warning and use aggregated plots only
                performance_warning = True
                st.warning(f"""
                ⚠️ **Large Dataset Detected ({n_rows:,} rows)**
                
                For optimal performance with datasets over {VERY_LARGE_DATASET_THRESHOLD:,} rows, we recommend:
                - Using **statistical summaries** instead of individual data points
                - **Aggregated visualizations** (barplot, histogram) perform better
                - Avoid point-based plots (scatter, swarm) which may be slow
                """)
                
                # Ask user preference for large datasets
                plot_choice = st.radio(
                    "Choose visualization approach:",
                    options=[
                        "📊 Aggregated plots only (Recommended)", 
                        "📈 Sample data for faster plotting", 
                        "📋 Statistical summary only (Fastest)",
                        "⚠️ Plot all data (may be slow)"
                    ],
                    index=0,
                    help="Statistical summary shows group comparisons without visualization for maximum speed."
                )
                
                if plot_choice.startswith("📋"):
                    # Statistics-only mode for very large datasets
                    st.info("📋 **Statistics-Only Mode**: Showing detailed statistical analysis without visualization for optimal performance.")
                    
                    # Enhanced statistical summary for large datasets
                    st.markdown("### 📊 Comprehensive Statistical Summary")
                    
                    # Basic group statistics
                    group_stats_basic = data.groupby(group_column)[score_column].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(6)
                    
                    # Add quantiles separately
                    q25_series = data.groupby(group_column)[score_column].quantile(0.25).round(6)
                    q75_series = data.groupby(group_column)[score_column].quantile(0.75).round(6)
                    iqr_series = (q75_series - q25_series).round(6)
                    
                    # Combine all statistics
                    group_stats_basic['q25'] = q25_series
                    group_stats_basic['q75'] = q75_series
                    group_stats_basic['iqr'] = iqr_series
                    
                    st.dataframe(group_stats_basic, use_container_width=True)
                    
                    # Effect size calculation (Cohen's d equivalent for non-parametric)
                    if n_groups == 2:
                        groups = data[group_column].unique()
                        group1_data = data[data[group_column] == groups[0]][score_column].dropna()
                        group2_data = data[data[group_column] == groups[1]][score_column].dropna()
                        
                        # Calculate effect size (r = Z / sqrt(N) for Mann-Whitney U)
                        from scipy.stats import mannwhitneyu
                        try:
                            statistic, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                            z_score = abs(statistic - (len(group1_data) * len(group2_data) / 2)) / np.sqrt(len(group1_data) * len(group2_data) * (len(group1_data) + len(group2_data) + 1) / 12)
                            effect_size = z_score / np.sqrt(len(group1_data) + len(group2_data))
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mann-Whitney U", f"{statistic:.2f}")
                            with col2:
                                st.metric("Effect Size (r)", f"{effect_size:.4f}")
                            with col3:
                                effect_interpretation = "Small" if effect_size < 0.3 else "Medium" if effect_size < 0.5 else "Large"
                                st.metric("Effect Size", effect_interpretation)
                        except Exception as e:
                            st.warning(f"Could not calculate effect size: {e}")
                    
                    # Skip visualization completely
                    skip_visualization = True
                elif plot_choice.startswith("📈"):
                    # Smart stratified sampling to maintain group representation
                    sample_size = min(MAX_PLOT_POINTS, n_rows // 10)
                    plot_data = data.groupby(group_column, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), sample_size // n_groups)) if len(x) > 0 else x
                    ).reset_index(drop=True)
                    sampling_applied = True
                    skip_visualization = False
                    st.info(f"📊 Using stratified sample: {len(plot_data):,} points (from {n_rows:,} total)")
                elif plot_choice.startswith("📊"):
                    # Force aggregated plot types only
                    if plot_type in ["scatter", "swarm", "kde", "density"]:
                        plot_type = "barplot"
                        st.info("🔄 Switched to Bar Plot for better performance with large datasets")
                    skip_visualization = False
                else:
                    # Plot all data (risky)
                    skip_visualization = False
                
            elif n_rows > LARGE_DATASET_THRESHOLD:
                # For large datasets (10k-50k), offer sampling option
                st.info(f"""
                📊 **Medium-Large Dataset ({n_rows:,} rows)**
                
                Some plot types may render slowly. Consider using sampling for better performance.
                """)
                
                skip_visualization = False
                if plot_type in ["scatter", "swarm"] and st.checkbox("🚀 Use data sampling for faster plotting", value=True, key="sampling_checkbox"):
                    sample_size = min(MAX_PLOT_POINTS, n_rows // 5)
                    plot_data = data.groupby(group_column, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), sample_size // n_groups)) if len(x) > 0 else x
                    ).reset_index(drop=True)
                    sampling_applied = True
                    st.success(f"✅ Using sample: {len(plot_data):,} points for visualization")
            else:
                skip_visualization = False
            
            # Only create visualization if not skipped
            if not skip_visualization:
                # Performance timing
                viz_start_time = time.time()
                
                # Show brief info for large datasets
                if len(plot_data) > 5000 or plot_type in ["kde", "density"]:
                    st.info(f"⚠️ Creating {plot_type} chart with {len(plot_data):,} data points - this may take a moment...")
                
                try:
                    # Import required libraries for visualization
                    import plotly.express as px
                    import plotly.graph_objects as go
                    import plotly.io as pio
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Visualization: Interactive vs Static
                    if interactive_mode:
                        st.write("🎨 Creating interactive Plotly visualization...")
                        # Optimized interactive plots with plotly
                        if plot_type == "boxplot":
                            fig = px.box(plot_data, x=group_column, y=score_column, 
                                       color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                       title=f"{score_column} by {group_column}")
                        elif plot_type == "violin":
                            # Optimize violin plots for large datasets
                            if len(plot_data) > 10000:
                                st.info("💡 Large dataset: Using box plot instead of violin for better performance")
                                fig = px.box(plot_data, x=group_column, y=score_column, 
                                           color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                           title=f"{score_column} Distribution by {group_column}")
                            else:
                                fig = px.violin(plot_data, x=group_column, y=score_column, 
                                              color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                              title=f"{score_column} Distribution by {group_column}")
                        elif plot_type == "histogram":
                            # Optimize histogram bins for large datasets
                            optimal_bins = min(50, max(10, int(np.sqrt(len(plot_data)))))
                            fig = px.histogram(plot_data, x=score_column, color=group_column, 
                                             color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                             title=f"{score_column} Histogram by {group_column}",
                                             nbins=optimal_bins)
                        elif plot_type == "swarm":
                            # Limit swarm plot size and use strip plot for large data
                            if len(plot_data) > 2000:
                                st.info("💡 Large dataset: Using strip plot instead of swarm for better performance")
                            fig = px.strip(plot_data, x=group_column, y=score_column, 
                                         color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                         title=f"{score_column} by {group_column}")
                        elif plot_type == "kde":
                            # Optimized KDE with reduced resolution for large datasets
                            fig = go.Figure()
                            colors = getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3)
                            from scipy.stats import gaussian_kde
                            
                            # Reduce KDE resolution for large datasets
                            kde_resolution = 100 if len(plot_data) < 5000 else 50
                            
                            for i, group in enumerate(plot_data[group_column].unique()):
                                group_data = plot_data[plot_data[group_column] == group][score_column].dropna()
                                if len(group_data) > 1:
                                    # Sample for KDE if too many points
                                    if len(group_data) > 1000:
                                        group_data = group_data.sample(1000)
                                    
                                    # Calculate KDE with reduced resolution
                                    kde = gaussian_kde(group_data)
                                    x_range = np.linspace(group_data.min(), group_data.max(), kde_resolution)
                                    y_kde = kde(x_range)
                                    
                                    fig.add_trace(go.Scatter(
                                        x=x_range,
                                        y=y_kde,
                                        mode='lines',
                                        name=str(group),
                                        line=dict(color=colors[i % len(colors)], width=3),
                                        fill='tonexty' if i > 0 else 'tozeroy',
                                        opacity=0.6
                                    ))
                            
                            fig.update_layout(
                                title=f"{score_column} KDE (Kernel Density Estimation) by {group_column}",
                                xaxis_title=score_column,
                                yaxis_title="Density"
                            )
                        elif plot_type == "barplot":
                            # Efficient aggregated bar plot (always fast)
                            group_means = plot_data.groupby(group_column)[score_column].agg(['mean', 'std']).reset_index()
                            fig = px.bar(group_means, x=group_column, y='mean', 
                                       error_y='std',
                                       color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                       title=f"{score_column} Mean by {group_column}")
                        elif plot_type == "scatter":
                            # Optimized scatter with opacity for overlapping points
                            point_opacity = max(0.1, min(1.0, 500 / len(plot_data)))
                            fig = px.scatter(plot_data, x=group_column, y=score_column, 
                                           color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                           title=f"{score_column} Scatter by {group_column}",
                                           opacity=point_opacity)
                        elif plot_type == "density":
                            # Optimized density with reduced bins
                            fig = go.Figure()
                            colors = getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3)
                            optimal_bins = min(30, max(10, int(np.sqrt(len(plot_data) / n_groups))))
                            
                            for i, group in enumerate(plot_data[group_column].unique()):
                                group_data = plot_data[plot_data[group_column] == group][score_column].dropna()
                                if len(group_data) > 1:
                                    fig.add_trace(go.Histogram(
                                        x=group_data, 
                                        name=str(group), 
                                        histnorm='probability density',
                                        opacity=0.6,
                                        marker_color=colors[i % len(colors)],
                                        nbinsx=optimal_bins
                                    ))
                            fig.update_layout(title=f"{score_column} Density Distribution by {group_column}", barmode='overlay')
                        else:
                            fig = px.scatter(plot_data, x=group_column, y=score_column, 
                                           color=group_column, color_discrete_sequence=getattr(px.colors.qualitative, color_palette, px.colors.qualitative.Set3),
                                           title=f"{score_column} by {group_column}")
                        
                        # Update layout with optimal chart size
                        fig.update_layout(
                            xaxis_title=group_column, 
                            yaxis_title=score_column,
                            width=chart_width * 80,  # Optimized for better display
                            height=chart_height * 80,
                            title_font_size=14,
                            font_size=11
                        )
                        
                    else:
                        # Import matplotlib and seaborn for static plots
                        plt.style.use('default')
                        fig_plt, ax = plt.subplots(figsize=(chart_width, chart_height), dpi=chart_dpi)
                        
                        if plot_type == "boxplot":
                            sns.boxplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, legend=False)
                        elif plot_type == "violin":
                            if len(plot_data) > 10000:
                                st.info("💡 Large dataset: Using box plot instead of violin for better performance")
                                sns.boxplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, legend=False)
                            else:
                                sns.violinplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, legend=False)
                        elif plot_type == "histogram":
                            optimal_bins = min(50, max(10, int(np.sqrt(len(plot_data)))))
                            try:
                                colors = plt.cm.get_cmap(color_palette)
                            except:
                                colors = plt.cm.get_cmap('Set3')  # Fallback colormap
                            for i, group in enumerate(plot_data[group_column].unique()):
                                group_data = plot_data[plot_data[group_column] == group][score_column].dropna()
                                if len(group_data) > 0:
                                    color_idx = i / max(1, len(plot_data[group_column].unique()) - 1)
                                    ax.hist(group_data, alpha=0.7, label=str(group), bins=optimal_bins, 
                                           color=colors(color_idx))
                            ax.legend()
                            ax.set_xlabel(score_column)
                            ax.set_ylabel("Frequency")
                        elif plot_type == "swarm":
                            # Use stripplot for large datasets
                            sns.stripplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, alpha=0.7, legend=False)
                        elif plot_type == "kde":
                            for group in plot_data[group_column].unique():
                                group_data = plot_data[plot_data[group_column] == group][score_column].dropna()
                                if len(group_data) > 1000:
                                    group_data = group_data.sample(1000)  # Sample for performance
                                if len(group_data) > 1:
                                    sns.kdeplot(group_data, label=group, ax=ax)
                            ax.legend()
                        elif plot_type == "barplot":
                            sns.barplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, errorbar='sd', legend=False)
                        elif plot_type == "scatter":
                            point_alpha = max(0.1, min(1.0, 500 / len(plot_data)))
                            sns.stripplot(data=plot_data, x=group_column, y=score_column, hue=group_column, ax=ax, palette=color_palette, size=8, alpha=point_alpha, legend=False)
                        elif plot_type == "density":
                            for group in plot_data[group_column].unique():
                                group_data = plot_data[plot_data[group_column] == group][score_column].dropna()
                                if len(group_data) > 1:
                                    sns.histplot(group_data, kde=True, stat="density", alpha=0.6, label=str(group), ax=ax, bins='auto')
                            ax.legend()
                        
                        ax.set_title(f"{score_column} by {group_column}")
                        ax.set_xlabel(group_column)
                        ax.set_ylabel(score_column)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    
                    # Performance summary
                    viz_time = time.time() - viz_start_time
                    if sampling_applied:
                        st.success(f"⚡ Visualization completed in {viz_time:.2f}s using {len(plot_data):,} sample points (from {n_rows:,} total)")
                    elif viz_time > 2:
                        st.info(f"🕐 Visualization completed in {viz_time:.2f}s")
                    else:
                        st.success(f"✅ Visualization completed successfully in {viz_time:.2f}s")
                    
                except Exception as viz_error:
                    st.error(f"❌ Visualization error: {viz_error}")
                    st.info("💡 Try using a different plot type or enable data sampling for large datasets.")
                    
                    # Fallback: Offer simple statistical summary instead
                    if len(data) > LARGE_DATASET_THRESHOLD:
                        st.markdown("### 📊 Statistical Summary (Fallback)")
                        summary_stats = data.groupby(group_column)[score_column].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
                        st.dataframe(summary_stats, use_container_width=True)
            
            # Display the final visualization
            if 'fig' in locals() and interactive_mode:
                st.plotly_chart(fig, use_container_width=True)
            elif 'fig_plt' in locals() and not interactive_mode:
                st.pyplot(fig_plt, use_container_width=True)
            
            # Performance tips for future use
            if performance_warning and not sampling_applied:
                with st.expander("💡 Performance Tips for Large Datasets", expanded=False):
                    st.markdown("""
                    **For faster visualization with large datasets:**
                    
                    1. **📊 Use aggregated plots**: Bar plots, histograms are always fast
                    2. **🎯 Enable data sampling**: Maintains statistical patterns while improving speed
                    3. **⚡ Avoid point-based plots**: Scatter and swarm plots slow down with many points
                    4. **🔢 Consider statistical summaries**: Group statistics often more informative than individual points
                    5. **📈 Interactive mode**: Plotly generally handles large data better than matplotlib
                    """)
            
            # Detailed Statistical Results Panel (if enabled)
            if show_statistics and stats_results:
                st.markdown("---")
                st.markdown("""
                <div style='background: linear-gradient(90deg, #1f77b4, #ff7f0e); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h2 style='color: white; margin: 0; text-align: center;'>📊 Detailed Statistical Analysis</h2>
                    <p style='color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;'>Comprehensive statistical insights and tests</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Use columns layout for better organization
                col1, col2 = st.columns([0.6, 0.4], gap="large")
                
                with col1:
                    # Section 1: Kruskal-Wallis Details
                    st.markdown("### 🧮 Kruskal-Wallis Test Results")
                    
                    # Create a nice info box for the main results
                    significance_color = "success" if stats_results['significant'] else "warning"
                    significance_icon = "✅" if stats_results['significant'] else "❌"
                    significance_text = "Significant difference detected!" if stats_results['significant'] else "No significant difference found."
                    
                    # Adaptive styling for dark/light theme
                    st.markdown(f"""
                    <div style='
                        background-color: var(--background-color, #f0f2f6); 
                        padding: 1rem; 
                        border-radius: 8px; 
                        border-left: 4px solid #1f77b4;
                        border: 1px solid var(--border-color, #e0e0e0);
                    '>
                        <h4 style='margin: 0 0 0.5rem 0; color: #1f77b4;'>{significance_icon} Test Results</h4>
                        <p style='margin: 0; color: var(--text-color, #262730);'><strong>H-statistic:</strong> {float(stats_results['h_statistic']):.6f}</p>
                        <p style='margin: 0; color: var(--text-color, #262730);'><strong>p-value:</strong> {float(stats_results['p_value']):.8f}</p>
                        <p style='margin: 0; color: var(--text-color, #262730);'><strong>Degrees of freedom:</strong> {stats_results['n_groups'] - 1}</p>
                        {'<p style="margin: 0; color: var(--text-color, #262730);"><strong>Sample size:</strong> ' + f"{stats_results.get('sample_size', 'N/A'):,} (stratified from {stats_results.get('original_size', 'N/A'):,})" + '</p>' if stats_results.get('sampling_applied', False) else ''}
                        <p style='margin: 0.5rem 0 0 0; font-weight: bold; color: var(--text-color, #262730);'>{significance_text}</p>
                    </div>
                    
                    <style>
                    /* Dark theme adaptation */
                    @media (prefers-color-scheme: dark) {{
                        :root {{
                            --background-color: #2d3142;
                            --text-color: #ffffff;
                            --border-color: #4a4e69;
                        }}
                    }}
                    /* Light theme (default) */
                    @media (prefers-color-scheme: light) {{
                        :root {{
                            --background-color: #f0f2f6;
                            --text-color: #262730;
                            --border-color: #e0e0e0;
                        }}
                    }}
                    /* Streamlit dark theme detection */
                    .stApp[data-theme="dark"] {{
                        --background-color: #2d3142;
                        --text-color: #ffffff;
                        --border-color: #4a4e69;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    if stats_results['significant']:
                        st.info("💡 **Interpretation:** There are statistically significant differences between the groups (p < 0.05).")
                    else:
                        st.info("💡 **Interpretation:** The groups do not show statistically significant differences (p ≥ 0.05).")
                    
                    # Section 2: Post-hoc Dunn Test
                    st.markdown("### 🔬 Post-hoc Analysis")
                    with st.expander("Pairwise Group Comparisons (Dunn Test)", expanded=False):
                        # Use statistical_data for post-hoc analysis to maintain performance
                        analysis_data = statistical_data if statistical_sampling_applied else data
                        
                        if statistical_sampling_applied:
                            st.info(f"⚡ Post-hoc analysis performed on {len(analysis_data):,} stratified samples for optimal performance.")
                        
                        try:
                            import scikit_posthocs as sp # type: ignore
                            dunn_results = sp.posthoc_dunn(analysis_data, val_col=score_column, group_col=group_column, p_adjust='bonferroni')
                            # Format the results for better display
                            dunn_formatted = dunn_results.round(8)  # Round to 8 decimal places
                            # Replace very small values with scientific notation (using map instead of applymap)
                            dunn_formatted = dunn_formatted.map(lambda x: f"{x:.2e}" if x < 0.001 and x != 0 else f"{x:.6f}")
                            st.dataframe(dunn_formatted, use_container_width=True)
                            st.caption("📝 Values show p-values for pairwise comparisons (Bonferroni corrected). Lower values indicate stronger evidence of difference.")
                        except ImportError:
                            st.warning("⚠️ **scikit-posthocs** package is not installed.")
                            st.info("""
                            **To install the optional package for advanced post-hoc tests:**
                            ```bash
                            pip install scikit-posthocs
                            ```
                            
                            **Alternative:** You can perform manual pairwise comparisons using built-in methods:
                            """)
                            
                            # Manual pairwise comparison fallback
                            st.markdown("**Manual Pairwise Comparisons (Mann-Whitney U):**")
                            from scipy.stats import mannwhitneyu
                            from itertools import combinations
                            
                            groups = analysis_data[group_column].unique()
                            pairwise_results = []
                            
                            for group1, group2 in combinations(groups, 2):
                                if pd.notna(group1) and pd.notna(group2):
                                    data1 = analysis_data[analysis_data[group_column] == group1][score_column].dropna()
                                    data2 = analysis_data[analysis_data[group_column] == group2][score_column].dropna()
                                    
                                    if len(data1) > 0 and len(data2) > 0:
                                        try:
                                            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                                            pairwise_results.append({
                                                'Group 1': str(group1),
                                                'Group 2': str(group2),
                                                'Mann-Whitney U': f"{statistic:.2f}",
                                                'p-value': f"{p_value:.6f}",
                                                'Significant (p<0.05)': "✅" if p_value < 0.05 else "❌"
                                            })
                                        except Exception as e:
                                            st.warning(f"Could not compare {group1} vs {group2}: {e}")
                            
                            if pairwise_results:
                                pairwise_df = pd.DataFrame(pairwise_results)
                                st.dataframe(pairwise_df, use_container_width=True)
                                st.caption("📝 Manual pairwise comparisons using Mann-Whitney U test (no multiple comparison correction).")
                            else:
                                st.warning("Could not perform pairwise comparisons.")
                        except Exception as e:
                            st.error(f"Error running post-hoc test: {e}")
                            st.info("💡 Try installing scikit-posthocs: `pip install scikit-posthocs`")
                
                with col2:
                    # Section 3: Outlier Detection
                    st.markdown("### 🔎 Data Quality Checks")
                    
                    with st.expander("Outlier Detection (IQR Method)", expanded=True):
                        # Use appropriate data for outlier analysis
                        outlier_analysis_data = statistical_data if statistical_sampling_applied and len(statistical_data) > 1000 else data
                        
                        if statistical_sampling_applied and outlier_analysis_data is statistical_data:
                            st.caption(f"⚡ Outlier analysis on {len(outlier_analysis_data):,} samples for performance.")
                        
                        outlier_info = {}
                        for group in outlier_analysis_data[group_column].unique():
                            if pd.notna(group):
                                vals = outlier_analysis_data[outlier_analysis_data[group_column] == group][score_column].dropna()
                                q1 = vals.quantile(0.25)
                                q3 = vals.quantile(0.75)
                                iqr = q3 - q1
                                lower = q1 - 1.5 * iqr
                                upper = q3 + 1.5 * iqr
                                outliers = vals[(vals < lower) | (vals > upper)]
                                outlier_info[group] = len(outliers)
                        
                        outlier_df = pd.DataFrame.from_dict(outlier_info, orient='index', columns=['Outlier Count'])
                        outlier_df.index.name = 'Group'
                        st.dataframe(outlier_df, use_container_width=True)
                        
                        total_outliers = sum(outlier_info.values())
                        if total_outliers > 0:
                            st.warning(f"⚠️ Found **{total_outliers}** outliers across all groups")
                        else:
                            st.success("✅ No outliers detected in any group")
                    
                    # Section 4: Normality Test
                    with st.expander("Normality Test (Shapiro-Wilk)", expanded=True):
                        from scipy.stats import shapiro
                        
                        # Use appropriate data for normality testing (limit sample size for performance)
                        normality_analysis_data = statistical_data if statistical_sampling_applied else data
                        
                        if statistical_sampling_applied:
                            st.caption(f"⚡ Normality test on {len(normality_analysis_data):,} samples for performance.")
                        
                        normality_results = {}
                        for group in normality_analysis_data[group_column].unique():
                            if pd.notna(group):
                                vals = normality_analysis_data[normality_analysis_data[group_column] == group][score_column].dropna()
                                if len(vals) >= 3:
                                    # For very large groups, sample for Shapiro-Wilk (which has limits)
                                    if len(vals) > 5000:
                                        vals = vals.sample(5000)
                                    stat, pval = shapiro(vals)
                                    is_normal = "Yes" if pval > 0.05 else "No"
                                    normality_results[group] = {
                                        'W-statistic': f"{stat:.6f}",
                                        'p-value': f"{pval:.8f}",
                                        'Normal?': is_normal
                                    }
                                else:
                                    normality_results[group] = {
                                        'W-statistic': 'N/A',
                                        'p-value': 'N/A',
                                        'Normal?': 'Too few samples'
                                    }
                        
                        normality_df = pd.DataFrame(normality_results).T
                        st.dataframe(normality_df, use_container_width=True)
                        st.caption("📝 p > 0.05 suggests data follows normal distribution")
            
            # Enhanced Data Export Section
            st.markdown("---")
            st.markdown("""
            <div style='background: linear-gradient(90deg, #28a745, #20c997); padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                <h2 style='color: white; margin: 0; text-align: center;'>💾 Export Results & Data</h2>
                <p style='color: white; margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;'>Download analysis results in multiple formats</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Local imports for export functionality  
            from datetime import datetime
            import json
            
            # Create export tabs for better organization
            export_tab1, export_tab2, export_tab3 = st.tabs(["📊 Analysis Results", "📈 Plots & Visualizations", "📋 Raw Data"])
            
            with export_tab1:
                if stats_results:
                    st.markdown("### Analysis Results Export")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Main Statistical Results**")
                        # Prepare main results data with proper formatting and JSON serialization
                        main_results = {
                            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'score_column': score_column,
                            'group_column': group_column,
                            'h_statistic': round(float(stats_results['h_statistic']), 6),
                            'p_value': round(float(stats_results['p_value']), 8),
                            'significant': bool(stats_results['significant']),
                            'n_groups': int(stats_results['n_groups']),
                            'degrees_of_freedom': int(stats_results['n_groups'] - 1)
                        }
                        
                        # CSV export
                        results_df = pd.DataFrame([main_results])
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📄 Download as CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"taxoconserv_main_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="main_results_csv"
                        )
                        
                        # JSON export with proper serialization
                        json_buffer = io.StringIO()
                        import json
                        json.dump(main_results, json_buffer, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📄 Download as JSON",
                            data=json_buffer.getvalue(),
                            file_name=f"taxoconserv_main_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            key="main_results_json"
                        )
                    
                    with col2:
                        st.markdown("**Group Statistics**")
                        if 'stats_summary' in stats_results:
                            group_stats = stats_results['stats_summary']
                            
                            # CSV export for group stats
                            group_csv = io.StringIO()
                            group_stats.to_csv(group_csv)
                            st.download_button(
                                label="📊 Download Group Stats CSV",
                                data=group_csv.getvalue(),
                                file_name=f"group_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                key="group_stats_csv"
                            )
                            
                            # JSON export for group stats
                            group_json = group_stats.reset_index().to_json(orient="records", indent=2)
                            st.download_button(
                                label="📊 Download Group Stats JSON",
                                data=group_json,
                                file_name=f"group_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                key="group_stats_json"
                            )
                else:
                    st.info("Run analysis first to export statistical results.")
            
            with export_tab2:
                st.markdown("### Plot & Visualization Export")
                
                if 'fig' in locals() and fig is not None:
                    plot_mode = st.session_state.get('fine_plot_mode', 'Interactive (Plotly)')
                    interactive_mode = plot_mode == "Interactive (Plotly)"
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Interactive Plots (Plotly)**")
                        if interactive_mode:
                            # HTML export for interactive plots
                            try:
                                import plotly.io as pio
                                html_buffer = io.StringIO()
                                pio.write_html(fig, file=html_buffer, auto_open=False)
                                html_str = html_buffer.getvalue()
                                
                                st.download_button(
                                    label="🌐 Download Interactive Plot (HTML)",
                                    data=html_str.encode('utf-8'),
                                    file_name=f"conservation_plot_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html",
                                    key="plot_html"
                                )
                                st.caption("✨ HTML format preserves interactivity")
                            except Exception as e:
                                st.warning(f"Interactive plot export not available: {e}")
                        else:
                            st.info("Switch to Interactive mode in Fine Settings to export HTML plots")
                    
                    with col2:
                        st.markdown("**Static Plots (Publication Quality)**")
                        if not interactive_mode:
                            from matplotlib.figure import Figure as MplFigure
                            if isinstance(fig, MplFigure):
                                # PNG export
                                try:
                                    buf_png = io.BytesIO()
                                    fig.savefig(buf_png, format="png", bbox_inches='tight', dpi=300)
                                    buf_png.seek(0)
                                    
                                    st.download_button(
                                        label="🖼️ Download as PNG (High Quality)",
                                        data=buf_png,
                                        file_name=f"conservation_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png",
                                        key="plot_png"
                                    )
                                except Exception as e:
                                    st.warning(f"PNG export failed: {e}")
                                
                                # SVG export
                                try:
                                    buf_svg = io.BytesIO()
                                    fig.savefig(buf_svg, format="svg", bbox_inches='tight')
                                    buf_svg.seek(0)
                                    
                                    st.download_button(
                                        label="📐 Download as SVG (Vector)",
                                        data=buf_svg,
                                        file_name=f"conservation_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                        mime="image/svg+xml",
                                        key="plot_svg"
                                    )
                                    st.caption("📐 SVG format is perfect for publications")
                                except Exception as e:
                                    st.warning(f"SVG export failed: {e}")
                            else:
                                st.info("Static plot export only available in Matplotlib mode")
                        else:
                            st.info("Switch to Static mode in Fine Settings to export PNG/SVG")
                else:
                    st.info("Run analysis first to export plots.")
            
            with export_tab3:
                st.markdown("### Raw Data Export")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Dataset**")
                    if data is not None:
                        # Full dataset CSV
                        csv_data = data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Full Dataset (CSV)",
                            data=csv_data,
                            file_name=f"conservation_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="full_data_csv"
                        )
                        
                        # Excel export
                        try:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                data.to_excel(writer, sheet_name='Data', index=False)
                                if stats_results and 'stats_summary' in stats_results:
                                    stats_results['stats_summary'].to_excel(writer, sheet_name='Group_Stats')
                            
                            st.download_button(
                                label="📊 Download as Excel (XLSX)",
                                data=excel_buffer.getvalue(),
                                file_name=f"conservation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="full_data_excel"
                            )
                        except ImportError:
                            st.info("Install openpyxl for Excel export: `pip install openpyxl`")
                        except Exception as e:
                            st.warning(f"Excel export failed: {e}")
                    else:
                        st.info("No data loaded to export.")
                
                with col2:
                    st.markdown("**Data Summary**")
                    if data is not None:
                        st.write(f"**Rows:** {len(data):,}")
                        st.write(f"**Columns:** {len(data.columns)}")
                        st.write(f"**Groups:** {data[group_column].nunique()}")
                        st.write(f"**File size (estimated):** {len(csv_data.encode('utf-8')) / 1024:.1f} KB")
                        
                        # Data type summary
                        st.markdown("**Column Types:**")
                        for col, dtype in data.dtypes.items():
                            st.write(f"- `{col}`: {dtype}")
                    else:
                        st.info("Load data to see summary.")
        
        # Simplified fine settings - moved here for better organization
        with st.sidebar.expander("⚙️ Fine Settings", expanded=False):
            # Color palette selection (moved from main section)
            palette_options = [
                "Set1", "Set2", "Set3", "tab10", "Dark2", "Pastel1", "Accent",
                "viridis", "plasma", "magma", "inferno", "cividis", "colorblind"
            ]
            
            # Get current color palette selection with key for session state
            current_palette = st.session_state.get('color_palette', 'Set3')
            palette_index = palette_options.index(current_palette) if current_palette in palette_options else 2  # Set3 is at index 2
            
            color_palette = st.selectbox(
                "🎨 Color Palette",
                options=palette_options,
                index=palette_index,
                key='color_palette',
                help="Select color palette for visualizations"
            )

            # Palette preview
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import seaborn as sns
            from io import BytesIO
            import base64
            def get_palette_colors(palette_name, n_colors=8):
                try:
                    return sns.color_palette(palette_name, n_colors)
                except Exception:
                    return sns.color_palette("Set3", n_colors)
            preview_colors = get_palette_colors(color_palette, 8)
            fig_palette, ax_palette = plt.subplots(figsize=(4, 0.5))
            for i, color in enumerate(preview_colors):
                ax_palette.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color))
            ax_palette.set_xlim(0, 8)
            ax_palette.set_ylim(0, 1)
            ax_palette.axis('off')
            buf = BytesIO()
            fig_palette.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close(fig_palette)
            palette_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            st.markdown(f"<b>Palette Preview:</b><br><img src='data:image/png;base64,{palette_img}' style='width:100%;height:30px;border-radius:6px;'>", unsafe_allow_html=True)
            
            # Additional visualization settings
            show_additional_plots = st.checkbox(
                "Additional Plot Types",
                value=False,
                key='show_additional_plots',
                help="Enable all plot types (boxplot, violin, histogram, swarm, kde, barplot, scatter, density)"
            )
            
            # Plot rendering mode selection
            plot_mode = st.radio(
                "🖥️ Plot Rendering Mode",
                options=["Interactive (Plotly)", "Static (Matplotlib)"],
                index=0,
                key='fine_plot_mode',
                help="Interactive: Web-based plots with zoom/hover\nStatic: Traditional publication-ready plots"
            )
            
            # Layout mode selection
            layout_mode = st.radio(
                "📐 Page Layout",
                options=["Centered (Optimized)", "Wide (Full Screen)", "Compact (Minimal)"],
                index=0,
                key='layout_mode',
                help="Centered: Best for readability\nWide: Uses full screen width\nCompact: Minimal spacing"
            )
            
            # Layout will be applied at the page level
            if layout_mode != st.session_state.get('layout_mode', 'Centered (Optimized)'):
                st.rerun()  # Rerun to apply new layout
            
            # Statistical display options
            show_statistics = st.checkbox(
                "📊 Show Statistical Results",
                value=True,
                key='show_statistics',
                help="Show/hide the statistical analysis panel (Kruskal-Wallis test, group statistics, post-hoc tests, etc.)"
            )
            
            # Advanced analysis options
            st.markdown("**🔬 Advanced Analysis:**")
            use_multi_score = st.checkbox(
                "Multi-Score Analysis",
                value=False,
                key='use_multi_score',
                help="Compare multiple conservation scores (PhyloP, GERP, etc.)"
            )
        
        # Apply advanced grouping - simplified to basic filtering only
        processed_data = data.copy()
        final_group_column = group_column
        
        # Multi-score analysis
        if st.session_state.get('use_multi_score', False):
            st.markdown("---")
            multi_score_results = create_multi_score_analysis_ui(data, group_column)
            
            # If multi-score analysis was performed, we can show additional insights
            if multi_score_results:
                st.session_state['multi_score_results'] = multi_score_results
            
            st.markdown("---")
        
        # Apply advanced grouping - simplified to basic filtering only
            # Defensive: check again if columns exist
            if score_column not in data.columns or group_column not in data.columns:
                st.error("❌ Selected columns not found in data!")
            elif not pd.api.types.is_numeric_dtype(data[score_column]):
                st.error("❌ Score column must contain numeric values!")
                st.error("❌ Score column must contain numeric values!")
            elif data[score_column].isnull().any():
                st.error("❌ Score column contains missing values!")
            elif data[group_column].isnull().any():
                st.error("❌ Group column contains missing values!")
            else:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                # Step 1: Data validation
                status_text.text("🔍 Validating data...")
                progress_bar.progress(20)
                # Step 2: Statistical analysis
                status_text.text("🧮 Performing statistical analysis...")
                progress_bar.progress(40)
                
                # Use cached statistical analysis for better performance
                if MODULES_AVAILABLE:
                    start_time = time.time()
                    data_hash = cache.get_cache_key(data)
                    st.session_state['current_data'] = data  # Store for cache access
                    analysis_results = cached_statistical_analysis(data_hash, score_column, group_column)
                    analysis_time = time.time() - start_time
                    try:
                        performance_monitor.record_metric('analysis_time', analysis_time)
                    except Exception:
                        pass  # Ignore performance monitoring errors
                else:
                    # Fallback without caching
                    analysis_results = perform_statistical_analysis(data, score_column, group_column)
                if analysis_results is None:
                    st.error("❌ Statistical analysis failed!")
                else:
                    # Step 3: Generate visualization
                    status_text.text("🎨 Creating visualization...")
                    progress_bar.progress(70)
                    
                    progress_bar.progress(100)
                    status_text.text("")
                    # --- Results layout ---
                    # Use full-width layout for results section
                    st.markdown("""
                        <style>
                        .block-container {max-width: 1800px !important; padding-left: 2rem; padding-right: 2rem;}
                        .stColumn {padding: 0 1.5rem;}
                        .main-header {margin-left: auto; margin-right: auto;}
                        </style>
                    """, unsafe_allow_html=True)
                    if show_statistics:
                        col1, col2 = st.columns([0.7, 1.3], gap="large")
                        with col1:
                            info_expander = st.expander("Show/Hide Statistical Results", expanded=True)
                            with info_expander:
                                # Section 1: Kruskal-Wallis
                                with st.expander("🧮 Kruskal-Wallis Test", expanded=True):
                                    st.write(f"**H-statistic:** {analysis_results['h_statistic']:.4f}")
                                    st.write(f"**p-value:** {analysis_results['p_value']:.6f}")
                                    st.write(f"**Groups:** {analysis_results['n_groups']}")
                                    if analysis_results['significant']:
                                        st.success("Significant difference detected!")
                                    else:
                                        st.warning("No significant difference found.")
                                # Section 2: Group Statistics
                                with st.expander("📋 Group Statistics", expanded=False):
                                    stats_df = analysis_results['stats_summary']
                                    stats_df.index.name = 'Taxon Group'
                                    st.dataframe(stats_df, use_container_width=True)
                                # Section 3: Post-hoc Dunn Test
                                with st.expander("🔬 Post-hoc Dunn Test", expanded=False):
                                    try:
                                        import scikit_posthocs as sp # type: ignore
                                        dunn_results = sp.posthoc_dunn(data, val_col=score_column, group_col=group_column, p_adjust='bonferroni')
                                        # Format the results for better display
                                        dunn_formatted = dunn_results.round(8)  # Round to 8 decimal places
                                        # Replace very small values with scientific notation (using map instead of applymap)
                                        dunn_formatted = dunn_formatted.map(lambda x: f"{x:.2e}" if x < 0.001 and x != 0 else f"{x:.6f}")
                                        st.dataframe(dunn_formatted, use_container_width=True)
                                    except ImportError:
                                        st.warning("scikit-posthocs package is not installed. To use the post-hoc Dunn test, run 'pip install scikit-posthocs'.")
                                # Section 4: Outlier Detection
                                with st.expander("🔎 Outlier Detection (IQR method)", expanded=False):
                                    # Use fast outlier detection for large datasets
                                    if MODULES_AVAILABLE and len(data) > 1000:
                                        outlier_table = get_fast_outlier_detection(data, score_column, group_column)
                                        if not outlier_table.empty:
                                            st.dataframe(outlier_table, use_container_width=True)
                                        else:
                                            st.info("No outliers detected in any group.")
                                    else:
                                        # Original pandas-based outlier detection
                                        outlier_info = {}
                                        outlier_table = None
                                        stats_df = analysis_results.get('stats_summary')
                                        if stats_df is not None and not stats_df.empty:
                                            for group in stats_df.index:
                                                vals = data[data[group_column] == group][score_column].dropna()
                                                q1 = vals.quantile(0.25)
                                                q3 = vals.quantile(0.75)
                                                iqr = q3 - q1
                                                lower = q1 - 1.5 * iqr
                                                upper = q3 + 1.5 * iqr
                                                outliers = vals[(vals < lower) | (vals > upper)]
                                                outlier_info[group] = (len(outliers), list(outliers))
                                            outlier_table = pd.DataFrame({
                                                'Outlier Count': {g: outlier_info[g][0] for g in outlier_info},
                                                'Outlier Values': {g: outlier_info[g][1] for g in outlier_info}
                                            })
                                            st.dataframe(outlier_table, use_container_width=True)
                                        else:
                                            st.warning("No group statistics available for outlier detection.")
                                # Section 5: Normality Test
                                with st.expander("🧪 Normality Test (Shapiro-Wilk)", expanded=False):
                                    from scipy.stats import shapiro
                                    normality_results = {}
                                    normality_df = None
                                    stats_df = analysis_results.get('stats_summary')
                                    if stats_df is not None and not stats_df.empty:
                                        for group in stats_df.index:
                                            vals = data[data[group_column] == group][score_column].dropna()
                                            if len(vals) >= 3:
                                                stat, pval = shapiro(vals)
                                                normality_results[group] = {'W': stat, 'p-value': pval}
                                            else:
                                                normality_results[group] = {'W': None, 'p-value': None}
                                        normality_df = pd.DataFrame(normality_results).T
                                        st.dataframe(normality_df, use_container_width=True)
                                    else:
                                        st.warning("No group statistics available for normality testing.")
            
            # Initialize variables outside of the statistical section to avoid UnboundLocalError
            # These will be available for export even if statistics section is hidden
            normality_df = None
            outlier_table = None
            
            # Store normality_df and outlier_table in a broader scope if they were created
            if show_statistics and 'analysis_results' in locals() and analysis_results:
                try:
                    from scipy.stats import shapiro
                    normality_results = {}
                    stats_df = analysis_results.get('stats_summary')
                    if stats_df is not None and not stats_df.empty:
                        for group in stats_df.index:
                            vals = data[data[group_column] == group][score_column].dropna()
                            if len(vals) >= 3:
                                stat, pval = shapiro(vals)
                                normality_results[group] = {'W': stat, 'p-value': pval}
                            else:
                                normality_results[group] = {'W': None, 'p-value': None}
                        normality_df = pd.DataFrame(normality_results).T
                except Exception:
                    normality_df = None
                
                # Also compute outlier_table for export
                try:
                    if MODULES_AVAILABLE and len(data) > 1000:
                        outlier_table = get_fast_outlier_detection(data, score_column, group_column)
                    else:
                        outlier_info = {}
                        stats_df = analysis_results.get('stats_summary')
                        if stats_df is not None and not stats_df.empty:
                            for group in stats_df.index:
                                vals = data[data[group_column] == group][score_column].dropna()
                                q1 = vals.quantile(0.25)
                                q3 = vals.quantile(0.75)
                                iqr = q3 - q1
                                lower = q1 - 1.5 * iqr
                                upper = q3 + 1.5 * iqr
                                outliers = vals[(vals < lower) | (vals > upper)]
                                outlier_info[group] = (len(outliers), list(outliers))
                            outlier_table = pd.DataFrame({
                                'Outlier Count': {g: outlier_info[g][0] for g in outlier_info},
                                'Outlier Values': {g: outlier_info[g][1] for g in outlier_info}
                            })
                except Exception:
                    outlier_table = None
            else:
                # Show statistics disabled - use full width for plot
                col1, col2 = st.columns([0.05, 0.95], gap="small")
                with col1:
                    st.empty()  # Empty placeholder
                    
                    with col2:
                        # Only show plot if fig is valid
                        if fig is not None:
                            st.markdown("<div id='results_plot'></div>", unsafe_allow_html=True)
                            # Get plot mode from session state
                            plot_mode = st.session_state.get('fine_plot_mode', 'Interactive (Plotly)')
                            interactive_mode = plot_mode == "Interactive (Plotly)"
                            if interactive_mode:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # Static mode: only display if fig is a Matplotlib figure
                                from matplotlib.figure import Figure as MplFigure
                                if isinstance(fig, MplFigure):
                                    try:
                                        st.pyplot(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error displaying plot: {e}")
                                else:
                                    st.error("Static plot is not a Matplotlib figure. Cannot render.")
                            
                            # Additional plots if enabled
                            show_additional_plots = st.session_state.get('show_additional_plots', False)
                            if show_additional_plots:
                                st.markdown("### 📊 Additional Visualizations")
                                
                                # Get numeric columns for analysis
                                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                                conservation_cols = [col for col in num_cols if any(keyword in col.lower() 
                                                   for keyword in ['phylop', 'phastcons', 'gerp', 'conservation', 'cadd', 'revel'])]
                                
                                # Smart correlation matrix - show different types based on available data
                                if len(conservation_cols) > 1:
                                    # Conservation-focused correlation
                                    st.markdown("**[i] Conservation Score Correlations**")
                                    st.caption(f"Analyzing {len(conservation_cols)} conservation scores")
                                    
                                    corr_matrix = data[conservation_cols].corr()
                                    
                                    if interactive_mode:
                                        import plotly.express as px
                                        corr_fig = px.imshow(
                                            corr_matrix, 
                                            title="Conservation Score Correlation Matrix",
                                            color_continuous_scale='RdBu_r',
                                            aspect='auto',
                                            text_auto=True
                                        )
                                        st.plotly_chart(corr_fig, use_container_width=True)
                                    else:
                                        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                                        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                                                   square=True, ax=ax_corr, fmt='.3f')
                                        ax_corr.set_title("Conservation Score Correlations")
                                        st.pyplot(fig_corr)
                                        
                                elif len(num_cols) > 1:
                                    # General numeric correlation
                                    st.markdown("**🔗 Numeric Variable Correlations**")
                                    st.caption(f"Analyzing {len(num_cols)} numeric variables")
                                    
                                    corr_matrix = data[num_cols].corr()
                                    
                                    if interactive_mode:
                                        import plotly.express as px
                                        corr_fig = px.imshow(
                                            corr_matrix, 
                                            title="Numeric Variable Correlation Matrix",
                                            color_continuous_scale='RdBu_r',
                                            aspect='auto'
                                        )
                                        st.plotly_chart(corr_fig, use_container_width=True)
                                    else:
                                        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                                        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                                                   square=True, ax=ax_corr)
                                        ax_corr.set_title("Numeric Variable Correlations")
                                        st.pyplot(fig_corr)
                                
                                # Conservation score comparison plots
                                if len(conservation_cols) > 1:
                                    st.markdown("**🧬 Conservation Score Comparison**")
                                    
                                    # Create pairwise plots for first 3 conservation scores
                                    plot_cols = conservation_cols[:3]
                                    
                                    if interactive_mode and len(plot_cols) >= 2:
                                        import plotly.express as px
                                        scatter_fig = px.scatter_matrix(
                                            data, 
                                            dimensions=plot_cols,
                                            color=group_column,
                                            title="Conservation Score Pairwise Comparison",
                                            opacity=0.7
                                        )
                                        st.plotly_chart(scatter_fig, use_container_width=True)
                                    else:
                                        # Simple correlation scatter for top 2 scores
                                        if len(plot_cols) >= 2:
                                            fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
                                            for group in data[group_column].unique():
                                                if pd.notna(group):
                                                    group_data = data[data[group_column] == group]
                                                    ax_scatter.scatter(
                                                        group_data[plot_cols[0]], 
                                                        group_data[plot_cols[1]], 
                                                        label=str(group), alpha=0.7
                                                    )
                                            ax_scatter.set_xlabel(plot_cols[0].replace('_', ' ').title())
                                            ax_scatter.set_ylabel(plot_cols[1].replace('_', ' ').title())
                                            ax_scatter.set_title(f"{plot_cols[0]} vs {plot_cols[1]}")
                                            ax_scatter.legend()
                                            st.pyplot(fig_scatter)
                            
                            # --- Scroll down to results section ---
                            import streamlit.components.v1 as components
                            if st.session_state.get('scroll_to_results', False):
                                components.html("""
                                    <script>
                                    setTimeout(function() {
                                        var el = document.getElementById('results_plot');
                                        if (el) {
                                            el.scrollIntoView({behavior: 'smooth', block: 'start'});
                                        }
                                    }, 600);
                                    </script>
                                """, height=0)
                                st.session_state['scroll_to_results'] = False
                        else:
                            st.error("No plot was generated. Please check your data and visualization settings.")
                            # Remove empty box: do not show any placeholder or empty container
            
            # Data export with organized interface
            if analysis_results is not None:
                with st.expander("📥 Export Results & Data"):
                    import json
                    from datetime import datetime
                    # Organized export interface
                    st.markdown("**📊 Download Analysis Results**")
                    col1, col2 = st.columns(2)
                    # --- Main Results ---
                    with col1:
                        st.markdown("**Main Results**")
                        results_data = {
                            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'h_statistic': analysis_results['h_statistic'],
                            'p_value': analysis_results['p_value'],
                            'significant': int(analysis_results['significant']),
                            'n_groups': analysis_results['n_groups']
                        }
                        results_df = pd.DataFrame([results_data])
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        st.download_button(
                            label="📄 Main Results CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"taxoconserv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        json_buffer = io.StringIO()
                        json.dump(results_data, json_buffer, indent=2)
                        json_buffer.seek(0)
                        st.download_button(
                            label="JSON",
                            data=json_buffer.getvalue(),
                            file_name=f"taxoconserv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    # --- Group Statistics ---
                    with col1:
                        st.markdown("**Group Statistics**")
                        stats_df = analysis_results.get('stats_summary')
                        if stats_df is not None:
                            stats_csv = io.StringIO()
                            stats_df.to_csv(stats_csv)
                            stats_csv.seek(0)
                            st.download_button(
                                label="📊 Group Stats CSV",
                                data=stats_csv.getvalue(),
                                file_name="group_statistics.csv",
                                mime="text/csv"
                            )
                            stats_json = stats_df.reset_index().to_json(orient="records", indent=2)
                            st.download_button(
                                label="JSON",
                                data=stats_json,
                                file_name="group_statistics.json",
                                mime="application/json"
                            )
                    # --- Post-hoc Dunn Test ---
                    with col2:
                        st.markdown("**Post-hoc Dunn Test**")
                        try:
                            import scikit_posthocs as sp # type: ignore
                            dunn_results = sp.posthoc_dunn(data, val_col=score_column, group_col=group_column, p_adjust='bonferroni')
                            # Format the results for better display and export
                            dunn_formatted = dunn_results.round(8)  # Round to 8 decimal places
                            dunn_csv = io.StringIO()
                            if isinstance(dunn_formatted.index, pd.MultiIndex):
                                dunn_results_reset = dunn_formatted.reset_index()
                            else:
                                dunn_results_reset = dunn_formatted
                            dunn_results_reset.to_csv(dunn_csv, index=False)
                            dunn_csv.seek(0)
                            st.download_button(
                                label="CSV",
                                data=dunn_csv.getvalue(),
                                file_name="posthoc_results.csv",
                                mime="text/csv"
                            )
                            dunn_json = dunn_results_reset.to_json(orient="records", indent=2)
                            st.download_button(
                                label="JSON",
                                data=dunn_json,
                                file_name="posthoc_results.json",
                                mime="application/json"
                            )
                        except ImportError:
                            st.info("scikit-posthocs package is not installed.")
                        except Exception:
                            pass
                    # --- Outlier Table ---
                    with col2:
                        st.markdown("**Outlier Table**")
                        if 'outlier_table' in locals() and outlier_table is not None and not outlier_table.empty:
                            outlier_csv = io.StringIO()
                            outlier_table.to_csv(outlier_csv)
                            outlier_csv.seek(0)
                            st.download_button(
                                label="CSV",
                                data=outlier_csv.getvalue(),
                                file_name="outlier_table.csv",
                                mime="text/csv"
                            )
                            outlier_json = outlier_table.reset_index().to_json(orient="records", indent=2)
                            st.download_button(
                                label="JSON",
                                data=outlier_json,
                                file_name="outlier_table.json",
                                mime="application/json"
                            )
                    # --- Normality Test ---
                    with col2:
                        st.markdown("**Normality Test**")
                        if normality_df is not None and not normality_df.empty:
                            normality_csv = io.StringIO()
                            normality_df.to_csv(normality_csv)
                            normality_csv.seek(0)
                            st.download_button(
                                label="CSV",
                                data=normality_csv.getvalue(),
                                file_name="normality_test_results.csv",
                                mime="text/csv"
                            )
                            normality_json = normality_df.reset_index().to_json(orient="records", indent=2)
                            st.download_button(
                                label="JSON",
                                data=normality_json,
                                file_name="normality_test_results.json",
                                mime="application/json"
                            )
                    # --- Plot Image Export ---
                    with col2:
                        st.markdown("**Plot Image Export**")
                        # Check figure type for proper export
                        if fig is not None:
                            # Get plot mode setting
                            plot_mode = st.session_state.get('fine_plot_mode', 'Interactive (Plotly)')
                            interactive_mode = plot_mode == "Interactive (Plotly)"
                            
                            # Check if we have a plotly figure (interactive mode)
                            import plotly.graph_objects as go
                            if interactive_mode and isinstance(fig, (go.Figure, dict)):
                                # HTML indirme seçeneği (kaleido gerektirmez)
                                import plotly.io as pio
                                html_str_io = io.StringIO()
                                pio.write_html(fig, file=html_str_io, auto_open=False)
                                html_str = html_str_io.getvalue()
                                html_bytes = html_str.encode('utf-8')
                                st.download_button(
                                    label="Download Plot as HTML (Interactive)",
                                    data=html_bytes,
                                    file_name="conservation_plot.html",
                                    mime="text/html",
                                    key="download_html_interactive"
                                )
                            # Check if we have a matplotlib figure (static mode)  
                            elif not interactive_mode:
                                # Static figure export (Matplotlib)
                                from matplotlib.figure import Figure as MplFigure
                                if isinstance(fig, MplFigure):
                                    # Export PNG
                                    try:
                                        buf_png = io.BytesIO()
                                        fig.savefig(buf_png, format="png", bbox_inches='tight', dpi=300)
                                        buf_png.seek(0)
                                        st.download_button(
                                            label="Download Plot as PNG",
                                            data=buf_png,
                                            file_name="conservation_plot.png",
                                            mime="image/png",
                                            key="download_png_static"
                                        )
                                    except Exception as e:
                                        st.error(f"Static PNG export failed: {e}")
                                    # Export SVG
                                    try:
                                        buf_svg = io.BytesIO()
                                        fig.savefig(buf_svg, format="svg", bbox_inches='tight', dpi=300)
                                        buf_svg.seek(0)
                                        st.download_button(
                                            label="Download Plot as SVG (Publication Quality)",
                                            data=buf_svg,
                                            file_name="conservation_plot.svg",
                                            mime="image/svg+xml",
                                            key="download_svg_static"
                                        )
                                    except Exception as e:
                                        st.error(f"Static SVG export failed: {e}")
                                else:
                                    st.info("Plot export not available for this plot type.")
                            else:
                                st.info("Plot export not available for current plot type.")

def run_variant_analysis():
    """VCF variant conservation analysis interface"""
    st.header("🔬 Variant Conservation Analysis")
    
    st.markdown("""
    Upload a VCF file to analyze conservation scores for individual variants, or use sample data to explore the interface.
    This module provides detailed conservation patterns for specific genomic positions.
    """)
    
    # Data input section
    st.subheader("📁 Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VCF file uploader
        vcf_file = st.file_uploader(
            "Upload VCF File",
            type=['vcf'],
            help="Upload a VCF file containing variants for conservation analysis",
            key="vcf_file_uploader"
        )
        
        # Sample data button
        if st.button(
            "🧪 Load Sample VCF Data",
            help="Load built-in sample variant dataset for testing",
            type="secondary",
            key="variant_sample_data_button"
        ):
            st.session_state['vcf_sample_loaded'] = True
            st.rerun()
    
    with col2:
        # Conservation database uploader
        conservation_db = st.file_uploader(
            "Upload Conservation Database (Optional)",
            type=['csv', 'tsv'],
            help="Upload conservation score database or use built-in data",
            key="conservation_db_uploader"
        )
        
        use_demo_conservation = st.checkbox(
            "Use Built-in Conservation Database",
            value=True,
            help="Use built-in conservation data for analysis"
        )
    
    # About sample data
    with st.expander("ℹ️ About Sample VCF Data", expanded=False):
        st.markdown("""
        **Sample VCF Dataset Features**
        - **Variants**: 50 clinically relevant variants
        - **Genes**: BRCA1, BRCA2, TP53, PTEN, KRAS, etc.
        - **Scores**: phyloP, GERP, phastCons, CADD, REVEL
        - **Annotations**: Consequence, pathogenicity predictions
        - **Use**: Testing variant conservation analysis workflow
        """)
    
    # Configuration section
    st.subheader("⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col2:
        max_variants = st.number_input(
            "Max Variants to Analyze",
            min_value=1,
            max_value=10000,
            value=100,
            help="🔢 Performance Limit: Large VCF files can contain thousands of variants. This setting limits how many variants to analyze to keep the interface responsive. For example, if your VCF has 5000 variants but this is set to 100, only the first 100 variants will be processed."
        )
        
        st.caption("💡 **Why limit variants?** VCF files can be very large (10K+ variants). Processing all variants at once might be slow, so we analyze a manageable subset first.")
    
    # Handle sample VCF data loading
    variant_data = None
    conservation_database = None
    
    # Initialize session state for persistent data storage
    if 'variant_data_persistent' not in st.session_state:
        st.session_state['variant_data_persistent'] = None
    if 'conservation_database_persistent' not in st.session_state:
        st.session_state['conservation_database_persistent'] = None
    
    if st.session_state.get('vcf_sample_loaded', False):
        try:
            # Load sample VCF data
            with st.spinner("Loading sample VCF data..."):
                from src.input_parser import create_sample_vcf_data, create_demo_data
                import numpy as np  # Ensure numpy is available
                variant_data = create_sample_vcf_data()
                
                # Also load compatible conservation database
                conservation_database = create_demo_data()
                
                # Ensure position compatibility
                if variant_data is not None and conservation_database is not None:
                    # Align conservation data positions with VCF positions for demonstration
                    vcf_positions = variant_data['POS'].tolist()[:20]  # Take first 20 VCF positions
                    
                    # Create aligned conservation data
                    aligned_conservation = conservation_database.copy()
                    aligned_conservation['position'] = np.tile(vcf_positions, 
                                                             (len(aligned_conservation) // len(vcf_positions)) + 1)[:len(aligned_conservation)]
                    
                    conservation_database = aligned_conservation
                
            st.success("✅ Sample VCF data and conservation database loaded successfully!")
            st.info(f"📊 Loaded {len(variant_data)} sample variants with {len(conservation_database)} conservation data points")
            
            # Store data in session state for persistence
            st.session_state['variant_data'] = variant_data
            st.session_state['conservation_database'] = conservation_database
            st.session_state['variant_data_persistent'] = variant_data
            st.session_state['conservation_database_persistent'] = conservation_database
            
            # Clear the flag
            st.session_state['vcf_sample_loaded'] = False
            
        except Exception as e:
            st.error(f"Error loading sample VCF data: {e}")
            st.session_state['vcf_sample_loaded'] = False    
    
    # Check if data exists in session state (try multiple sources)
    if variant_data is None:
        if 'variant_data_persistent' in st.session_state and st.session_state['variant_data_persistent'] is not None:
            variant_data = st.session_state['variant_data_persistent']
            conservation_database = st.session_state['conservation_database_persistent']
            st.info("📋 Using persistent sample data")
        elif 'variant_data' in st.session_state and st.session_state['variant_data'] is not None:
            variant_data = st.session_state['variant_data']
            conservation_database = st.session_state['conservation_database']
            st.info("📋 Using session sample data")
    
    # Display variant data analysis
    if variant_data is not None:
        st.subheader("📊 Variant Conservation Analysis")
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Variants", len(variant_data))
        with col2:
            st.metric("Unique Genes", variant_data['GENE'].nunique())
        with col3:
            conservation_scores = ['phyloP_score', 'GERP_score', 'phastCons_score']
            available_scores = [col for col in conservation_scores if col in variant_data.columns]
            st.metric("Conservation Scores", len(available_scores))
        with col4:
            pathogenic_variants = variant_data[variant_data['Pathogenicity'].isin(['Pathogenic', 'Likely_pathogenic'])]
            st.metric("Pathogenic/Likely", len(pathogenic_variants))
        
        # Display data table
        st.subheader("📋 Variant Data Table")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            selected_genes = st.multiselect(
                "Filter by Gene",
                options=sorted(variant_data['GENE'].unique()),
                default=[]
            )
        with col2:
            selected_pathogenicity = st.multiselect(
                "Filter by Pathogenicity",
                options=variant_data['Pathogenicity'].unique(),
                default=[]
            )
        
        # Apply filters
        filtered_data = variant_data.copy()
        if selected_genes:
            filtered_data = filtered_data[filtered_data['GENE'].isin(selected_genes)]
        if selected_pathogenicity:
            filtered_data = filtered_data[filtered_data['Pathogenicity'].isin(selected_pathogenicity)]
        
        # Display filtered data
        display_cols = ['GENE', 'CHROM', 'POS', 'REF', 'ALT', 'Consequence', 'phyloP_score', 'GERP_score', 'phastCons_score', 'Pathogenicity']
        available_display_cols = [col for col in display_cols if col in filtered_data.columns]
        
        st.dataframe(filtered_data[available_display_cols], use_container_width=True)
        
        # Conservation analysis
        if st.button("🔍 Analyze Conservation Patterns", type="primary", key="variant_conservation_analysis_button"):
            st.write("🚀 Button clicked! Starting analysis...")  # Debug message
            st.subheader("📈 Conservation Score Analysis")
            
            # Check data availability
            if conservation_database is None:
                st.error("❌ Conservation database not loaded! Please load sample data first.")
                return
            
            if filtered_data.empty:
                st.error("❌ No variant data available for analysis!")
                return
            
            st.info(f"🔍 Analyzing {len(filtered_data)} variants with conservation database of {len(conservation_database)} entries")
            
            with st.spinner("Analyzing variant conservation patterns..."):
                
                # Initialize variant analyzer for comprehensive analysis
                try:
                    from src.variant_analysis import VariantConservationAnalyzer
                    analyzer = VariantConservationAnalyzer(conservation_database)
                    
                    # Convert variant data to analyzer format
                    variants_for_analysis = []
                    for _, row in filtered_data.iterrows():
                        variant = {
                            'chromosome': row['CHROM'],
                            'position': row['POS'],
                            'ref': row['REF'],
                            'alt': [row['ALT']] if isinstance(row['ALT'], str) else row['ALT'],
                            'variant_id': f"{row['CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}"
                        }
                        variants_for_analysis.append(variant)
                    
                    # Perform batch conservation analysis
                    conservation_results = []
                    for variant in variants_for_analysis:
                        result = analyzer.get_conservation_for_variant(variant)
                        conservation_results.append(result)
                    
                    # Display ACMG interpretation summary with enhanced metrics
                    st.subheader("🏥 ACMG Clinical Interpretation")
                    
                    acmg_summary = {'PP3': 0, 'PP3_weak': 0, 'BP4': 0, 'insufficient': 0}
                    consensus_scores = []
                    high_confidence_variants = 0
                    
                    for result in conservation_results:
                        if result.get('conservation_available'):
                            # Check for enhanced ACMG interpretation
                            enhanced_acmg = result.get('enhanced_acmg', {})
                            if enhanced_acmg:
                                criterion = enhanced_acmg.get('primary_criterion', 'insufficient_evidence')
                                if criterion == 'PP3':
                                    acmg_summary['PP3'] += 1
                                elif criterion in ['PP3_moderate', 'PP3_weak']:
                                    acmg_summary['PP3_weak'] += 1
                                elif criterion == 'BP4':
                                    acmg_summary['BP4'] += 1
                                else:
                                    acmg_summary['insufficient'] += 1
                            else:
                                # Fallback to basic ACMG interpretation
                                acmg_criteria = result.get('acmg_interpretation', {}).get('acmg_criteria', [])
                                if 'PP3' in acmg_criteria:
                                    acmg_summary['PP3'] += 1
                                elif 'PP3_weak' in acmg_criteria:
                                    acmg_summary['PP3_weak'] += 1
                                elif 'BP4' in acmg_criteria:
                                    acmg_summary['BP4'] += 1
                                else:
                                    acmg_summary['insufficient'] += 1
                            
                            # Collect consensus scores
                            consensus_data = result.get('consensus_conservation', {})
                            if consensus_data.get('consensus_score') is not None:
                                consensus_scores.append(consensus_data['consensus_score'])
                                if consensus_data.get('confidence_level') == 'high':
                                    high_confidence_variants += 1
                        else:
                            acmg_summary['insufficient'] += 1
                    
                    # Display enhanced metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔴 PP3 (Strong Conservation)", acmg_summary['PP3'])
                    with col2:
                        st.metric("🟡 PP3 Weak (Moderate)", acmg_summary['PP3_weak'])
                    with col3:
                        st.metric("🟢 BP4 (Low Conservation)", acmg_summary['BP4'])
                    with col4:
                        st.metric("⚪ Insufficient Evidence", acmg_summary['insufficient'])
                    
                    # Additional consensus score metrics
                    if consensus_scores:
                        st.markdown("### 📊 Consensus Conservation Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_consensus = np.mean(consensus_scores)
                            st.metric("Average Consensus Score", f"{avg_consensus:.3f}")
                        with col2:
                            st.metric("High Confidence Variants", high_confidence_variants)
                        with col3:
                            coverage = len(consensus_scores) / len(conservation_results) * 100
                            st.metric("Analysis Coverage", f"{coverage:.1f}%")
                        
                        # Consensus score distribution
                        if len(consensus_scores) > 1:
                            fig = px.histogram(
                                x=consensus_scores, 
                                nbins=20,
                                title="Consensus Conservation Score Distribution",
                                labels={'x': 'Consensus Score', 'y': 'Count'}
                            )
                            
                            # Add interpretation threshold lines
                            fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                                        annotation_text="High Conservation (0.8)")
                            fig.add_vline(x=0.6, line_dash="dash", line_color="orange", 
                                        annotation_text="Moderate (0.6)")
                            fig.add_vline(x=0.3, line_dash="dash", line_color="green", 
                                        annotation_text="Low Conservation (0.3)")
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Advanced analysis failed: {str(e)}")
                    st.error("📍 Error details:")
                    st.code(f"Error type: {type(e).__name__}")
                    st.code(f"Error message: {str(e)}")
                    import traceback
                    st.code(f"Traceback: {traceback.format_exc()}")
                    st.info("💡 Try reloading sample data or check console for details")
            
            # Conservation score distributions
            conservation_cols = ['phyloP_score', 'GERP_score', 'phastCons_score']
            available_conservation_cols = [col for col in conservation_cols if col in filtered_data.columns]
            
            if available_conservation_cols:
                # Create conservation score plots
                for score_col in available_conservation_cols:
                    st.markdown(f"**{score_col} Distribution**")
                    
                    # Create three columns for different visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Box plot by pathogenicity
                        if len(filtered_data['Pathogenicity'].unique()) > 1:
                            fig = px.box(filtered_data, x='Pathogenicity', y=score_col, 
                                       color='Pathogenicity',
                                       title=f"{score_col} by Pathogenicity Classification")
                            fig.update_layout(xaxis_title="Pathogenicity", yaxis_title=score_col, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with viz_col2:
                        # Histogram with conservation thresholds
                        fig = px.histogram(filtered_data, x=score_col, nbins=20,
                                         title=f"{score_col} Distribution")
                        
                        # Add conservation threshold lines
                        if 'phylop' in score_col.lower():
                            fig.add_vline(x=2.0, line_dash="dash", line_color="red", 
                                        annotation_text="High Conservation (2.0)")
                            fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                                        annotation_text="Moderate (0.5)")
                        elif 'gerp' in score_col.lower():
                            fig.add_vline(x=4.0, line_dash="dash", line_color="red", 
                                        annotation_text="High Constraint (4.0)")
                            fig.add_vline(x=2.0, line_dash="dash", line_color="orange", 
                                        annotation_text="Moderate (2.0)")
                        elif 'phastcons' in score_col.lower():
                            fig.add_vline(x=0.8, line_dash="dash", line_color="red", 
                                        annotation_text="High Conservation (0.8)")
                            fig.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                                        annotation_text="Moderate (0.5)")
                        
                        fig.update_layout(xaxis_title=score_col, yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Gene-level conservation analysis
                if len(selected_genes) > 1 or not selected_genes:
                    st.markdown("**Conservation by Gene**")
                    
                    # Calculate mean conservation scores by gene
                    gene_conservation = filtered_data.groupby('GENE')[available_conservation_cols].mean().round(3)
                    gene_conservation = gene_conservation.reset_index()
                    
                    # Display top conserved genes
                    for score_col in available_conservation_cols:
                        top_genes = gene_conservation.nlargest(10, score_col)
                        st.markdown(f"*Top 10 genes by {score_col}:*")
                        
                        fig = px.bar(top_genes, x='GENE', y=score_col,
                                   title=f"Top Genes by {score_col}")
                        fig.update_layout(xaxis_title="Gene", yaxis_title=score_col)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics with interpretation
                st.subheader("📊 Conservation Summary Statistics")
                summary_stats = filtered_data[available_conservation_cols].describe().round(3)
                st.dataframe(summary_stats, use_container_width=True)
                
                # Conservation interpretation guide
                with st.expander("📖 Conservation Score Interpretation Guide", expanded=False):
                    st.markdown("""
                    **PhyloP Scores:**
                    - **> 2.0**: Strong evolutionary conservation (supports pathogenicity)
                    - **0.5 - 2.0**: Moderate conservation 
                    - **< 0.0**: Accelerated evolution (supports benign)
                    
                    **GERP++ Scores:**
                    - **> 4.0**: Strong evolutionary constraint (supports pathogenicity)
                    - **2.0 - 4.0**: Moderate constraint
                    - **< 0.0**: Accelerated evolution (supports benign)
                    
                    **phastCons Scores:**
                    - **> 0.8**: High conservation probability (supports pathogenicity)
                    - **0.5 - 0.8**: Moderate conservation
                    - **< 0.2**: Low conservation probability (supports benign)
                    
                    **ACMG Guidelines:**
                    - **PP3**: Multiple computational evidence supporting deleterious effect
                    - **BP4**: Multiple computational evidence suggesting no impact
                    """)
            
            else:
                st.warning("No conservation score columns found in the data.")
        
        # Export data with enhanced options
        st.subheader("💾 Export Enhanced Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic CSV export
            csv_data = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Basic Data (CSV)",
                data=csv_data,
                file_name=f"variant_conservation_basic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Enhanced analysis export (if available)
            if 'conservation_results' in locals() and conservation_results:
                try:
                    # Create comprehensive report DataFrame
                    export_data = []
                    for i, result in enumerate(conservation_results):
                        row_data = {
                            'Variant_ID': result.get('variant_id', f'variant_{i}'),
                            'Chromosome': result.get('chromosome', ''),
                            'Position': result.get('position', ''),
                            'Ref_Allele': result.get('ref', ''),
                            'Alt_Allele': ','.join(result.get('alt', [])) if isinstance(result.get('alt'), list) else result.get('alt', ''),
                            'Conservation_Available': result.get('conservation_available', False)
                        }
                        
                        # Add consensus conservation data
                        consensus_data = result.get('consensus_conservation', {})
                        if consensus_data:
                            row_data.update({
                                'Consensus_Score': consensus_data.get('consensus_score', ''),
                                'Confidence_Level': consensus_data.get('confidence_level', ''),
                                'Score_Agreement': consensus_data.get('score_agreement', ''),
                                'Conservation_Interpretation': consensus_data.get('interpretation', '')
                            })
                        
                        # Add ACMG interpretation
                        enhanced_acmg = result.get('enhanced_acmg', {})
                        basic_acmg = result.get('acmg_interpretation', {})
                        
                        if enhanced_acmg:
                            row_data.update({
                                'ACMG_Criterion': enhanced_acmg.get('primary_criterion', ''),
                                'Evidence_Strength': enhanced_acmg.get('evidence_strength', ''),
                                'ACMG_Confidence': enhanced_acmg.get('confidence', ''),
                                'Supporting_Details': '; '.join(enhanced_acmg.get('supporting_details', [])),
                                'Recommendations': '; '.join(enhanced_acmg.get('recommendations', []))
                            })
                        elif basic_acmg:
                            row_data.update({
                                'ACMG_Criterion': ','.join(basic_acmg.get('acmg_criteria', [])),
                                'Evidence_Strength': basic_acmg.get('evidence_strength', ''),
                                'Conservation_Level': basic_acmg.get('conservation_level', '')
                            })
                        
                        # Add individual conservation scores
                        conservation_scores = result.get('conservation_scores', {})
                        for score_name, score_data in conservation_scores.items():
                            if isinstance(score_data, dict) and 'mean' in score_data:
                                row_data[f'{score_name}_mean'] = score_data['mean']
                                row_data[f'{score_name}_std'] = score_data.get('std', '')
                        
                        export_data.append(row_data)
                    
                    # Create DataFrame and export
                    comprehensive_df = pd.DataFrame(export_data)
                    comprehensive_csv = comprehensive_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📊 Download Comprehensive Analysis (CSV)",
                        data=comprehensive_csv,
                        file_name=f"variant_conservation_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.warning(f"Enhanced export not available: {e}")
        
        # Clinical report option
        if 'conservation_results' in locals() and conservation_results:
            with st.expander("📋 Generate Clinical Report", expanded=False):
                st.markdown("### Conservation Analysis Report")
                
                # Report metadata
                report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                total_variants = len(conservation_results)
                analyzed_variants = sum(1 for r in conservation_results if r.get('conservation_available'))
                coverage_percentage = analyzed_variants/total_variants*100 if total_variants > 0 else 0
                
                # Generate simple clinical report
                clinical_report = f"""# Variant Conservation Analysis Report

**Generated:** {report_date}  
**Total Variants:** {total_variants}  
**Successfully Analyzed:** {analyzed_variants} ({coverage_percentage:.1f}%)

## Summary

This report provides conservation analysis for {total_variants} variants using phyloP, phastCons, and GERP++ scores.

### ACMG Interpretation Summary
- **PP3 (Supporting Pathogenic):** {acmg_summary.get('PP3', 0)} variants
- **PP3 Weak (Moderate Evidence):** {acmg_summary.get('PP3_weak', 0)} variants  
- **BP4 (Supporting Benign):** {acmg_summary.get('BP4', 0)} variants
- **Insufficient Evidence:** {acmg_summary.get('insufficient', 0)} variants

### Key Findings
"""

                
                # Add consensus scoring results if available
                if consensus_scores:
                    avg_consensus = np.mean(consensus_scores)
                    high_conservation = sum(1 for s in consensus_scores if s > 0.8)
                    clinical_report += f"""
- **Average Consensus Conservation Score:** {avg_consensus:.3f}
- **Highly Conserved Variants (>0.8):** {high_conservation} 
- **High Confidence Analyses:** {high_confidence_variants}
"""

                clinical_report += """

## Methodology

Conservation scores were analyzed using:
1. **phyloP:** Evolutionary conservation based on phylogenetic p-values
2. **phastCons:** Hidden Markov Model-based conservation probability  
3. **GERP++:** Genomic Evolutionary Rate Profiling
4. **Consensus Scoring:** Weighted combination of multiple metrics

## ACMG Guidelines Applied

- **PP3:** Multiple computational evidence supporting deleterious effect
- **BP4:** Multiple computational evidence suggesting no impact
- Evidence strength determined by score agreement and confidence levels

## Limitations

- Conservation analysis is one component of variant interpretation
- Should be combined with other ACMG criteria for clinical decisions
- Scores may not reflect functional impact in all biological contexts

---
*Report generated by TaxoConserv v2.1.0*
"""
                
                st.markdown(clinical_report)
                
                # PDF Export only
                try:
                    # Create PDF from clinical report
                    from reportlab.lib.pagesizes import letter, A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib.units import inch
                    from reportlab.lib import colors
                    from io import BytesIO
                    
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch, bottomMargin=1*inch)
                    styles = getSampleStyleSheet()
                    
                    # Custom styles for the report
                    title_style = ParagraphStyle(
                        'CustomTitle',
                        parent=styles['Heading1'],
                        fontSize=16,
                        spaceAfter=20,
                        textColor=colors.darkblue
                    )
                    
                    heading_style = ParagraphStyle(
                        'CustomHeading',
                        parent=styles['Heading2'],
                        fontSize=14,
                        spaceAfter=12,
                        textColor=colors.darkgreen
                    )
                    
                    # Build PDF content
                    story = []
                    
                    # Parse clinical_report and add to PDF
                    lines = clinical_report.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('# '):
                            # Main title
                            story.append(Paragraph(line[2:], title_style))
                            story.append(Spacer(1, 12))
                        elif line.startswith('## '):
                            # Section heading
                            story.append(Paragraph(line[3:], heading_style))
                            story.append(Spacer(1, 8))
                        elif line.startswith('**') and line.endswith('**'):
                            # Bold text
                            text = line[2:-2]
                            story.append(Paragraph(f"<b>{text}</b>", styles['Normal']))
                            story.append(Spacer(1, 6))
                        elif line.startswith('- '):
                            # Bullet point
                            story.append(Paragraph(f"• {line[2:]}", styles['Normal']))
                            story.append(Spacer(1, 4))
                        elif line and not line.startswith('---'):
                            # Regular text
                            story.append(Paragraph(line, styles['Normal']))
                            story.append(Spacer(1, 6))
                        elif line.startswith('---'):
                            # Add line separator
                            story.append(Spacer(1, 12))
                    
                    # Build PDF
                    doc.build(story)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📄 Download Report (PDF)",
                        data=buffer.getvalue(),
                        file_name=f"conservation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                except ImportError:
                    st.warning("⚠️ PDF generation requires reportlab package. Installing...")
                    try:
                        import subprocess
                        subprocess.check_call(["pip", "install", "reportlab"])
                        st.success("✅ reportlab installed. Please refresh the page to generate PDF.")
                    except:
                        st.error("❌ Could not install reportlab. Please install manually: pip install reportlab")
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {e}")
                    # Fallback to simple text export
                    st.download_button(
                        label="📑 Download Report (TXT)",
                        data=clinical_report.replace('#', '').replace('*', ''),
                        file_name=f"conservation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )    # Handle VCF file upload (existing code)
    elif vcf_file is not None:
        try:
            # Import variant analysis module
            from src.variant_analysis import VariantConservationAnalyzer
            
            # Initialize analyzer
            analyzer = VariantConservationAnalyzer()
            
            # Load conservation database
            if use_demo_conservation:
                from src.input_parser import create_demo_data
                conservation_data = create_demo_data()
                analyzer.conservation_data = conservation_data
                st.success("✅ Demo conservation database loaded")
            elif conservation_db is not None:
                conservation_data = pd.read_csv(conservation_db)
                analyzer.conservation_data = conservation_data
                st.success("✅ Custom conservation database loaded")
            else:
                st.warning("⚠️ No conservation database loaded. Please upload a database or use demo data.")
                return
            
            # Save VCF file temporarily and parse
            vcf_content = vcf_file.read()
            temp_vcf_path = f"temp_variants_{int(time.time())}.vcf"
            
            with open(temp_vcf_path, 'wb') as f:
                f.write(vcf_content)
            
            # Parse VCF file
            with st.spinner("Parsing VCF file..."):
                variants = analyzer.parse_vcf_file(temp_vcf_path)
            
            # Limit variants for performance
            if len(variants) > max_variants:
                variants = variants[:max_variants]
                st.info(f"ℹ️ Analyzing first {max_variants} variants (out of {len(variants)} total)")
            
            st.success(f"✅ Parsed {len(variants)} variants from VCF file")
            
            # Analyze conservation for variants
            if st.button("🔍 Analyze Variant Conservation", type="primary", key="vcf_conservation_analysis_button"):
                with st.spinner("Analyzing conservation scores..."):
                    results = analyzer.analyze_variant_batch(variants)
                
                # Display results
                st.subheader("📊 Variant Conservation Results")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_variants = len(results)
                    st.metric("Total Variants", total_variants)
                
                with col2:
                    with_conservation = results['conservation_available'].sum()
                    st.metric("With Conservation Data", with_conservation)
                
                with col3:
                    if with_conservation > 0:
                        coverage = (with_conservation / total_variants) * 100
                        st.metric("Coverage", f"{coverage:.1f}%")
                
                with col4:
                    if 'overall_statistics' in results.columns:
                        high_conservation = 0  # Placeholder for high conservation count
                        st.metric("High Conservation", high_conservation)
                
                # Detailed results table
                st.subheader("📋 Detailed Results")
                
                # Filter and display results
                display_cols = ['variant_id', 'chromosome', 'position', 'ref', 'alt', 'conservation_available']
                
                if 'conservation_interpretation' in results.columns:
                    display_cols.append('conservation_interpretation')
                
                display_results = results[display_cols].copy()
                st.dataframe(display_results, use_container_width=True)
                
                # Conservation visualization for variants with data
                conservation_variants = results[results['conservation_available'] == True]
                
                if len(conservation_variants) > 0:
                    st.subheader("📈 Conservation Visualization")
                    
                    # Create conservation score plots
                    if len(conservation_variants) > 0:
                        st.info("🚧 Conservation plotting for variants is under development. Results table shows conservation analysis.")
                
                # Download results
                st.subheader("💾 Download Results")
                
                csv_data = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_data,
                    file_name=f"variant_conservation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Clean up temporary file
            try:
                os.remove(temp_vcf_path)
            except:
                pass
                
        except Exception as e:
            st.error(f"❌ Error analyzing VCF file: {e}")
            st.info("💡 Make sure the VCF file is properly formatted and contains valid variant data.")
    
    else:
        st.info("👆 Upload a VCF file to start variant conservation analysis")
        
        # Help section
        with st.expander("ℹ️ About Variant Conservation Analysis", expanded=False):
            st.markdown("""
            **Variant Conservation Analysis** provides:
            
            - **Per-variant conservation scores** (PhyloP, GERP, phastCons)
            - **Taxonomic conservation patterns** for each variant position
            - **Conservation interpretation** for ACMG criteria support
            - **Statistical analysis** of conservation significance
            - **Export capabilities** for clinical documentation
            
            **Input Requirements:**
            - VCF file with variant positions
            - Conservation score database (or use demo data)
            
            **Use Cases:**
            - Clinical variant interpretation support
            - Research variant prioritization
            - Conservation evidence for ACMG PP3/BP4 criteria
            """)

if __name__ == "__main__":
    main()
