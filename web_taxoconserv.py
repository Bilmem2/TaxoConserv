#!/usr/bin/env python3
"""
TaxoConserv Web Interface
A Streamlit-based web application for taxonomic conservation analysis

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
      v1.0
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

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("---")
    # Data Input Section
    st.sidebar.subheader("📁 Data Input")
    # File uploader için dinamik key kullan
    if 'file_uploader_key' not in st.session_state:
        st.session_state['file_uploader_key'] = 0
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Conservation Data",
        type=['csv', 'tsv'],
        help="Upload a CSV or TSV file containing conservation scores and taxonomic groups",
        key='uploaded_file_' + str(st.session_state['file_uploader_key'])
    )
    
    # Sample data button with icon
    if st.sidebar.button(
        "🧪 Load Sample Data",
        help="Load built-in conservation dataset for testing (PhyloP, GERP, phastCons scores)",
        use_container_width=True,
        type="secondary"
    ):
        st.session_state['demo_loaded'] = True
        st.rerun()
    
    # Add sample data info
    with st.sidebar.expander("ℹ️ About Sample Data", expanded=False):
        st.markdown("""
        **Conservation Dataset Features**
        - **Scores**: PhyloP, GERP, phastCons
        - **Groups**: Taxonomic families  
        - **Size**: 500+ entries
        - **Use**: Testing & demonstration
        """)
    
    st.sidebar.markdown("---")
    # Reset button with icon
    if st.sidebar.button("🔄 Reset", help="Clear all selections and restart the app"):
        # Tüm session_state'i temizle
        keys_to_delete = list(st.session_state.keys())
        for k in keys_to_delete:
            del st.session_state[k]
        
        # file_uploader'ın key'ini artırarak arayüzde de sıfırlanmasını sağla
        st.session_state['file_uploader_key'] = st.session_state.get('file_uploader_key', 0) + 1
        
        # Veri yükleme flag'larını açıkça sıfırla
        st.session_state['demo_loaded'] = False
        st.session_state['file_loaded'] = False
        st.session_state['current_data'] = None
        st.session_state['uploaded_file_cleared'] = True  # Yeni flag ekle
        
        st.rerun()
    
    data = None
    # Reset flag'ini kontrol et - eğer reset yapıldıysa dosya yüklemeyi atla
    if st.session_state.get('uploaded_file_cleared', False):
        # Reset flag'ini temizle
        st.session_state['uploaded_file_cleared'] = False
        uploaded_file = None  # Dosyayı zorla temizle
    
    # Demo veri flag'i varsa ve True ise demo veriyi yükle
    if st.session_state.get('demo_loaded', False) == True:
        from src.input_parser import create_demo_data
        data = create_demo_data()
        st.sidebar.success("✅ Demo data loaded successfully!")
        st.session_state['group_column'] = 'taxon_group'
        # Detect conservation scores in demo data
        detected_scores = detect_conservation_scores(data)
        if detected_scores:
            # Prioritize conservation scores
            num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
            prioritized_scores = prioritize_conservation_scores(num_cols, detected_scores)
            st.session_state['score_column'] = prioritized_scores[0]
            st.session_state['detected_conservation_scores'] = detected_scores
        else:
            st.session_state['score_column'] = 'conservation_score'
    # Dosya yüklenirse veri oku - ama sadece reset yapılmadıysa
    elif uploaded_file is not None and not st.session_state.get('uploaded_file_cleared', False):
        try:
            start_time = time.time()
            
            # Use cached data processing for better performance
            if MODULES_AVAILABLE:
                file_content = uploaded_file.read()
                file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'tsv'
                data = cached_data_processing(file_content, file_type)
                
                # Optimize DataFrame memory usage
                original_memory = DataOptimizer.get_memory_usage(data)
                data = DataOptimizer.optimize_dataframe(data)
                optimized_memory = DataOptimizer.get_memory_usage(data)
                
                # Optimize for large datasets with DuckDB
                data = optimize_large_dataset(data)
                
                # Record performance metrics
                load_time = time.time() - start_time
                performance_monitor.record_metric('data_load_time', load_time)
                
                # Silent memory optimization - no user notification needed
            else:
                # Fallback without caching
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.tsv'):
                    data = pd.read_csv(uploaded_file, sep='\t')
                else:
                    st.sidebar.error("❌ Unsupported file type! Only CSV or TSV allowed.")
                    data = None
            
            if data is not None:
                st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
                st.session_state['file_loaded'] = True
                st.session_state['current_data'] = data  # Store for caching
                
                # Otomatik kolon seçimi (demo ile aynı mantık)
                if 'taxon_group' in data.columns:
                    st.session_state['group_column'] = 'taxon_group'
                else:
                    st.session_state['group_column'] = data.columns[0]
                # Detect conservation scores and prioritize them
                if MODULES_AVAILABLE:
                    data_hash = cache.get_cache_key(data)
                    detected_scores = cached_conservation_score_detection(data_hash, list(data.columns))
                else:
                    detected_scores = detect_conservation_scores(data)
                    
                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                
                if detected_scores:
                    # Prioritize conservation scores
                    prioritized_scores = prioritize_conservation_scores(num_cols, detected_scores)
                    st.session_state['score_column'] = prioritized_scores[0]
                    # Store detected scores for UI display
                    st.session_state['detected_conservation_scores'] = detected_scores
                else:
                    st.session_state['score_column'] = num_cols[0] if num_cols else data.columns[0]
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {e}")
            data = None
    
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
        st.sidebar.subheader("🏷️ Taxon Grouping")
        
        # Smart group column detection
        group_column_options = []
        for col in data.columns:
            unique_count = data[col].nunique()
            total_rows = len(data)
            # Include if: not too many unique values (< 50% of rows) and not too few (> 1)
            if 1 < unique_count < total_rows // 2 and unique_count <= 20:
                group_column_options.append(col)
        
        # If no good options found, include all non-numeric columns
        if not group_column_options:
            group_column_options = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        
        # If still no options, include all columns
        if not group_column_options:
            group_column_options = list(data.columns)
        
        group_column_value = st.session_state.get('group_column')
        group_column_index = group_column_options.index(group_column_value) if group_column_value in group_column_options else 0
        group_column = st.sidebar.selectbox(
            "🏷️ Grouping Column",
            options=group_column_options,
            help="Select the column to use for taxon grouping (e.g. taxon_group, family, genus)",
            key='group_column',
            index=group_column_index
        )
        with st.sidebar.expander("🧬 Advanced Grouping Options", expanded=False):
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
        st.sidebar.subheader("📊 Conservation Score Column")
        
        # Get detected conservation scores
        detected_scores = st.session_state.get('detected_conservation_scores', {})
        
        # Get all numeric columns and prioritize conservation scores
        all_numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        prioritized_options = prioritize_conservation_scores(all_numeric_cols, detected_scores)
        
        # Create options with descriptions for conservation scores
        score_column_options = []
        score_column_labels = []
        
        for col in prioritized_options:
            if col in detected_scores:
                # Add description for conservation scores
                description = detected_scores[col]
                label = f"{col} ({description})"
                score_column_labels.append(label)
                score_column_options.append(col)
            else:
                # Regular numeric column
                score_column_labels.append(col)
                score_column_options.append(col)
        
        # Get current selection
        score_column_value = st.session_state.get('score_column')
        score_column_index = score_column_options.index(score_column_value) if score_column_value in score_column_options else 0
        
        # Create selectbox with labels but return actual column names
        selected_label = st.sidebar.selectbox(
            "Select Conservation Score Column",
            options=score_column_labels,
            index=score_column_index,
            help="Conservation scores are automatically detected and prioritized"
        )
        
        # Get the actual column name from the selected label
        score_column = score_column_options[score_column_labels.index(selected_label)]
        st.session_state['score_column'] = score_column
        
        # Show description for selected score with enhanced info
        if score_column in detected_scores:
            st.sidebar.info(f"📋 **{score_column}**: {detected_scores[score_column]}")
        elif score_column in all_numeric_cols:
            st.sidebar.info(f"📋 **{score_column}**: Numeric score column")
        
        # Enhanced score information expander
        with st.sidebar.expander("🧬 Score Information", expanded=False):
            try:
                plot_info = get_enhanced_plot_info(score_column)
                
                st.markdown(f"**Score Type:** {plot_info['detected_type'].title()}")
                st.markdown(f"**Data Type:** {plot_info['score_type'].title()}")
                
                if plot_info['typical_range'] != 'varies' and plot_info['typical_range'] != 'unknown':
                    st.markdown(f"**Typical Range:** {plot_info['typical_range']}")
                
                st.markdown(f"**Interpretation:** {plot_info['interpretation']}")
                
                st.markdown("**Recommended Plots:**")
                for i, plot in enumerate(plot_info['plots']):
                    if i == 0:
                        st.markdown(f"• **{plot}** (Primary recommendation)")
                    else:
                        st.markdown(f"• {plot}")
                        
            except Exception:
                st.markdown("*Score analysis not available*")
        
        # Multi-score analysis expander (if multiple conservation scores detected)
        if len(detected_scores) > 1:
            with st.sidebar.expander("🔬 Multi-Score Analysis", expanded=False):
                st.markdown("**Available Conservation Scores:**")
                for col, desc in detected_scores.items():
                    st.markdown(f"• **{col}**: {desc}")
                
                st.markdown("---")
                st.markdown("**Correlation Analysis:**")
                
                # Calculate correlations between conservation scores
                conservation_cols = list(detected_scores.keys())
                if len(conservation_cols) > 1:
                    corr_matrix = data[conservation_cols].corr(method='pearson')
                    
                    # Create correlation heatmap
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    from io import BytesIO
                    import base64
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                    plt.title('Conservation Score Correlations', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    
                    # Convert to base64 for display
                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    corr_img = base64.b64encode(buf.getvalue()).decode('utf-8')
                    
                    st.markdown(f"<img src='data:image/png;base64,{corr_img}' style='width:100%;border-radius:8px;'>", 
                               unsafe_allow_html=True)
                    
                    # Summary statistics
                    st.markdown("**Summary Statistics:**")
                    summary_stats = data[conservation_cols].describe().round(3)
                    st.dataframe(summary_stats, use_container_width=True)
                else:
                    st.info("Only one conservation score detected. Correlation analysis requires multiple scores.")
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
            data_issues.append(f"Small groups (n<3): {', '.join([f'{g}({n})' for g, n in small_groups.items()])}")
        if n_groups < 2:
            data_issues.append("Need at least 2 groups for comparison")
            
        if data_issues:
            with st.sidebar.expander("⚠️ Data Quality Issues", expanded=True):
                for issue in data_issues:
                    st.warning(issue)
                st.info("💡 Consider cleaning your data or adjusting group mappings above.")
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
        st.sidebar.subheader("🎨 Visualization Options")
        # Get recommended plot types and tooltip
        recommended_plots, tooltip_text = suggest_plot_types(score_column)
        st.sidebar.markdown("**Recommended plot types (auto-detected):** " + ", ".join([f"`{plot}`" for plot in recommended_plots]))
        
        # Enhanced recommendation explanation
        with st.sidebar.expander("💡 Why these plots?", expanded=False):
            st.markdown(tooltip_text)
            
            # Show plot-specific explanations
            try:
                for plot in recommended_plots:
                    plot_explanation = score_plot_mapper.get_plot_explanation(score_column, plot)
                    st.markdown(f"**{plot.title()}:** {plot_explanation}")
            except:
                pass

        # Show all plot options, recommended first
        core_plot_options = ["boxplot", "violin", "swarm", "heatmap", "barplot", "histogram", "kde"]
        advanced_plot_options = ["pairplot"]  # Removed correlation from main options
        
        all_plot_options = recommended_plots + [opt for opt in core_plot_options if opt not in recommended_plots] + advanced_plot_options
        
        # Get current plot type selection with key for session state
        current_plot_type = st.session_state.get('plot_type', all_plot_options[0])
        plot_type_index = all_plot_options.index(current_plot_type) if current_plot_type in all_plot_options else 0
        
        plot_type = st.sidebar.selectbox(
            "📈 Plot Type",
            options=all_plot_options,
            index=plot_type_index,
            key='plot_type',
            help="Recommended options are listed first."
        )
        
        palette_options = [
            "Set1", "Set2", "Set3", "tab10", "Dark2", "Pastel1", "Accent",
            "viridis", "plasma", "magma", "inferno", "cividis", "colorblind"
        ]
        
        # Get current color palette selection with key for session state
        current_palette = st.session_state.get('color_palette', 'Set3')
        palette_index = palette_options.index(current_palette) if current_palette in palette_options else 2  # Set3 is at index 2
        
        color_palette = st.sidebar.selectbox(
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
        fig, ax = plt.subplots(figsize=(4, 0.5))
        for i, color in enumerate(preview_colors):
            ax.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color))
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        palette_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        st.sidebar.markdown(f"<b>Palette Preview:</b><br><img src='data:image/png;base64,{palette_img}' style='width:100%;height:30px;border-radius:6px;'>", unsafe_allow_html=True)
        
        # Interactive mode with session state
        interactive_mode = st.sidebar.checkbox(
            "🔄 Interactive Plots",
            value=st.session_state.get('interactive_mode', False),
            key='interactive_mode',
            help="Generate interactive plots with Plotly"
        )
        
        # Show statistics with session state
        show_statistics = st.sidebar.checkbox(
            "📊 Show Statistical Results",
            value=st.session_state.get('show_statistics', True),
            key='show_statistics',
            help="Display statistical analysis results"
        )
        
        # Advanced analysis options - simplified
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔬 Advanced Analysis")
        
        # Only keep the most useful advanced features
        use_multi_score = st.sidebar.checkbox(
            "🔬 Multi-Score Analysis",
            value=False,
            help="Compare multiple conservation scores (PhyloP, GERP, etc.)"
        )
        
        # Combine advanced stats into one expandable section
        with st.sidebar.expander("🧮 Statistical Options", expanded=False):
            advanced_stats_level = st.selectbox(
                "Statistical Analysis Level",
                options=["Basic", "Detailed", "Expert"],
                index=0,
                help="Choose the level of statistical analysis detail"
            )
            
            # Set variables based on selected level
            use_advanced_stats = advanced_stats_level in ["Detailed", "Expert"]
            show_assumptions = advanced_stats_level == "Expert"
            show_effect_sizes = advanced_stats_level in ["Detailed", "Expert"]
            show_posthoc = advanced_stats_level == "Expert"
        
        # Simplified visualization options
        with st.sidebar.expander("🎨 Visualization Options", expanded=False):
            show_additional_plots = st.checkbox(
                "Additional Plot Types",
                value=False,
                help="Show correlation matrix and pairwise plots"
            )
        
        # Apply advanced grouping - simplified to basic filtering only
        processed_data = data.copy()
        final_group_column = group_column
        
        # Multi-score analysis
        if use_multi_score:
            st.markdown("---")
            st.markdown("### 🔬 Multi-Score Conservation Analysis")
            multi_score_results = create_multi_score_analysis_ui(data, group_column)
            
            # If multi-score analysis was performed, we can show additional insights
            if multi_score_results:
                st.session_state['multi_score_results'] = multi_score_results
            
            st.markdown("---")
        
        # Advanced statistics analysis - conditional based on user selection
        if use_advanced_stats:
            st.markdown("---")
            st.markdown("### 🧮 Detailed Statistical Analysis")
            
            # Create simplified version of advanced stats
            if advanced_stats_level == "Expert":
                # Full advanced stats UI
                advanced_stats_results = create_advanced_statistics_ui(data, score_column, group_column)
            else:
                # Simplified version - create a basic version
                from src.analysis import perform_statistical_analysis
                basic_stats = perform_statistical_analysis(data, score_column, group_column)
                
                if basic_stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Kruskal-Wallis H", f"{basic_stats['h_statistic']:.4f}")
                        st.metric("p-value", f"{basic_stats['p_value']:.6f}")
                    with col2:
                        st.metric("Significant?", "✅ Yes" if basic_stats['significant'] else "❌ No")
                        st.metric("Groups", basic_stats['n_groups'])
                    
                    if show_effect_sizes:
                        # Calculate basic effect size (eta-squared approximation)
                        from scipy.stats import kruskal
                        n_total = len(data.dropna(subset=[score_column, group_column]))
                        k_groups = data[group_column].nunique()
                        eta_squared_approx = (basic_stats['h_statistic'] - k_groups + 1) / (n_total - k_groups)
                        st.metric("Effect Size (η²)", f"{eta_squared_approx:.4f}")
                
                advanced_stats_results = None
            
            # Store results for potential export
            if advanced_stats_results:
                st.session_state['advanced_stats_results'] = advanced_stats_results
            
            st.markdown("---")
        # Analysis button only enabled if data and columns are selected
        st.sidebar.markdown("---")
        
        analysis_enabled = data is not None and group_column and score_column
        run_analysis_clicked = st.sidebar.button("▶️ Run Analysis", type="primary", disabled=not analysis_enabled)
        analysis_results = None  # Always define before use
        # Scroll flag: run_analysis_clicked only
        if run_analysis_clicked:
            st.session_state['scroll_to_results'] = True
            # Defensive: check again if columns exist
            if score_column not in data.columns or group_column not in data.columns:
                st.error("❌ Selected columns not found in data!")
            elif not pd.api.types.is_numeric_dtype(data[score_column]):
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
                    performance_monitor.record_metric('analysis_time', analysis_time)
                else:
                    # Fallback without caching
                    analysis_results = perform_statistical_analysis(data, score_column, group_column)
                if analysis_results is None:
                    st.error("❌ Statistical analysis failed!")
                else:
                    # Step 3: Generate visualization
                    status_text.text("🎨 Creating visualization...")
                    progress_bar.progress(70)
                    
                    # Optimize visualization generation
                    if MODULES_AVAILABLE:
                        start_time = time.time()
                    
                    fig, plot_type_used = create_local_visualization(
                        data, score_column, group_column, plot_type, color_palette, interactive_mode
                    )
                    
                    if MODULES_AVAILABLE:
                        plot_time = time.time() - start_time
                        performance_monitor.record_metric('plot_generation_time', plot_time)
                    
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
                    else:
                        col1, col2 = st.columns([0.1, 1.9], gap="large")
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
                                    st.dataframe(dunn_results, use_container_width=True)
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
                    with col2:
                        # Only show plot if fig is valid
                        if fig is not None:
                            st.markdown("<div id='results_plot'></div>", unsafe_allow_html=True)
                            if plot_type_used == "interactive":
                                st.plotly_chart(fig, use_container_width=True)
                                # --- Plotly image download buttons ---
                                # Only HTML download is offered for interactive plots; PNG/SVG info messages removed
                                # ...existing code...
                            else:
                                try:
                                    st.pyplot(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying plot: {e}")
                            
                            # Additional plots if enabled
                            if show_additional_plots:
                                st.markdown("### 📊 Additional Visualizations")
                                
                                # Get numeric columns for analysis
                                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                                conservation_cols = [col for col in num_cols if any(keyword in col.lower() 
                                                   for keyword in ['phylop', 'phastcons', 'gerp', 'conservation', 'cadd', 'revel'])]
                                
                                # Smart correlation matrix - show different types based on available data
                                if len(conservation_cols) > 1:
                                    # Conservation-focused correlation
                                    st.markdown("**� Conservation Score Correlations**")
                                    st.caption(f"Analyzing {len(conservation_cols)} conservation scores")
                                    
                                    corr_matrix = data[conservation_cols].corr()
                                    
                                    if interactive_mode:
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
            st.markdown("---")
            st.subheader("🔍 Additional Analysis")
            
            # Summary statistics
            with st.expander("📊 Detailed Summary Statistics"):
                st.write("**Overall Statistics:**")
                overall_stats = data[score_column].describe()
                st.dataframe(overall_stats.to_frame().T, use_container_width=True)
                
                st.write("**Group Comparisons:**")
                
                # Use fast group statistics for better performance
                if MODULES_AVAILABLE and len(data) > 1000:
                    group_stats = get_fast_group_statistics(data, score_column, group_column)
                else:
                    group_stats = data.groupby(group_column)[score_column].describe()
                
                st.dataframe(group_stats, use_container_width=True)
                
                # --- Data Preview with Pagination for Large Datasets ---
                st.write("**📋 Data Preview:**")
                if len(data) > 1000 and MODULES_AVAILABLE:
                    # Use lazy loading for large datasets
                    st.info(f"� Large dataset ({len(data):,} rows) - showing paginated view")
                    
                    # Pagination controls
                    page_size = st.selectbox("Rows per page:", [50, 100, 200, 500], index=1)
                    total_pages = (len(data) + page_size - 1) // page_size
                    page = st.number_input("Page:", min_value=1, max_value=total_pages, value=1) - 1
                    
                    # Get paginated data
                    paginated_data, pagination_info = lazy_loader.get_paginated_data(data, page, page_size)
                    
                    st.write(f"Showing rows {pagination_info['start_idx']+1}-{pagination_info['end_idx']} of {pagination_info['total_rows']:,}")
                    st.dataframe(paginated_data, use_container_width=True, height=300)
                else:
                    # Show all data for smaller datasets
                    display_data = data.head(500) if len(data) > 500 else data
                    if len(data) > 500:
                        st.info(f"Showing first 500 rows of {len(data):,} total rows")
                    st.dataframe(display_data, use_container_width=True, height=300)
                
                # --- Correlation analysis ---
                st.write("**📈 Correlation Analysis (numeric columns):**")
                num_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
                if len(num_cols) > 1:
                    corr_matrix = data[num_cols].corr(method='pearson')
                    st.dataframe(corr_matrix, use_container_width=True)
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
                    sns.heatmap(corr_matrix, annot=True, cmap='viridis', ax=ax_corr)
                    st.pyplot(fig_corr)
            # Data export with organized interface
            if analysis_results is not None:
                with st.expander("📥 Export Results & Data"):
                    import json
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
                            dunn_csv = io.StringIO()
                            if isinstance(dunn_results.index, pd.MultiIndex):
                                dunn_results_reset = dunn_results.reset_index()
                            else:
                                dunn_results_reset = dunn_results
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
                        # Plotly interactive
                        if fig is not None and plot_type_used == "interactive":
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
                        # Matplotlib static
                        elif fig is not None and plot_type_used == "static":
                            try:
                                static_png = io.BytesIO()
                                fig.savefig(static_png, format="png", bbox_inches='tight', dpi=300)
                                static_png.seek(0)
                                st.download_button(
                                    label="Download Plot as PNG",
                                    data=static_png,
                                    file_name="conservation_plot.png",
                                    mime="image/png",
                                    key="download_png_static"
                                )
                            except Exception as e:
                                st.error(f"Static PNG export failed: {e}")
                            try:
                                static_svg = io.BytesIO()
                                fig.savefig(static_svg, format="svg", bbox_inches='tight', dpi=300)
                                static_svg.seek(0)
                                st.download_button(
                                    label="Download Plot as SVG (Publication Quality)",
                                    data=static_svg,
                                    file_name="conservation_plot.svg",
                                    mime="image/svg+xml",
                                    key="download_svg_static"
                                )
                            except Exception as e:
                                st.error(f"Static SVG export failed: {e}")

if __name__ == "__main__":
    main()
