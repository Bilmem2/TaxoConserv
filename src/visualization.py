"""
TaxoConserv - Visualization Module

Copyright 2025 Can Sevilmiş
Licensed under Apache License 2.0
For research and educational purposes only.

Unified visualization module for TaxoConserv.
All plot functions are routed through create_visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import logging
from typing import Optional
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Color palettes for consistent styling
COLOR_PALETTES = {
    "colorblind": ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"],
    "Set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"],
    "Set2": ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"],
    "Set3": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5"],
    "viridis": ["#440154", "#31688e", "#35b779", "#fde725"],
    "plasma": ["#0d0887", "#6a00a8", "#b12a90", "#e16462"],
    "magma": ["#000004", "#721f81", "#b73779", "#fcfdbf"],
    "inferno": ["#000004", "#781c6d", "#ed6925", "#fcffa4"]
}

try:
    from .utils import get_palette_for_groups, _setup_plot_style, _customize_plot, _save_plot
except ImportError:
    # Fallback implementations if utils module not available
    def get_palette_for_groups(palette_name_or_colors, n_groups: int) -> list:
        """Get color palette for the specified number of groups."""
        if isinstance(palette_name_or_colors, list):
            colors = palette_name_or_colors
        elif isinstance(palette_name_or_colors, str) and palette_name_or_colors in COLOR_PALETTES:
            colors = COLOR_PALETTES[palette_name_or_colors]
        else:
            # Fallback to default palette
            colors = COLOR_PALETTES["colorblind"]
        
        # Repeat colors if needed
        if n_groups > len(colors):
            colors = colors * (n_groups // len(colors) + 1)
        
        return colors[:n_groups]
    
    def _setup_plot_style(color_palette: str):
        """Setup plot style and color palette."""
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11
        })
    
    def _customize_plot(ax, title: Optional[str], group_column: str, score_column: str):
        """Customize plot appearance."""
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3)
    
    def _save_plot(fig, output_file: str, dpi: int, plot_df: pd.DataFrame, 
                   group_column: str, score_column: str) -> str:
        """Save plot with metadata and statistics."""
        plt.tight_layout()
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        return output_file

logger = logging.getLogger("taxoconserv.visualization")

def create_visualization(
    df: pd.DataFrame,
    plot_type: str = "boxplot",
    group_column: str = "taxon_group",
    score_column: str = "conservation_score",
    output_path: str = "conservation_plot",
    color_palette: str = "colorblind",
    output_format: str = "png",
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    dpi: int = 300,
    interactive: bool = False,
    show_gene: bool = False,
    gene_info: Optional[dict] = None,
    show_pvalue: bool = False,
    test_results: Optional[dict] = None,
    plot_all: bool = False
) -> str | dict:
    # Title assignment only if not provided
    if title is None:
        title = f"{plot_type.title()} of {score_column.replace('_', ' ').title()}"

    # Interaktif modda plot üretimi
    if interactive:
        # Only support: boxplot, violin, swarm, heatmap, barplot, histogram, kde, pairplot, correlation
        if plot_type == "boxplot":
            return generate_interactive_plot(df, "boxplot", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "violin":
            return generate_interactive_plot(df, "violin", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "swarm":
            return generate_interactive_plot(df, "swarm", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "heatmap":
            return generate_interactive_plot(df, "heatmap", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "barplot":
            return generate_interactive_plot(df, "barplot", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "histogram":
            return generate_interactive_plot(df, "histogram", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "kde":
            return generate_interactive_plot(df, "kde", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "pairplot":
            return generate_interactive_plot(df, "pairplot", group_column, score_column, output_path, color_palette, title, figsize)
        elif plot_type == "correlation":
            return generate_interactive_plot(df, "correlation", group_column, score_column, output_path, color_palette, title, figsize)
        else:
            logger.warning(f"Unknown or unsupported plot type for interactive mode: {plot_type}. Falling back to boxplot.")
            return generate_interactive_plot(df, "boxplot", group_column, score_column, output_path, color_palette, title, figsize)

    # Statik modda plot üretimi
    if plot_type == "boxplot":
        return generate_boxplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "violin":
        return generate_violin_plot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "swarm":
        return generate_swarm_plot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "heatmap":
        return generate_heatmap(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "barplot":
        return generate_barplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "histogram":
        return generate_histogram(df, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "kde":
        return generate_kde_plot(df, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "pairplot":
        return generate_pairplot(df, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "correlation":
        return generate_correlation_matrix(df, output_path, color_palette, output_format, figsize, title, dpi)
    else:
        logger.warning(f"Unknown or unsupported plot type: {plot_type}. Falling back to boxplot.")
        return generate_boxplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)

    # Tek bir grafik üretilecekse
    if plot_type == "boxplot":
        return generate_boxplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "violin":
        return generate_violin_plot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "swarm":
        return generate_swarm_plot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "heatmap":
        return generate_heatmap(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "barplot":
        agg = "sum" if title and "sum" in title.lower() else "mean"
        return generate_barplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi, agg=agg)
    elif plot_type == "histogram":
        return generate_histogram(df, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "kde":
        return generate_kde_plot(df, score_column, output_path, color_palette, output_format, figsize, title, dpi)
    elif plot_type == "pairplot":
        result = generate_pairplot(df, output_path, color_palette, output_format, figsize, title, dpi)
        if not result or result.endswith("_failed.png"):
            logger.error("Pairplot generation failed.")
            return "pairplot_failed.png"
        return result
    elif plot_type == "correlation":
        result = generate_correlation_matrix(df, output_path, color_palette, output_format, figsize, title, dpi)
        if not result or result.endswith("_failed.png"):
            logger.error("Correlation matrix generation failed.")
            return "correlation_matrix_failed.png"
        return result
    else:
        logger.warning(f"Unknown plot type: {plot_type}. Falling back to boxplot.")
        return generate_boxplot(df, group_column, score_column, output_path, color_palette, output_format, figsize, title, dpi)

def generate_barplot(df: pd.DataFrame,
                    group_column: str = "taxon_group",
                    score_column: str = "conservation_score",
                    output_path: str = "conservation_barplot",
                    color_palette: str = "colorblind",
                    output_format: str = "png",
                    figsize: tuple = (10, 6),
                    title: Optional[str] = None,
                    dpi: int = 300,
                    agg: str = "mean") -> str:
    """Generate barplot of group means or sums."""
    logger.info(f"📊 Generating barplot for {group_column} vs {score_column} ({agg})...")
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    if agg == "sum":
        values = plot_df.groupby(group_column)[score_column].sum().sort_values()
        ylabel = f"Sum {score_column.replace('_', ' ').title()}"
    else:
        values = plot_df.groupby(group_column)[score_column].mean().sort_values()
        ylabel = f"Mean {score_column.replace('_', ' ').title()}"
    n_groups = values.shape[0]
    palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
    fig, ax = plt.subplots(figsize=figsize)
    values.plot(kind="bar", color=palette_colors, ax=ax)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
    n_total = len(df)
    n_groups = df[group_column].nunique()
    mean = df[score_column].mean()
    std = df[score_column].std()
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        ax.set_title(f"{ylabel} by {group_column.replace('_', ' ').title()}\nGroups: {n_groups}, Samples: {n_total}\nMean: {mean:.3f}, Std: {std:.3f}", fontsize=14, fontweight='bold', pad=20)
    # Annotate bars with value
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    caption = f"Bars show the {agg} {score_column} for each {group_column}."
    fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
    plt.tight_layout()
    output_file = f"{output_path}.{output_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"✅ Barplot saved to: {output_file}")
    return output_file
def generate_stripplot(df: pd.DataFrame,
                      group_column: str = "taxon_group",
                      score_column: str = "conservation_score",
                      output_path: str = "conservation_stripplot",
                      color_palette: str = "colorblind",
                      output_format: str = "png",
                      figsize: tuple = (10, 6),
                      title: Optional[str] = None,
                      dpi: int = 300) -> str:
    """Generate stripplot (fast alternative to swarmplot)."""
    logger.info(f"📊 Generating stripplot for {group_column} vs {score_column}...")
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    _setup_plot_style(color_palette)
    # Modern ve okunaklı font ayarları
    plt.rcParams.update({
        'font.family': 'DejaVu Sans, Arial',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11
    })
    # İki sütunlu düzen: Solda grafik, sağda özet ve açıklama
    fig, (ax, info_ax) = plt.subplots(1, 2, figsize=(figsize[0]+3, figsize[1]), gridspec_kw={'width_ratios': [4, 1]})
    n_groups = plot_df[group_column].nunique()
    palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
    sns.stripplot(data=plot_df, x=group_column, y=score_column,
                 palette=palette_colors, size=3, ax=ax, jitter=True)
    # Outlier annotation (sade, gri, kutusuz, veri noktasının hemen üstünde)
    for i, group in enumerate(sorted(plot_df[group_column].unique())):
        group_scores = plot_df[plot_df[group_column] == group][score_column]
        q1 = group_scores.quantile(0.25)
        q3 = group_scores.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = group_scores[(group_scores < lower) | (group_scores > upper)]
        for outlier in outliers:
            ax.scatter(i, outlier, color='#e74c3c', edgecolor='#222', s=40, zorder=5)
            ax.annotate("outlier", (i, outlier), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='#888', fontweight='normal')
    # Sade başlık ve eksenler
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    else:
        ax.set_title(f"{score_column.replace('_', ' ').title()} by {group_column.replace('_', ' ').title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
    plt.sca(ax)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    # Sağ panel: özet ve açıklama
    info_ax.axis('off')
    n_total = len(plot_df)
    mean = plot_df[score_column].mean()
    std = plot_df[score_column].std()
    stats_text = f"Groups: {n_groups}\nSamples: {n_total}\nMean: {mean:.2f}\nStd: {std:.2f}"
    info_ax.text(0, 1, "Summary", fontsize=12, fontweight='bold', color='#222', va='top', ha='left', transform=info_ax.transAxes)
    info_ax.text(0, 0.85, stats_text, fontsize=10, color='#444', va='top', ha='left', transform=info_ax.transAxes)
    plt.tight_layout()
    output_file = f"{output_path}.{output_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"✅ Stripplot saved to: {output_file}")
    return output_file
def generate_pairplot(df: pd.DataFrame,
                     output_path: str = "conservation_pairplot",
                     color_palette: str = "colorblind",
                     output_format: str = "png",
                     figsize: tuple = (10, 6),
                     title: Optional[str] = None,
                     dpi: int = 300) -> str:
    """Generate pairplot for multiple numeric columns."""
    logger.info(f"🔗 Generating pairplot for numeric columns...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        logger.warning("Not enough numeric columns for pairplot.")
        # Return a dummy file path to avoid None
        return "pairplot_failed.png"
    _setup_plot_style(color_palette)
    # Validate palette
    try:
        available_palettes = list(sns.palettes.SEABORN_PALETTES.keys())
        if color_palette not in available_palettes and color_palette not in plt.colormaps():
            logger.warning(f"Palette '{color_palette}' not found, using 'colorblind'.")
            color_palette = "colorblind"
        pairplot_obj = sns.pairplot(df[num_cols], diag_kind="kde", palette=color_palette)
        fig = getattr(pairplot_obj, 'figure', None)
        if fig is None:
            fig = plt.gcf()
        n_total = len(df)
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        else:
            fig.suptitle(f"Pairplot of Numeric Features\nSamples: {n_total}", fontsize=16, fontweight='bold', y=1.02)
        caption = f"Pairplot shows relationships between all numeric features."
        fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
        try:
            fig.tight_layout()
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}")
        output_file = f"{output_path}.{output_format}"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Pairplot saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating pairplot: {e}")
        plt.close('all')
        return "pairplot_failed.png"
def generate_correlation_matrix(df: pd.DataFrame,
                               output_path: str = "correlation_matrix",
                               color_palette: str = "viridis",
                               output_format: str = "png",
                               figsize: tuple = (10, 8),
                               title: Optional[str] = None,
                               dpi: int = 300) -> str:
    """Generate correlation matrix heatmap for numeric columns."""
    logger.info(f"🔗 Generating correlation matrix heatmap...")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        logger.warning("Not enough numeric columns for correlation matrix.")
        # Return a dummy file path to avoid None
        return "correlation_matrix_failed.png"
    try:
        corr = df[num_cols].corr()
        _setup_plot_style(color_palette)
        # Validate palette
        if color_palette not in plt.colormaps():
            logger.warning(f"Palette '{color_palette}' not found, using 'viridis'.")
            color_palette = "viridis"
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, cmap=color_palette, fmt=".2f", ax=ax)
        n_total = len(df)
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        else:
            ax.set_title(f"Correlation Matrix\nSamples: {n_total}", fontsize=16, fontweight='bold', pad=20)
        caption = f"Correlation matrix shows pairwise correlations between numeric features."
        fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
        try:
            plt.tight_layout()
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}")
        output_file = f"{output_path}.{output_format}"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Correlation matrix saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating correlation matrix: {e}")
        plt.close('all')
        return "correlation_matrix_failed.png"

def generate_histogram(df: pd.DataFrame,
                      score_column: str = "conservation_score",
                      output_path: str = "conservation_histogram",
                      color_palette: str = "colorblind",
                      output_format: str = "png",
                      figsize: tuple = (10, 6),
                      title: Optional[str] = None,
                      dpi: int = 300) -> str:
    """Generate histogram of score distribution."""
    logger.info(f"📊 Generating histogram for {score_column}...")
    plot_df = df[[score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    try:
        fig, ax = plt.subplots(figsize=figsize)
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), 1)
        ax.hist(plot_df[score_column], bins=20, color=palette_colors[0], edgecolor='black', alpha=0.8)
        n_total = len(df)
        mean = df[score_column].mean()
        median = df[score_column].median()
        std = df[score_column].std()
        ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f"Distribution of {score_column.replace('_', ' ').title()}\nSamples: {n_total}\nMean: {mean:.3f}, Median: {median:.3f}, Std: {std:.3f}", fontsize=14, fontweight='bold', pad=20)
        caption = f"Histogram shows the distribution of {score_column} across all samples."
        fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
        plt.tight_layout()
        output_file = f"{output_path}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Histogram saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating histogram: {e}")
        plt.close('all')
        return ""


def generate_boxplot(df: pd.DataFrame, 
                    group_column: str = "taxon_group",
                    score_column: str = "conservation_score",
                    output_path: str = "conservation_boxplot",
                    color_palette: str = "colorblind",
                    output_format: str = "png",
                    figsize: tuple = (10, 6),
                    title: Optional[str] = None,
                    dpi: int = 300) -> str:
    """
    Generate enhanced boxplot visualization of conservation scores by taxonomic groups.
    
    Args:
        df: Input DataFrame with conservation scores
        group_column: Column name for taxonomic groups
        score_column: Column name for conservation scores
        output_path: Base path to save the plot (without extension)
        color_palette: Color palette for the plot
        output_format: Output format ('png', 'pdf', 'svg')
        figsize: Figure size in inches (width, height)
        title: Custom title for the plot
        dpi: Resolution for the saved image
        
    Returns:
        Path to the saved plot file
    """
    
    logger.info(f"📊 Generating boxplot for {group_column} vs {score_column}...")
    
    # Validate input
    if df.empty:
        logger.warning("⚠️ Input DataFrame is empty.")
        return ""
    
    # Create clean DataFrame for plotting
    plot_df = df[[group_column, score_column]].copy().dropna()
    
    if plot_df.empty:
        logger.warning("⚠️ No valid data after removing missing values.")
        return ""
    
    unique_groups = plot_df[group_column].unique()
    logger.info(f"📈 Creating boxplot for {len(unique_groups)} groups: {sorted(unique_groups)}")
    
    try:
        # Validate columns
        for col in [group_column, score_column]:
            if col not in plot_df.columns:
                logger.error(f"Missing required column: {col}")
                return "boxplot_failed.png"
        # İki sütunlu düzen: Solda grafik, sağda özet ve açıklama
        _setup_plot_style(color_palette)
        fig, (ax, info_ax) = plt.subplots(1, 2, figsize=(figsize[0]+3, figsize[1]), gridspec_kw={'width_ratios': [4, 1]})
        n_groups = plot_df[group_column].nunique()
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
        sns.boxplot(data=plot_df, x=group_column, y=score_column,
                   palette=palette_colors, ax=ax)
        # Sade başlık ve eksenler
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        else:
            ax.set_title(f"Conservation Score Distribution by Taxonomic Group", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        plt.sca(ax)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        # Sağ panel: özet ve açıklama
        info_ax.axis('off')
        n_total = len(plot_df)
        mean = plot_df[score_column].mean()
        median = plot_df[score_column].median()
        std = plot_df[score_column].std()
        stats_text = f"Groups: {n_groups}\nSamples: {n_total}\nMean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}"
        info_ax.text(0, 1, "Summary", fontsize=12, fontweight='bold', color='#222', va='top', ha='left', transform=info_ax.transAxes)
        info_ax.text(0, 0.85, stats_text, fontsize=10, color='#444', va='top', ha='left', transform=info_ax.transAxes)
        try:
            plt.tight_layout()
        except Exception as e:
            logger.warning(f"tight_layout failed: {e}")
        output_file = f"{output_path}.{output_format}"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Boxplot saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating boxplot: {e}")
        plt.close('all')
        return "boxplot_failed.png"
    """
    Generate boxplot visualization of conservation scores by taxonomic groups.
    
    Args:
        df (pd.DataFrame): Input DataFrame with conservation scores
        group_column (str): Column name for taxonomic groups
        score_column (str): Column name for conservation scores
        output_path (str): Path to save the plot (PNG format)
        palette (str): Color palette for the plot (default: "Set2")
        figsize (tuple): Figure size in inches (width, height)
        title (str, optional): Custom title for the plot
        dpi (int): Resolution for the saved image
        
    Returns:
        str: Path to the saved plot file
        
    Raises:
        ValueError: If required columns are missing or DataFrame is empty
        Exception: If plot generation or saving fails
    """
    
    logger.info(f"📊 Generating boxplot for {group_column} vs {score_column}...")
    
    # Validate input DataFrame
    if df.empty:
        logger.warning("⚠️ Input DataFrame is empty. Cannot generate boxplot.")
        return ""
    
    # Check required columns
    missing_columns = [col for col in [group_column, score_column] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ Missing required columns: {missing_columns}")
    
    # Create clean DataFrame for plotting
    plot_df = df[[group_column, score_column]].copy()
    
    # Remove rows with missing values
    plot_df = plot_df.dropna()
    
    if plot_df.empty:
        logger.warning("⚠️ No valid data remaining after removing missing values.")
        return ""
    
    # Check if we have multiple groups
    unique_groups = plot_df[group_column].unique()
    if len(unique_groups) < 2:
        logger.warning(f"⚠️ Only {len(unique_groups)} unique group(s) found. Boxplot may not be meaningful.")
    
    logger.info(f"📈 Creating boxplot for {len(unique_groups)} groups: {sorted(unique_groups)}")
    
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create boxplot
        sns.boxplot(data=plot_df, 
                   x=group_column, 
                   y=score_column,
                   hue=group_column,
                   palette=palette,
                   legend=False,
                   ax=ax)
        
        # Customize plot appearance
        if title is None:
            title = f"Conservation Score Distribution by {group_column.replace('_', ' ').title()}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add sample size annotations
        _add_sample_size_annotations(ax, plot_df, group_column, score_column)
        
        # Improve layout
        plt.tight_layout()
        
        # Save plot
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_path_obj, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        logger.info(f"✅ Boxplot saved to: {output_path_obj}")
        
        # Display basic statistics
        _log_plot_statistics(plot_df, group_column, score_column)
        
        # Close figure to free memory
        plt.close(fig)
        
        return str(output_path_obj)
        
    except Exception as e:
        logger.error(f"❌ Error generating boxplot: {str(e)}")
        plt.close('all')  # Clean up any open figures
        raise Exception(f"Plot generation failed: {str(e)}")


def generate_summary_plot(df: pd.DataFrame, 
                         group_column: str = "taxon_group",
                         score_column: str = "conservation_score",
                         output_path: str = "taxoconserv_summary.png",
                         palette: str = "Set2") -> str:
    """
    Generate a comprehensive summary plot with boxplot and statistics.
    
    Args:
        df (pd.DataFrame): Input DataFrame with conservation scores
        group_column (str): Column name for taxonomic groups
        score_column (str): Column name for conservation scores
        output_path (str): Path to save the plot
        palette (str): Color palette for the plot
        
    Returns:
        str: Path to the saved plot file
    """
    
    logger.info("📊 Generating comprehensive summary plot...")
    
    # Validate input
    if df.empty:
        logger.warning("⚠️ Input DataFrame is empty.")
        return ""
    
    plot_df = df[[group_column, score_column]].dropna()
    
    if plot_df.empty:
        logger.warning("⚠️ No valid data for plotting.")
        return ""
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Boxplot
        sns.boxplot(data=plot_df, x=group_column, y=score_column, 
                   hue=group_column, palette=palette, legend=False, ax=ax1)
        ax1.set_title("Distribution by Taxonomic Group", fontweight='bold')
        ax1.set_xlabel(group_column.replace('_', ' ').title())
        ax1.set_ylabel(score_column.replace('_', ' ').title())
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Violin plot for additional detail
        sns.violinplot(data=plot_df, x=group_column, y=score_column, 
                      hue=group_column, palette=palette, legend=False, ax=ax2, inner='box')
        ax2.set_title("Density Distribution by Taxonomic Group", fontweight='bold')
        ax2.set_xlabel(group_column.replace('_', ' ').title())
        ax2.set_ylabel(score_column.replace('_', ' ').title())
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle("TaxoConserv: Conservation Score Analysis", 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path_obj, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        logger.info(f"✅ Summary plot saved to: {output_path_obj}")
        
        plt.close(fig)
        return str(output_path_obj)
        
    except Exception as e:
        logger.error(f"❌ Error generating summary plot: {str(e)}")
        plt.close('all')
        raise Exception(f"Summary plot generation failed: {str(e)}")


def _add_sample_size_annotations(ax, df: pd.DataFrame, group_column: str, score_column: str) -> None:
    """Add sample size annotations to boxplot."""
    
    # Calculate sample sizes
    sample_sizes = df.groupby(group_column, observed=True)[score_column].count()
    
    # Add annotations
    for i, (group, size) in enumerate(sample_sizes.items()):
        ax.annotate(f'n={size}', 
                   xy=(i, ax.get_ylim()[0]), 
                   xytext=(i, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
                   ha='center', va='top',
                   fontsize=9, color='gray')


def _log_plot_statistics(df: pd.DataFrame, group_column: str, score_column: str) -> None:
    """Log basic statistics for the plot."""
    
    logger.info("📈 Plot Statistics:")
    
    # Overall statistics
    overall_stats = df[score_column].describe()
    logger.info(f"   Overall: n={len(df)}, mean={overall_stats['mean']:.3f}, "
               f"std={overall_stats['std']:.3f}")
    
    # Group statistics
    for group in sorted(df[group_column].unique()):
        group_data = df[df[group_column] == group][score_column]
        logger.info(f"   {group}: n={len(group_data)}, mean={group_data.mean():.3f}, "
                   f"median={group_data.median():.3f}, std={group_data.std():.3f}")


def customize_plot_style() -> None:
    """
    Set custom matplotlib style for consistent plot appearance.
    """
    
    plt.style.use('seaborn-v0_8')
    
    # Custom parameters
    custom_params = {
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.grid.alpha': 0.3,
        'grid.linestyle': '--',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    }
    
    plt.rcParams.update(custom_params)
    logger.info("🎨 Custom plot style applied")


def save_plot_with_metadata(fig, output_path: str, dpi: int = 300) -> None:
    """
    Save plot with metadata and proper formatting.
    
    Args:
        fig: Matplotlib figure object
        output_path (str): Path to save the plot
        dpi (int): Resolution for the saved image
    """
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    metadata = {
        'Title': 'TaxoConserv Analysis',
        'Author': 'TaxoConserv Tool',
        'Software': 'TaxoConserv v2.0.0',
        'Creation Time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    fig.savefig(output_path_obj, dpi=dpi, bbox_inches='tight', 
               facecolor='white', edgecolor='none', metadata=metadata)
    
    logger.info(f"📁 Plot saved with metadata to: {output_path_obj}")


def _setup_plot_style(color_palette: str):
    """Setup plot style and color palette."""
    sns.set_style("whitegrid")
    if color_palette == "colorblind":
        sns.set_palette("colorblind")
    else:
        sns.set_palette(color_palette)


def _customize_plot(ax, title: Optional[str], group_column: str, score_column: str):
    """Customize plot appearance."""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
    
    # Rotate x-axis labels if needed
    if len(ax.get_xticklabels()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)


def _save_plot(fig, output_file: str, dpi: int, plot_df: pd.DataFrame, 
               group_column: str, score_column: str) -> str:
    """Save plot with metadata and statistics."""
    
    # Log statistics
    logger.info("📈 Plot Statistics:")
    overall_stats = plot_df[score_column].describe()
    logger.info(f"   Overall: n={len(plot_df)}, mean={overall_stats['mean']:.3f}, std={overall_stats['std']:.3f}")
    
    for group in sorted(plot_df[group_column].unique()):
        group_data = plot_df[plot_df[group_column] == group][score_column]
        logger.info(f"   {group}: n={len(group_data)}, mean={group_data.mean():.3f}, median={group_data.median():.3f}, std={group_data.std():.3f}")
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Boxplot saved to: {output_file}")
    return output_file


def generate_violin_plot(df: pd.DataFrame,
                        group_column: str = "taxon_group",
                        score_column: str = "conservation_score", 
                        output_path: str = "conservation_violin",
                        color_palette: str = "colorblind",
                        output_format: str = "png",
                        figsize: tuple = (10, 6),
                        title: Optional[str] = None,
                        dpi: int = 300) -> str:
    """Generate violin plot showing score distributions."""
    
    logger.info(f"🎻 Generating violin plot for {group_column} vs {score_column}...")
    
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    
    try:
        _setup_plot_style(color_palette)
        # İki sütunlu düzen: Solda grafik, sağda özet ve açıklama
        fig, (ax, info_ax) = plt.subplots(1, 2, figsize=(figsize[0]+3, figsize[1]), gridspec_kw={'width_ratios': [4, 1]})
        n_groups = plot_df[group_column].nunique()
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
        sns.violinplot(data=plot_df, x=group_column, y=score_column,
                      hue=group_column, palette=palette_colors, legend=False,
                      inner="box", ax=ax)
        # Outlier noktalarını vurgula (sade, gri, kutusuz, veri noktasının hemen üstünde)
        for i, group in enumerate(sorted(plot_df[group_column].unique())):
            group_scores = plot_df[plot_df[group_column] == group][score_column]
            q1 = group_scores.quantile(0.25)
            q3 = group_scores.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = group_scores[(group_scores < lower) | (group_scores > upper)]
            for outlier in outliers:
                ax.scatter(i, outlier, color='#e74c3c', edgecolor='#222', s=40, zorder=5)
                ax.annotate("outlier", (i, outlier), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=8, color='#888', fontweight='normal')
        # Sade başlık ve eksenler
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        else:
            ax.set_title(f"{score_column.replace('_', ' ').title()} by {group_column.replace('_', ' ').title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        plt.sca(ax)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        # Sağ panel: özet ve açıklama
        info_ax.axis('off')
        n_total = len(plot_df)
        mean = plot_df[score_column].mean()
        median = plot_df[score_column].median()
        std = plot_df[score_column].std()
        stats_text = f"Groups: {n_groups}\nSamples: {n_total}\nMean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}"
        info_ax.text(0, 1, "Summary", fontsize=12, fontweight='bold', color='#222', va='top', ha='left', transform=info_ax.transAxes)
        info_ax.text(0, 0.85, stats_text, fontsize=10, color='#444', va='top', ha='left', transform=info_ax.transAxes)
        plt.tight_layout()
        output_file = f"{output_path}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Violin plot saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating violin plot: {e}")
        plt.close('all')
        return ""


def generate_swarm_plot(df: pd.DataFrame,
                       group_column: str = "taxon_group",
                       score_column: str = "conservation_score",
                       output_path: str = "conservation_swarm",
                       color_palette: str = "colorblind", 
                       output_format: str = "png",
                       figsize: tuple = (10, 6),
                       title: Optional[str] = None,
                       dpi: int = 300) -> str:
    """Generate swarm plot showing individual data points."""
    
    logger.info(f"🐝 Generating swarm plot for {group_column} vs {score_column}...")
    
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    
    # Check data size - swarm plots can be slow with large datasets
    if len(plot_df) > 1000:
        logger.warning(f"⚠️ Large dataset ({len(plot_df)} points). Consider sampling for better performance.")
        plot_df = plot_df.sample(n=1000, random_state=42)
    
    try:
        # İki sütunlu düzen: Solda grafik, sağda özet ve açıklama
        _setup_plot_style(color_palette)
        fig, (ax, info_ax) = plt.subplots(1, 2, figsize=(figsize[0]+3, figsize[1]), gridspec_kw={'width_ratios': [4, 1]})
        n_groups = plot_df[group_column].nunique()
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
        sns.swarmplot(data=plot_df, x=group_column, y=score_column,
                     palette=palette_colors,
                     size=3, ax=ax)
        # Sade başlık ve eksenler
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        else:
            ax.set_title(f"{score_column.replace('_', ' ').title()} by {group_column.replace('_', ' ').title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
        plt.sca(ax)
        plt.xticks(rotation=30, ha='right', fontsize=11)
        # Sağ panel: özet ve açıklama
        info_ax.axis('off')
        n_total = len(plot_df)
        mean = plot_df[score_column].mean()
        median = plot_df[score_column].median()
        std = plot_df[score_column].std()
        stats_text = f"Groups: {n_groups}\nSamples: {n_total}\nMean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}"
        info_ax.text(0, 1, "Summary", fontsize=12, fontweight='bold', color='#222', va='top', ha='left', transform=info_ax.transAxes)
        info_ax.text(0, 0.85, stats_text, fontsize=10, color='#444', va='top', ha='left', transform=info_ax.transAxes)
        plt.tight_layout()
        output_file = f"{output_path}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Swarm plot saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error generating swarm plot: {e}")
        return ""


def generate_heatmap(df: pd.DataFrame,
                    group_column: str = "taxon_group",
                    score_column: str = "conservation_score",
                    output_path: str = "conservation_heatmap",
                    color_palette: str = "viridis",
                    output_format: str = "png", 
                    figsize: tuple = (10, 6),
                    title: Optional[str] = None,
                    dpi: int = 300) -> str:
    """Generate heatmap of average scores by groups."""
    
    logger.info(f"🔥 Generating heatmap for {group_column} vs {score_column}...")
    
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        logger.warning("No data available for heatmap")
        return ""
    
    try:
        # Calculate summary statistics for heatmap
        heatmap_data = plot_df.groupby(group_column)[score_column].agg([
            'mean', 'median', 'std', 'count'
        ]).round(3)
        
        _setup_plot_style(color_palette)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Handle colormap selection safely
        try:
            # Try to use the palette as a matplotlib colormap first
            cmap = plt.get_cmap(color_palette)
        except Exception:
            # Fallback to viridis if the palette doesn't work as colormap
            try:
                # Try as seaborn palette
                palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "viridis"), 8)
                cmap = mcolors.ListedColormap(palette_colors)
            except Exception:
                # Ultimate fallback
                cmap = 'viridis'
        
        # Create heatmap
        sns.heatmap(heatmap_data.T, annot=True, cmap=cmap, 
                   fmt='.3f', cbar_kws={'label': 'Value'}, ax=ax)
        
        # Add title and labels
        n_total = len(plot_df)
        n_groups = plot_df[group_column].nunique()
        mean = plot_df[score_column].mean()
        std = plot_df[score_column].std()
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f"Statistical Summary: {score_column.replace('_', ' ').title()}\nGroups: {n_groups}, Samples: {n_total}\nMean: {mean:.3f}, Std: {std:.3f}", fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Statistics", fontsize=12)
        
        # Add caption
        caption = f"Heatmap shows the mean, median, std, and count of {score_column} for each {group_column}."
        fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        output_file = f"{output_path}.{output_format}"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Heatmap saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error generating heatmap: {e}")
        plt.close('all')
        return ""



def generate_kde_plot(df: pd.DataFrame,
                     score_column: str = "conservation_score",
                     output_path: str = "conservation_kde",
                     color_palette: str = "colorblind",
                     output_format: str = "png",
                     figsize: tuple = (10, 6),
                     title: Optional[str] = None,
                     dpi: int = 300) -> str:
    """Generate KDE (Kernel Density Estimate) plot for score distribution."""
    logger.info(f"📈 Generating KDE plot for {score_column}...")
    plot_df = df[[score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    try:
        _setup_plot_style(color_palette)
        fig, ax = plt.subplots(figsize=figsize)
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), 1)
        sns.kdeplot(x=plot_df[score_column], fill=True, color=palette_colors[0], alpha=0.7, ax=ax)
        n_total = len(plot_df)
        mean = plot_df[score_column].mean()
        median = plot_df[score_column].median()
        std = plot_df[score_column].std()
        ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f"KDE of {score_column.replace('_', ' ').title()}\nSamples: {n_total}\nMean: {mean:.3f}, Median: {median:.3f}, Std: {std:.3f}", fontsize=14, fontweight='bold', pad=20)
        caption = f"KDE plot shows the estimated density of {score_column} across all samples."
        fig.text(0.5, 0.01, caption, ha='center', fontsize=10, color='gray')
        plt.tight_layout()
        output_file = f"{output_path}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ KDE plot saved to: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"❌ Error generating KDE plot: {e}")
        plt.close('all')
        return ""

def generate_density_plot(df: pd.DataFrame,
                        score_column: str = "conservation_score",
                        output_path: str = "conservation_density",
                        color_palette: str = "colorblind",
                        output_format: str = "png",
                        figsize: tuple = (10, 6),
                        title: Optional[str] = None,
                        dpi: int = 300) -> str:
    """Generate density plot (grouped KDE) for score distribution by group."""
    logger.info(f"📈 Generating density plot for {score_column} by group...")
    
    # Try to find a group column
    group_column = None
    if "taxon_group" in df.columns:
        group_column = "taxon_group"
    else:
        # Find first categorical column
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() < 20:
                group_column = col
                break
    
    if group_column is None or group_column not in df.columns:
        logger.warning("No suitable group column found for density plot")
        return ""
    
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        return ""
    
    try:
        _setup_plot_style(color_palette)
        fig, ax = plt.subplots(figsize=figsize)
        
        n_groups = plot_df[group_column].nunique()
        palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)
        
        # Create density plot for each group
        for i, group in enumerate(sorted(plot_df[group_column].unique())):
            group_data = plot_df[plot_df[group_column] == group][score_column]
            sns.kdeplot(x=group_data, label=group, color=palette_colors[i], alpha=0.7, ax=ax)
        
        ax.set_xlabel(score_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            ax.set_title(f"Density Plot of {score_column.replace('_', ' ').title()} by {group_column.replace('_', ' ').title()}", fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(title=group_column.replace('_', ' ').title())
        plt.tight_layout()
        
        output_file = f"{output_path}.{output_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"✅ Density plot saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error generating density plot: {e}")
        plt.close('all')
        return ""


def generate_interactive_plot(
    df,
    plot_type: str,
    group_column: str,
    score_column: str,
    output_path: str,
    color_palette: str,
    title: str,
    figsize: tuple = (10, 6),
    theme: str = "light"
) -> str:
    """Generate interactive plot using Plotly with theme-aware colors."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Falling back to static plot.")
        return ""
    
    import plotly.express as px
    import plotly.graph_objects as go
    
    logger.info(f"📈 Generating interactive {plot_type} plot...")
    
    # Theme detection
    if theme.lower() == "dark":
        template = "plotly_dark"
        title_font_color = "white"
    else:
        template = "plotly_white"
        title_font_color = "black"

    # Title assignment
    if not title:
        title = f"{plot_type.title()} - Conservation Score Distribution"

    # Prepare data
    plot_df = df[[group_column, score_column]].copy().dropna()
    if plot_df.empty:
        logger.warning("No data available for interactive plot.")
        return ""
    
    # Get color palette
    n_groups = plot_df[group_column].nunique()
    palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)

    # Select plot type
    try:
        if plot_type == "boxplot":
            fig = px.box(plot_df, x=group_column, y=score_column, color=group_column,
                        color_discrete_sequence=palette_colors, template=template)
        elif plot_type == "violin":
            fig = px.violin(plot_df, x=group_column, y=score_column, color=group_column,
                           color_discrete_sequence=palette_colors, template=template)
        elif plot_type == "swarm":
            # Plotly doesn't have swarm plots, use strip plot
            fig = px.strip(plot_df, x=group_column, y=score_column, color=group_column,
                          color_discrete_sequence=palette_colors, template=template)
        elif plot_type == "histogram":
            fig = px.histogram(plot_df, x=score_column, color=group_column,
                              color_discrete_sequence=palette_colors, template=template)
        elif plot_type == "kde":
            # Use density contour as approximation
            fig = px.density_contour(plot_df, x=score_column, color=group_column,
                                     color_discrete_sequence=palette_colors, template=template)
        elif plot_type == "heatmap":
            # For heatmap, use group means
            heatmap_data = plot_df.groupby(group_column)[score_column].agg(['mean', 'std', 'count']).round(3)
            fig = px.imshow(heatmap_data.T, 
                           labels=dict(x=group_column, y="Statistics", color="Value"),
                           color_continuous_scale=color_palette, template=template)
        elif plot_type == "correlation":
            # Get numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, color_continuous_scale=color_palette, template=template)
            else:
                logger.warning("Not enough numeric columns for correlation plot.")
                return ""
        elif plot_type == "barplot":
            # Create bar plot with means
            mean_data = plot_df.groupby(group_column)[score_column].mean().reset_index()
            fig = px.bar(mean_data, x=group_column, y=score_column, color=group_column,
                        color_discrete_sequence=palette_colors, template=template)
        else:
            # Default to boxplot
            fig = px.box(plot_df, x=group_column, y=score_column, color=group_column,
                        color_discrete_sequence=palette_colors, template=template)

        # Update layout for theme-aware title
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': title_font_color, 'family': 'Arial'},
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.95,
                'yanchor': 'top'
            },
            font=dict(color=title_font_color),
            margin=dict(t=60, b=40, l=40, r=40),
            showlegend=True
        )

        output_file = f"{output_path}.html"
        fig.write_html(output_file)
        logger.info(f"✅ Interactive plot saved to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error generating interactive plot: {e}")
        return ""


def score_column_type(score_column: str) -> str:
    """
    Detect conservation score type based on column name.
    Returns: 'numeric', 'signed', 'ratio', or 'unknown'
    """
    col = score_column.lower()
    if "gerp" in col:
        return "numeric"
    elif "phylop" in col:
        return "signed"
    elif "phastcons" in col:
        return "ratio"
    else:
        return "unknown"


def suggest_plots(score_type: str) -> list:
    """
    Suggest best plot types for a given score type.
    """
    if score_type == "numeric":
        return ["kde", "violin"]
    elif score_type == "signed":
        return ["boxplot", "swarm"]
    elif score_type == "ratio":
        return ["boxplot", "density", "heatmap"]
    else:
        return ["boxplot", "violin", "swarm", "histogram"]
    if plot_df.empty:
        return ""
    n_groups = plot_df[group_column].nunique()
    palette_colors = get_palette_for_groups(COLOR_PALETTES.get(color_palette, "colorblind"), n_groups)

    # Select plot type
    if plot_type == "boxplot":
        fig = px.box(plot_df, x=group_column, y=score_column, color=group_column,
                    color_discrete_sequence=palette_colors, template=template)
    elif plot_type == "violin":
        fig = px.violin(plot_df, x=group_column, y=score_column, color=group_column,
                        box=True, points="all", color_discrete_sequence=palette_colors, template=template)
    elif plot_type == "swarm":
        fig = px.strip(plot_df, x=group_column, y=score_column, color=group_column,
                       color_discrete_sequence=palette_colors, template=template)
    elif plot_type == "histogram":
        fig = px.histogram(plot_df, x=score_column, color=group_column,
                          color_discrete_sequence=palette_colors, template=template)
    elif plot_type == "kde":
        fig = px.density_contour(plot_df, x=score_column, color=group_column,
                                 color_discrete_sequence=palette_colors, template=template)
    elif plot_type == "heatmap":
        # For heatmap, use group means
        heatmap_data = plot_df.groupby(group_column)[score_column].mean().reset_index()
        fig = px.imshow([heatmap_data[score_column].values],
                        labels=dict(x=heatmap_data[group_column].tolist(), y=[score_column]),
                        color_continuous_scale=color_palette, template=template)
    elif plot_type == "correlation":
        corr = plot_df.corr()
        fig = px.imshow(corr, color_continuous_scale=color_palette, template=template)
    else:
        fig = px.box(plot_df, x=group_column, y=score_column, color=group_column,
                    color_discrete_sequence=palette_colors, template=template)

    # Update layout for theme-aware title
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 20, 'color': title_font_color, 'family': 'Arial', 'bold': True},
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        font=dict(color=title_font_color),
        margin=dict(t=60, b=40, l=40, r=40)
    )

    output_file = f"{output_path}.html"
    fig.write_html(output_file)
    logger.info(f"✅ Interactive plot saved to: {output_file}")
    return output_file
