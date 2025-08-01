"""
TaxoConserv - Utilities Module

Genel yardımcı fonksiyonlar ve tekrar eden kodlar burada tutulur.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import logging

COLOR_PALETTES = {
    "Set1": "Set1",
    "Set2": "Set2",
    "Set3": "Set3",
    "tab10": "tab10",
    "Dark2": "Dark2",
    "Pastel1": "Pastel1",
    "Accent": "Accent",
    "viridis": "viridis",
    "plasma": "plasma",
    "inferno": "inferno",
    "magma": "magma",
    "cividis": "cividis",
    "colorblind": "colorblind"
}

def get_palette_for_groups(palette_name, n_groups):
    try:
        return sns.color_palette(palette_name, n_groups)
    except Exception:
        return sns.color_palette("Set3", n_groups)

def _setup_plot_style(color_palette: str):
    sns.set_style("whitegrid")
    if color_palette == "colorblind":
        sns.set_palette("colorblind")
    else:
        sns.set_palette(color_palette)

def _customize_plot(ax, title, group_column, score_column):
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(group_column.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(score_column.replace('_', ' ').title(), fontsize=12)
    if len(ax.get_xticklabels()) > 5:
        plt.xticks(rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

def _save_plot(fig, output_file, dpi, plot_df, group_column, score_column):
    logger = logging.getLogger(__name__)
    logger.info("📈 Plot Statistics:")
    overall_stats = plot_df[score_column].describe()
    logger.info(f"   Overall: n={len(plot_df)}, mean={overall_stats['mean']:.3f}, std={overall_stats['std']:.3f}")
    for group in sorted(plot_df[group_column].unique()):
        group_data = plot_df[plot_df[group_column] == group][score_column]
        logger.info(f"   {group}: n={len(group_data)}, mean={group_data.mean():.3f}, median={group_data.median():.3f}, std={group_data.std():.3f}")
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"✅ Boxplot saved to: {output_file}")
    return output_file
