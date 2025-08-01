#!/usr/bin/env python3
"""
TaxoConserv - Analysis Module

İstatistiksel analiz fonksiyonları burada yer alır.
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal


def perform_statistical_analysis(data, score_column, taxon_column):
    """Perform statistical analysis on the data."""
    try:
        # Group data by taxon
        groups = []
        group_names = []
        for taxon in data[taxon_column].unique():
            group_data = data[data[taxon_column] == taxon][score_column].values
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(taxon)
        # Calculate descriptive statistics + mod, IQR, variance
        def mod_func(x):
            return x.mode().iloc[0] if not x.mode().empty else np.nan
        def iqr_func(x):
            return x.quantile(0.75) - x.quantile(0.25)
        stats_summary = data.groupby(taxon_column)[score_column].agg([
            'count', 'mean', 'median', 'std', 'min', 'max', mod_func, iqr_func, 'var'
        ]).rename(columns={'mod_func': 'mode', 'iqr_func': 'IQR', 'var': 'variance'}).round(4)
        # Run Kruskal-Wallis test if we have multiple groups
        if len(groups) > 1:
            h_stat, p_value = kruskal(*groups)
            significant = p_value < 0.05
        else:
            h_stat, p_value = 0, 1
            significant = False
        return {
            'stats_summary': stats_summary,
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant': significant,
            'group_names': group_names,
            'n_groups': len(groups)
        }
    except Exception as e:
        # UI tarafında Streamlit ile hata gösterilecek, burada raise etmek yeterli
        raise RuntimeError(f"Error in statistical analysis: {e}")
