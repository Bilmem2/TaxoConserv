# advanced_grouping.py
"""
Advanced grouping and filtering system for TaxoConserv
Dynamic group selection, filtering, and hierarchical grouping
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import re
import logging

logger = logging.getLogger(__name__)


class AdvancedGroupingSystem:
    """
    Advanced grouping system with dynamic filtering and hierarchy support
    """
    
    def __init__(self):
        self.available_filters = {
            'size_filter': 'Group Size Filter',
            'name_filter': 'Group Name Filter', 
            'value_filter': 'Score Value Filter',
            'hierarchy_filter': 'Hierarchical Filter'
        }
        
        self.grouping_strategies = {
            'original': 'Use Original Groups',
            'merge_small': 'Merge Small Groups',
            'split_large': 'Split Large Groups',
            'hierarchical': 'Hierarchical Grouping',
            'custom_mapping': 'Custom Group Mapping',
            'statistical_clustering': 'Statistical Clustering'
        }
    
    def apply_size_filter(self, data: pd.DataFrame, group_column: str, 
                         min_size: int = 3, max_size: Optional[int] = None) -> pd.DataFrame:
        """Filter groups by size"""
        group_sizes = data[group_column].value_counts()
        
        valid_groups = group_sizes[group_sizes >= min_size]
        if max_size:
            valid_groups = valid_groups[valid_groups <= max_size]
        
        filtered_data = data[data[group_column].isin(valid_groups.index)]
        
        logger.info(f"Size filter: {len(valid_groups)} groups remaining (min={min_size}, max={max_size})")
        return filtered_data
    
    def apply_name_filter(self, data: pd.DataFrame, group_column: str,
                         include_patterns: Optional[List[str]] = None, 
                         exclude_patterns: Optional[List[str]] = None) -> pd.DataFrame:
        """Filter groups by name patterns"""
        if not include_patterns and not exclude_patterns:
            return data
        
        group_names = data[group_column].unique()
        valid_groups = set(group_names)
        
        # Apply include patterns
        if include_patterns:
            included = set()
            for pattern in include_patterns:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    included.update([g for g in group_names if regex.search(str(g))])
                except re.error:
                    # Fallback to substring matching
                    included.update([g for g in group_names if pattern.lower() in str(g).lower()])
            valid_groups = included
        
        # Apply exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                    valid_groups = {g for g in valid_groups if not regex.search(str(g))}
                except re.error:
                    # Fallback to substring matching
                    valid_groups = {g for g in valid_groups if pattern.lower() not in str(g).lower()}
        
        filtered_data = data[data[group_column].isin(valid_groups)]
        
        logger.info(f"Name filter: {len(valid_groups)} groups remaining")
        return filtered_data
    
    def apply_value_filter(self, data: pd.DataFrame, group_column: str, 
                          score_column: str, min_mean: Optional[float] = None,
                          max_mean: Optional[float] = None,
                          min_median: Optional[float] = None,
                          max_median: Optional[float] = None) -> pd.DataFrame:
        """Filter groups by statistical properties of their scores"""
        group_stats = data.groupby(group_column)[score_column].agg(['mean', 'median'])
        valid_groups = group_stats.index
        
        if min_mean is not None:
            valid_groups = group_stats[group_stats['mean'] >= min_mean].index
        if max_mean is not None:
            valid_groups = group_stats.loc[valid_groups][group_stats['mean'] <= max_mean].index
        if min_median is not None:
            valid_groups = group_stats.loc[valid_groups][group_stats['median'] >= min_median].index
        if max_median is not None:
            valid_groups = group_stats.loc[valid_groups][group_stats['median'] <= max_median].index
        
        filtered_data = data[data[group_column].isin(valid_groups)]
        
        logger.info(f"Value filter: {len(valid_groups)} groups remaining")
        return filtered_data
    
    def create_hierarchical_groups(self, data: pd.DataFrame, 
                                  hierarchy_columns: List[str]) -> pd.DataFrame:
        """Create hierarchical grouping from multiple columns"""
        data = data.copy()
        
        # Ensure all hierarchy columns exist
        available_cols = [col for col in hierarchy_columns if col in data.columns]
        if not available_cols:
            logger.warning("No hierarchy columns found in data")
            return data
        
        # Create hierarchical group names
        def create_hierarchy_name(row):
            parts = []
            for col in available_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    parts.append(str(row[col]).strip())
            return " > ".join(parts) if parts else "Unknown"
        
        data['hierarchical_group'] = data.apply(create_hierarchy_name, axis=1)
        
        logger.info(f"Hierarchical grouping: {data['hierarchical_group'].nunique()} groups created")
        return data
    
    def merge_small_groups(self, data: pd.DataFrame, group_column: str,
                          min_size: int = 3, merge_name: str = "Other") -> pd.DataFrame:
        """Merge small groups into a single category"""
        data = data.copy()
        group_sizes = data[group_column].value_counts()
        small_groups = group_sizes[group_sizes < min_size].index
        
        if len(small_groups) > 0:
            data.loc[data[group_column].isin(small_groups), group_column] = merge_name
            logger.info(f"Merged {len(small_groups)} small groups into '{merge_name}'")
        
        return data
    
    def apply_custom_mapping(self, data: pd.DataFrame, group_column: str,
                           mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply custom group name mappings"""
        if not mapping:
            return data
        
        data = data.copy()
        
        # Apply mapping with case-insensitive matching
        for old_name, new_name in mapping.items():
            # Direct mapping
            mask = data[group_column].str.lower() == old_name.lower()
            data.loc[mask, group_column] = new_name
            
            # Partial matching for flexibility
            mask = data[group_column].str.contains(old_name, case=False, na=False)
            data.loc[mask, group_column] = new_name
        
        logger.info(f"Applied {len(mapping)} custom mappings")
        return data
    
    def statistical_clustering(self, data: pd.DataFrame, group_column: str,
                             score_column: str, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """Group data using statistical clustering based on score distributions"""
        try:
            from sklearn.cluster import KMeans  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
        except ImportError:
            logger.warning("sklearn not available for statistical clustering")
            return data
        
        data = data.copy()
        
        # Calculate group statistics for clustering
        group_stats = data.groupby(group_column)[score_column].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).fillna(0)
        
        if len(group_stats) < 2:
            logger.warning("Need at least 2 groups for clustering")
            return data
        
        # Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = min(max(2, len(group_stats) // 3), 6)
        
        # Standardize features
        scaler = StandardScaler()
        features = scaler.fit_transform(group_stats)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Create cluster mapping
        cluster_mapping = dict(zip(group_stats.index, cluster_labels))
        
        # Apply clustering
        data['statistical_cluster'] = data[group_column].map(cluster_mapping)
        data['statistical_cluster'] = 'Cluster_' + data['statistical_cluster'].astype(str)
        
        logger.info(f"Statistical clustering: created {n_clusters} clusters")
        return data
    
    def get_group_summary(self, data: pd.DataFrame, group_column: str,
                         score_column: str) -> pd.DataFrame:
        """Get comprehensive summary of groups"""
        summary = data.groupby(group_column)[score_column].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(4)
        
        summary.columns = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        summary = summary.sort_values('Mean', ascending=False)
        
        # Add quality indicators
        summary['Quality'] = 'Good'
        summary.loc[summary['Count'] < 3, 'Quality'] = 'Small'
        summary.loc[summary['Count'] > 1000, 'Quality'] = 'Large'
        summary.loc[summary['Std Dev'] > summary['Mean'], 'Quality'] = 'High Variance'
        
        return summary


def create_advanced_grouping_ui(data: pd.DataFrame, group_column: str, score_column: str) -> Tuple[pd.DataFrame, str]:
    """Create advanced grouping UI and return processed data and final group column"""
    
    grouping_system = AdvancedGroupingSystem()
    processed_data = data.copy()
    final_group_column = group_column
    
    st.markdown("### 🎛️ Advanced Group Selection")
    
    # Grouping strategy selection
    strategy = st.selectbox(
        "Grouping Strategy",
        options=list(grouping_system.grouping_strategies.keys()),
        format_func=lambda x: grouping_system.grouping_strategies[x],
        help="Select how to process and organize groups"
    )
    
    if strategy == 'original':
        # No processing needed
        pass
    
    elif strategy == 'merge_small':
        col1, col2 = st.columns(2)
        with col1:
            min_size = st.number_input("Minimum Group Size", min_value=1, value=3)
        with col2:
            merge_name = st.text_input("Merged Group Name", value="Other")
        
        if st.button("Apply Small Group Merging"):
            processed_data = grouping_system.merge_small_groups(
                processed_data, final_group_column, min_size, merge_name
            )
            st.success(f"Merged groups with < {min_size} members into '{merge_name}'")
    
    elif strategy == 'hierarchical':
        available_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        
        hierarchy_cols = st.multiselect(
            "Hierarchy Columns (in order)",
            options=available_cols,
            default=available_cols[:3] if len(available_cols) >= 3 else available_cols,
            help="Select columns to create hierarchical groups (e.g., Kingdom > Phylum > Class)"
        )
        
        if hierarchy_cols and st.button("Create Hierarchical Groups"):
            processed_data = grouping_system.create_hierarchical_groups(processed_data, hierarchy_cols)
            final_group_column = 'hierarchical_group'
            st.success(f"Created hierarchical groups using: {' > '.join(hierarchy_cols)}")
    
    elif strategy == 'custom_mapping':
        st.markdown("**Custom Group Mappings**")
        mapping_text = st.text_area(
            "Group Mappings (old_name:new_name, one per line)",
            value="primate:Mammals\nhominid:Mammals\navian:Birds",
            help="Map existing group names to new categories"
        )
        
        # Parse mapping
        custom_mapping = {}
        for line in mapping_text.strip().split('\n'):
            if ':' in line:
                old, new = line.split(':', 1)
                custom_mapping[old.strip()] = new.strip()
        
        if custom_mapping and st.button("Apply Custom Mapping"):
            processed_data = grouping_system.apply_custom_mapping(
                processed_data, final_group_column, custom_mapping
            )
            st.success(f"Applied {len(custom_mapping)} custom mappings")
    
    elif strategy == 'statistical_clustering':
        n_clusters = st.number_input(
            "Number of Clusters", 
            min_value=2, 
            max_value=10, 
            value=min(4, processed_data[group_column].nunique()),
            help="Number of statistical clusters to create"
        )
        
        if st.button("Apply Statistical Clustering"):
            processed_data = grouping_system.statistical_clustering(
                processed_data, final_group_column, score_column, n_clusters
            )
            final_group_column = 'statistical_cluster'
            st.success(f"Created {n_clusters} statistical clusters")
    
    # Filtering options
    st.markdown("### 🔍 Group Filtering")
    
    with st.expander("Size Filter", expanded=False):
        enable_size_filter = st.checkbox("Enable Size Filter")
        if enable_size_filter:
            col1, col2 = st.columns(2)
            with col1:
                min_size_filter = st.number_input("Minimum Size", min_value=1, value=3)
            with col2:
                max_size_filter = st.number_input("Maximum Size", min_value=1, value=1000)
                if max_size_filter <= min_size_filter:
                    max_size_filter = None
            
            processed_data = grouping_system.apply_size_filter(
                processed_data, final_group_column, min_size_filter, max_size_filter
            )
    
    with st.expander("Name Filter", expanded=False):
        enable_name_filter = st.checkbox("Enable Name Filter")
        if enable_name_filter:
            col1, col2 = st.columns(2)
            with col1:
                include_patterns = st.text_area(
                    "Include Patterns (one per line)",
                    help="Groups matching these patterns will be included"
                ).strip().split('\n') if st.text_area(
                    "Include Patterns (one per line)",
                    help="Groups matching these patterns will be included"
                ).strip() else []
            
            with col2:
                exclude_patterns = st.text_area(
                    "Exclude Patterns (one per line)",
                    help="Groups matching these patterns will be excluded"
                ).strip().split('\n') if st.text_area(
                    "Exclude Patterns (one per line)",
                    help="Groups matching these patterns will be excluded"
                ).strip() else []
            
            if include_patterns or exclude_patterns:
                processed_data = grouping_system.apply_name_filter(
                    processed_data, final_group_column, include_patterns, exclude_patterns
                )
    
    with st.expander("Value Filter", expanded=False):
        enable_value_filter = st.checkbox("Enable Value Filter")
        if enable_value_filter:
            # Calculate current stats for reference
            current_stats = processed_data.groupby(final_group_column)[score_column].agg(['mean', 'median'])
            min_mean_available = current_stats['mean'].min()
            max_mean_available = current_stats['mean'].max()
            
            st.write(f"Current mean range: {min_mean_available:.3f} to {max_mean_available:.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                min_mean = st.number_input(
                    "Minimum Mean Score", 
                    value=min_mean_available,
                    help="Groups with mean score below this will be excluded"
                )
                min_median = st.number_input(
                    "Minimum Median Score", 
                    value=current_stats['median'].min(),
                    help="Groups with median score below this will be excluded"
                )
            
            with col2:
                max_mean = st.number_input(
                    "Maximum Mean Score", 
                    value=max_mean_available,
                    help="Groups with mean score above this will be excluded"
                )
                max_median = st.number_input(
                    "Maximum Median Score", 
                    value=current_stats['median'].max(),
                    help="Groups with median score above this will be excluded"
                )
            
            processed_data = grouping_system.apply_value_filter(
                processed_data, final_group_column, score_column,
                min_mean, max_mean, min_median, max_median
            )
    
    # Group summary
    st.markdown("### 📊 Group Summary")
    
    if len(processed_data) > 0:
        summary = grouping_system.get_group_summary(processed_data, final_group_column, score_column)
        
        # Display summary with color coding
        def color_quality(val):
            if val == 'Small':
                return 'background-color: #ffebcc'
            elif val == 'Large':
                return 'background-color: #e6f3ff'
            elif val == 'High Variance':
                return 'background-color: #ffe6e6'
            else:
                return 'background-color: #e6ffe6'
        
        styled_summary = summary.style.map(color_quality, subset=['Quality'])  # type: ignore
        st.dataframe(styled_summary, use_container_width=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Groups", len(summary))
        with col2:
            st.metric("Total Samples", len(processed_data))
        with col3:
            st.metric("Avg Group Size", f"{summary['Count'].mean():.1f}")
        with col4:
            small_groups = len(summary[summary['Quality'] == 'Small'])
            st.metric("Small Groups", small_groups)
        
        # Group size distribution
        st.markdown("**Group Size Distribution**")
        size_dist = summary['Count'].value_counts().sort_index()
        st.bar_chart(size_dist)
        
    else:
        st.warning("No groups remaining after filtering!")
        processed_data = data.copy()  # Reset to original data
        final_group_column = group_column
    
    return processed_data, final_group_column


# Export key functions
__all__ = [
    'AdvancedGroupingSystem',
    'create_advanced_grouping_ui'
]
