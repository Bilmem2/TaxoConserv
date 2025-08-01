# multi_score_analysis.py
"""
Multi-Score Analysis System for TaxoConserv
Comparative analysis across multiple conservation scores
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage
import logging

logger = logging.getLogger(__name__)


class MultiScoreAnalyzer:
    """
    Advanced multi-score analysis system for comparative conservation analysis
    """
    
    def __init__(self):
        self.correlation_methods = {
            'pearson': 'Pearson (linear relationships)',
            'spearman': 'Spearman (monotonic relationships)',
            'kendall': 'Kendall (rank-based)'
        }
        
        self.clustering_methods = {
            'ward': 'Ward (minimizes variance)',
            'complete': 'Complete (maximum distance)',
            'average': 'Average (UPGMA)',
            'single': 'Single (minimum distance)'
        }
        
        self.comparison_metrics = [
            'correlation', 'rank_correlation', 'agreement_analysis',
            'distribution_comparison', 'group_consistency'
        ]
    
    def detect_conservation_scores(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect multiple conservation scores in dataset"""
        score_patterns = {
            'phylop': ['phylop', 'phlyop', 'phyloP'],
            'phastcons': ['phastcons', 'phastCons', 'phast'],
            'gerp': ['gerp', 'GERP'],
            'cadd': ['cadd', 'CADD'],
            'revel': ['revel', 'REVEL'],
            'sift': ['sift', 'SIFT'],
            'polyphen': ['polyphen', 'polyphen2', 'pp2'],
            'conservation_score': ['conservation', 'conserv'],
            'evolutionary_rate': ['evolution', 'evol', 'rate'],
            'constraint': ['constraint', 'constrained']
        }
        
        detected_scores = {}
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        for col in numeric_cols:
            col_lower = col.lower()
            for score_type, patterns in score_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    detected_scores[col] = score_type
                    break
        
        # If no specific patterns found, consider all numeric columns as potential scores
        if not detected_scores:
            detected_scores = {col: 'conservation_score' for col in numeric_cols}
        
        logger.info(f"Detected {len(detected_scores)} conservation scores")
        return detected_scores
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, 
                                   score_columns: List[str], 
                                   method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix between multiple scores"""
        if len(score_columns) < 2:
            raise ValueError("Need at least 2 score columns for correlation analysis")
        
        score_data = data[score_columns].dropna()
        
        if method == 'pearson':
            corr_matrix = score_data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = score_data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = score_data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return corr_matrix
    
    def perform_pairwise_analysis(self, data: pd.DataFrame, 
                                score_columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive pairwise analysis between scores"""
        results = {}
        
        for i, score1 in enumerate(score_columns):
            for score2 in score_columns[i+1:]:
                pair_key = f"{score1}_vs_{score2}"
                
                # Get paired data (remove rows with NaN in either column)
                paired_data = data[[score1, score2]].dropna()
                
                if len(paired_data) < 10:
                    logger.warning(f"Insufficient data for {pair_key} analysis")
                    continue
                
                # Calculate correlations
                pearson_corr, pearson_p = pearsonr(paired_data[score1], paired_data[score2])
                spearman_corr, spearman_p = spearmanr(paired_data[score1], paired_data[score2])
                
                # Agreement analysis (for similar-range scores)
                range1 = paired_data[score1].max() - paired_data[score1].min()
                range2 = paired_data[score2].max() - paired_data[score2].min()
                range_similarity = min(range1, range2) / max(range1, range2)
                
                # Rank agreement
                rank1 = paired_data[score1].rank()
                rank2 = paired_data[score2].rank()
                rank_agreement = (rank1 - rank2).abs().mean() / len(paired_data)
                
                results[pair_key] = {
                    'n_samples': len(paired_data),
                    'pearson_correlation': pearson_corr,
                    'pearson_p_value': pearson_p,
                    'spearman_correlation': spearman_corr,
                    'spearman_p_value': spearman_p,
                    'range_similarity': range_similarity,
                    'rank_agreement': 1 - rank_agreement,  # Convert to agreement score
                    'data': paired_data
                }
        
        return results
    
    def analyze_group_consistency(self, data: pd.DataFrame, 
                                score_columns: List[str], 
                                group_column: str) -> Dict[str, Any]:
        """Analyze consistency of scores across taxonomic groups"""
        results = {}
        
        for group in data[group_column].unique():
            if pd.isna(group):
                continue
                
            group_data = data[data[group_column] == group][score_columns].dropna()
            
            if len(group_data) < 3:
                continue
            
            # Calculate within-group correlations
            group_corr = group_data.corr(method='spearman')
            
            # Calculate consistency metrics
            mean_correlation = group_corr.values[np.triu_indices_from(group_corr.values, k=1)].mean()
            
            # Coefficient of variation for each score within the group
            cv_scores = {}
            for score in score_columns:
                if score in group_data.columns and len(group_data[score]) > 1:
                    mean_val = group_data[score].mean()
                    std_val = group_data[score].std()
                    cv_scores[score] = std_val / mean_val if mean_val != 0 else np.inf
            
            results[str(group)] = {
                'n_samples': len(group_data),
                'mean_correlation': mean_correlation,
                'correlation_matrix': group_corr,
                'coefficient_variations': cv_scores,
                'score_means': group_data.mean().to_dict(),
                'score_stds': group_data.std().to_dict()
            }
        
        return results
    
    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame, 
                                 title: str = "Score Correlation Matrix",
                                 interactive: bool = True) -> Any:
        """Create correlation heatmap visualization"""
        if interactive:
            # Plotly heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Conservation Scores",
                yaxis_title="Conservation Scores",
                width=600,
                height=500
            )
            
            return fig
        else:
            # Matplotlib heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            return fig
    
    def create_pairwise_scatterplot(self, pairwise_results: Dict[str, Any],
                                  interactive: bool = True) -> Any:
        """Create pairwise scatterplot matrix"""
        pairs = list(pairwise_results.keys())
        n_pairs = len(pairs)
        
        if n_pairs == 0:
            return None
        
        if interactive:
            # Create subplot structure
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=[pair.replace('_vs_', ' vs ') for pair in pairs],
                horizontal_spacing=0.1,
                vertical_spacing=0.1
            )
            
            for i, (pair_key, pair_data) in enumerate(pairwise_results.items()):
                row = (i // n_cols) + 1
                col = (i % n_cols) + 1
                
                score1, score2 = pair_key.split('_vs_')
                data = pair_data['data']
                
                # Add scatter plot
                fig.add_trace(
                    go.Scatter(
                        x=data[score1],
                        y=data[score2],
                        mode='markers',
                        marker=dict(size=4, opacity=0.6),
                        name=f"{score1} vs {score2}",
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add trend line
                z = np.polyfit(data[score1], data[score2], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(data[score1].min(), data[score1].max(), 100)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=p(x_trend),
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f"Trend",
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Pairwise Score Comparisons",
                height=n_rows * 300,
                width=1000
            )
            
            return fig
        else:
            # Matplotlib version
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
            if n_pairs == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (pair_key, pair_data) in enumerate(pairwise_results.items()):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                score1, score2 = pair_key.split('_vs_')
                data = pair_data['data']
                
                # Scatter plot
                ax.scatter(data[score1], data[score2], alpha=0.6, s=20)
                
                # Trend line
                z = np.polyfit(data[score1], data[score2], 1)
                p = np.poly1d(z)
                ax.plot(data[score1], p(data[score1]), "r--", alpha=0.8)
                
                # Labels and title
                ax.set_xlabel(score1.replace('_', ' ').title())
                ax.set_ylabel(score2.replace('_', ' ').title())
                ax.set_title(f"{score1} vs {score2}\nr={pair_data['pearson_correlation']:.3f}")
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(pairwise_results), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig
    
    def create_score_distribution_comparison(self, data: pd.DataFrame,
                                           score_columns: List[str],
                                           group_column: str,
                                           interactive: bool = True) -> Any:
        """Create comparative distribution plots for multiple scores"""
        if interactive:
            # Create subplots for each score
            n_scores = len(score_columns)
            fig = make_subplots(
                rows=n_scores, cols=1,
                subplot_titles=[f"{score.replace('_', ' ').title()} Distribution" for score in score_columns],
                vertical_spacing=0.1
            )
            
            colors = px.colors.qualitative.Set3
            
            for i, score in enumerate(score_columns):
                for j, group in enumerate(data[group_column].unique()):
                    if pd.isna(group):
                        continue
                        
                    group_data = data[data[group_column] == group][score].dropna()
                    
                    if len(group_data) > 0:
                        fig.add_trace(
                            go.Box(
                                y=group_data,
                                name=str(group),
                                boxpoints='outliers',
                                marker_color=colors[j % len(colors)],
                                showlegend=(i == 0)  # Only show legend for first subplot
                            ),
                            row=i+1, col=1
                        )
            
            fig.update_layout(
                title="Score Distributions Across Groups",
                height=300 * n_scores,
                width=800
            )
            
            return fig
        else:
            # Matplotlib version
            n_scores = len(score_columns)
            fig, axes = plt.subplots(n_scores, 1, figsize=(10, 4*n_scores))
            
            if n_scores == 1:
                axes = [axes]
            
            for i, score in enumerate(score_columns):
                sns.boxplot(data=data, x=group_column, y=score, ax=axes[i])
                axes[i].set_title(f"{score.replace('_', ' ').title()} Distribution")
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
    
    def generate_score_summary_report(self, data: pd.DataFrame,
                                    score_columns: List[str],
                                    group_column: str) -> Dict[str, Any]:
        """Generate comprehensive summary report for multi-score analysis"""
        
        # Basic statistics
        score_stats = data[score_columns].describe()
        
        # Correlation analysis
        corr_matrix = self.calculate_correlation_matrix(data, score_columns, 'pearson')
        spearman_corr = self.calculate_correlation_matrix(data, score_columns, 'spearman')
        
        # Pairwise analysis
        pairwise_results = self.perform_pairwise_analysis(data, score_columns)
        
        # Group consistency
        group_consistency = self.analyze_group_consistency(data, score_columns, group_column)
        
        # Score rankings per group
        group_rankings = {}
        for group in data[group_column].unique():
            if pd.isna(group):
                continue
            group_data = data[data[group_column] == group]
            group_means = group_data[score_columns].mean()
            group_rankings[str(group)] = group_means.rank(ascending=False).to_dict()
        
        return {
            'basic_statistics': score_stats,
            'pearson_correlations': corr_matrix,
            'spearman_correlations': spearman_corr,
            'pairwise_analysis': pairwise_results,
            'group_consistency': group_consistency,
            'group_rankings': group_rankings,
            'n_scores': len(score_columns),
            'n_groups': data[group_column].nunique(),
            'total_samples': len(data)
        }


def create_multi_score_analysis_ui(data: pd.DataFrame, group_column: str) -> Dict[str, Any]:
    """Create multi-score analysis UI and return results"""
    
    analyzer = MultiScoreAnalyzer()
    
    st.markdown("### 🔬 Multi-Score Conservation Analysis")
    
    # Detect conservation scores
    detected_scores = analyzer.detect_conservation_scores(data)
    all_numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
    
    # Score selection
    st.markdown("**Select Conservation Scores for Analysis:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show detected scores with descriptions
        if detected_scores:
            st.markdown("**🎯 Detected Conservation Scores:**")
            for score, score_type in detected_scores.items():
                st.markdown(f"• **{score}** ({score_type})")
        
        selected_scores = st.multiselect(
            "Choose scores to analyze",
            options=all_numeric_cols,
            default=list(detected_scores.keys())[:4] if detected_scores else all_numeric_cols[:2],
            help="Select 2 or more conservation scores for comparative analysis"
        )
    
    with col2:
        if len(selected_scores) >= 2:
            st.success(f"✅ {len(selected_scores)} scores selected")
            
            # Quick preview of selected scores
            preview_stats = data[selected_scores].describe().round(3)
            st.dataframe(preview_stats.T[['mean', 'std', 'min', 'max']], 
                        use_container_width=True)
        else:
            st.warning("⚠️ Select at least 2 scores")
            return {}
    
    if len(selected_scores) < 2:
        st.info("Please select at least 2 conservation scores to begin analysis.")
        return {}
    
    # Analysis options
    st.markdown("---")
    st.markdown("**Analysis Options:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        correlation_method = st.selectbox(
            "Correlation Method",
            options=list(analyzer.correlation_methods.keys()),
            format_func=lambda x: analyzer.correlation_methods[x]
        )
    
    with col2:
        show_pairwise = st.checkbox("Pairwise Analysis", value=True)
        show_distributions = st.checkbox("Distribution Comparison", value=True)
    
    with col3:
        interactive_plots = st.checkbox("Interactive Plots", value=True)
        show_group_analysis = st.checkbox("Group Consistency", value=True)
    
    # Run analysis button
    if st.button("🚀 Run Multi-Score Analysis", type="primary"):
        
        with st.spinner("Performing multi-score analysis..."):
            # Generate comprehensive report
            analysis_results = analyzer.generate_score_summary_report(
                data, selected_scores, group_column
            )
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Scores Analyzed", analysis_results['n_scores'])
        with col2:
            st.metric("Groups", analysis_results['n_groups'])
        with col3:
            st.metric("Total Samples", analysis_results['total_samples'])
        with col4:
            n_comparisons = len(analysis_results['pairwise_analysis'])
            st.metric("Pairwise Comparisons", n_comparisons)
        
        # Correlation Analysis
        st.markdown("### 🔗 Correlation Analysis")
        
        corr_matrix = analysis_results['pearson_correlations']
        fig_corr = analyzer.create_correlation_heatmap(
            corr_matrix, f"{correlation_method.title()} Correlation Matrix", interactive_plots
        )
        
        if interactive_plots:
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.pyplot(fig_corr)
        
        # Correlation summary
        with st.expander("📋 Correlation Summary", expanded=False):
            
            # Get correlation values (excluding diagonal)
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Correlation", f"{np.mean(corr_values):.3f}")
                st.metric("Max Correlation", f"{np.max(corr_values):.3f}")
            with col2:
                st.metric("Min Correlation", f"{np.min(corr_values):.3f}")
                st.metric("Std Correlation", f"{np.std(corr_values):.3f}")
            
            st.dataframe(corr_matrix, use_container_width=True)
        
        # Pairwise Analysis
        if show_pairwise and analysis_results['pairwise_analysis']:
            st.markdown("### 🔍 Pairwise Score Analysis")
            
            fig_pairwise = analyzer.create_pairwise_scatterplot(
                analysis_results['pairwise_analysis'], interactive_plots
            )
            
            if fig_pairwise:
                if interactive_plots:
                    st.plotly_chart(fig_pairwise, use_container_width=True)
                else:
                    st.pyplot(fig_pairwise)
            
            # Pairwise summary table
            with st.expander("📊 Detailed Pairwise Results", expanded=False):
                pairwise_summary = []
                for pair, results in analysis_results['pairwise_analysis'].items():
                    pairwise_summary.append({
                        'Score Pair': pair.replace('_vs_', ' vs '),
                        'Samples': results['n_samples'],
                        'Pearson r': f"{results['pearson_correlation']:.3f}",
                        'Pearson p': f"{results['pearson_p_value']:.3e}",
                        'Spearman r': f"{results['spearman_correlation']:.3f}",
                        'Rank Agreement': f"{results['rank_agreement']:.3f}"
                    })
                
                pairwise_df = pd.DataFrame(pairwise_summary)
                st.dataframe(pairwise_df, use_container_width=True)
        
        # Distribution Comparison
        if show_distributions:
            st.markdown("### 📈 Score Distribution Comparison")
            
            fig_dist = analyzer.create_score_distribution_comparison(
                data, selected_scores, group_column, interactive_plots
            )
            
            if interactive_plots:
                st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.pyplot(fig_dist)
        
        # Group Consistency Analysis
        if show_group_analysis and analysis_results['group_consistency']:
            st.markdown("### 🎯 Group Consistency Analysis")
            
            # Group consistency summary
            consistency_summary = []
            for group, results in analysis_results['group_consistency'].items():
                consistency_summary.append({
                    'Group': group,
                    'Samples': results['n_samples'],
                    'Mean Correlation': f"{results['mean_correlation']:.3f}",
                    'Score Variability': f"{np.mean(list(results['coefficient_variations'].values())):.3f}"
                })
            
            consistency_df = pd.DataFrame(consistency_summary)
            st.dataframe(consistency_df, use_container_width=True)
            
            # Group rankings
            with st.expander("🏆 Score Rankings by Group", expanded=False):
                rankings_data = analysis_results['group_rankings']
                rankings_df = pd.DataFrame(rankings_data).T
                
                # Style the rankings
                styled_rankings = rankings_df.style.background_gradient(
                    cmap='RdYlGn_r', axis=1
                ).format(precision=1)
                
                st.dataframe(styled_rankings, use_container_width=True)
        
        # Summary insights
        st.markdown("### 💡 Key Insights")
        
        insights = []
        
        # Correlation insights
        mean_corr = np.mean(corr_values)
        if mean_corr > 0.7:
            insights.append("🔗 **Strong correlations** detected between scores - they measure similar conservation aspects")
        elif mean_corr > 0.3:
            insights.append("🔗 **Moderate correlations** - scores capture related but distinct conservation features")
        else:
            insights.append("🔗 **Weak correlations** - scores measure different aspects of conservation")
        
        # Consistency insights
        if analysis_results['group_consistency']:
            group_correlations = [r['mean_correlation'] for r in analysis_results['group_consistency'].values()]
            mean_group_corr = np.mean(group_correlations)
            
            if mean_group_corr > 0.6:
                insights.append("🎯 **High group consistency** - scores behave similarly within taxonomic groups")
            else:
                insights.append("🎯 **Variable group consistency** - score relationships differ across groups")
        
        # Sample size insights
        if analysis_results['total_samples'] > 1000:
            insights.append("📊 **Large dataset** - results are statistically robust")
        elif analysis_results['total_samples'] < 100:
            insights.append("📊 **Small dataset** - interpret results with caution")
        
        for insight in insights:
            st.markdown(insight)
        
        return analysis_results
    
    return {}


# Export key functions
__all__ = [
    'MultiScoreAnalyzer',
    'create_multi_score_analysis_ui'
]
