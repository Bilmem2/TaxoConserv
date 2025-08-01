# advanced_statistics.py
"""
Advanced Statistical Analysis System for TaxoConserv
Extended statistical tests and analysis methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from scipy.stats import (
    kruskal, mannwhitneyu, friedmanchisquare, wilcoxon,
    levene, bartlett, shapiro, normaltest, kstest,
    spearmanr, pearsonr, kendalltau, chi2_contingency,
    ttest_ind, ttest_rel, f_oneway, tukey_hsd
)
import warnings
import logging

logger = logging.getLogger(__name__)


class AdvancedStatisticalAnalyzer:
    """
    Comprehensive statistical analysis system for conservation data
    """
    
    def __init__(self):
        self.parametric_tests = {
            'one_way_anova': 'One-way ANOVA (parametric)',
            'welch_anova': "Welch's ANOVA (unequal variances)",
            'tukey_hsd': 'Tukey HSD post-hoc test'
        }
        
        self.non_parametric_tests = {
            'kruskal_wallis': 'Kruskal-Wallis H-test',
            'mann_whitney': 'Mann-Whitney U test (pairwise)',
            'friedman': 'Friedman test (repeated measures)',
            'wilcoxon': 'Wilcoxon signed-rank test'
        }
        
        self.normality_tests = {
            'shapiro_wilk': 'Shapiro-Wilk test',
            'dagostino_pearson': "D'Agostino-Pearson test",
            'kolmogorov_smirnov': 'Kolmogorov-Smirnov test'
        }
        
        self.variance_tests = {
            'levene': "Levene's test",
            'bartlett': "Bartlett's test"
        }
        
        self.correlation_tests = {
            'pearson': 'Pearson correlation',
            'spearman': 'Spearman correlation',
            'kendall': 'Kendall tau correlation'
        }
        
        self.effect_size_methods = {
            'eta_squared': 'Eta squared (η²)',
            'omega_squared': 'Omega squared (ω²)',
            'epsilon_squared': 'Epsilon squared (ε²)',
            'cohens_d': "Cohen's d"
        }
    
    def check_normality(self, data: pd.Series, test: str = 'shapiro_wilk') -> Dict[str, Any]:
        """Perform normality tests on data"""
        clean_data = data.dropna()
        
        if len(clean_data) < 3:
            return {'error': 'Insufficient data for normality test'}
        
        try:
            if test == 'shapiro_wilk':
                if len(clean_data) > 5000:
                    # Shapiro-Wilk not reliable for large samples
                    test = 'dagostino_pearson'
                else:
                    statistic, p_value = shapiro(clean_data)
                    test_name = 'Shapiro-Wilk'
            
            if test == 'dagostino_pearson':
                statistic, p_value = normaltest(clean_data)
                test_name = "D'Agostino-Pearson"
            
            elif test == 'kolmogorov_smirnov':
                # Compare against normal distribution with same mean and std
                statistic, p_value = kstest(clean_data, 'norm', 
                                           args=(clean_data.mean(), clean_data.std()))
                test_name = 'Kolmogorov-Smirnov'
            
            # Interpretation
            alpha = 0.05
            is_normal = p_value > alpha
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': is_normal,
                'alpha': alpha,
                'n_samples': len(clean_data),
                'interpretation': f"Data {'appears to be' if is_normal else 'does not appear to be'} normally distributed (p={'>' if is_normal else '<'} {alpha})"
            }
            
        except Exception as e:
            logger.error(f"Normality test failed: {e}")
            return {'error': f'Normality test failed: {str(e)}'}
    
    def check_variance_homogeneity(self, data: pd.DataFrame, score_column: str, 
                                 group_column: str, test: str = 'levene') -> Dict[str, Any]:
        """Check homogeneity of variances across groups"""
        try:
            groups = []
            group_names = []
            
            for group_name in data[group_column].unique():
                if pd.notna(group_name):
                    group_data = data[data[group_column] == group_name][score_column].dropna()
                    if len(group_data) >= 2:  # Need at least 2 observations per group
                        groups.append(group_data)
                        group_names.append(str(group_name))
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups with sufficient data'}
            
            if test == 'levene':
                statistic, p_value = levene(*groups)
                test_name = "Levene's test"
            elif test == 'bartlett':
                statistic, p_value = bartlett(*groups)
                test_name = "Bartlett's test"
            else:
                return {'error': f'Unknown variance test: {test}'}
            
            alpha = 0.05
            homogeneous = p_value > alpha
            
            return {
                'test_name': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'homogeneous': homogeneous,
                'alpha': alpha,
                'n_groups': len(groups),
                'group_names': group_names,
                'interpretation': f"Variances {'appear to be' if homogeneous else 'do not appear to be'} homogeneous across groups (p={'>' if homogeneous else '<'} {alpha})"
            }
            
        except Exception as e:
            logger.error(f"Variance homogeneity test failed: {e}")
            return {'error': f'Variance test failed: {str(e)}'}
    
    def perform_parametric_tests(self, data: pd.DataFrame, score_column: str, 
                               group_column: str) -> Dict[str, Any]:
        """Perform parametric statistical tests"""
        results = {}
        
        try:
            # Prepare group data
            groups = []
            group_names = []
            
            for group_name in data[group_column].unique():
                if pd.notna(group_name):
                    group_data = data[data[group_column] == group_name][score_column].dropna()
                    if len(group_data) >= 2:
                        groups.append(group_data)
                        group_names.append(str(group_name))
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for parametric tests'}
            
            # One-way ANOVA
            try:
                f_stat, p_value = f_oneway(*groups)
                results['one_way_anova'] = {
                    'test_name': 'One-way ANOVA',
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'degrees_freedom': (len(groups) - 1, sum(len(g) for g in groups) - len(groups)),
                    'interpretation': f"Groups {'are' if p_value < 0.05 else 'are not'} significantly different (F={f_stat:.4f}, p={p_value:.6f})"
                }
            except Exception as e:
                results['one_way_anova'] = {'error': f'ANOVA failed: {str(e)}'}
            
            # Tukey HSD post-hoc test (if ANOVA is significant)
            if results.get('one_way_anova', {}).get('significant', False):
                try:
                    # Prepare data for Tukey HSD
                    all_data = []
                    all_groups = []
                    for i, group_data in enumerate(groups):
                        all_data.extend(group_data)
                        all_groups.extend([group_names[i]] * len(group_data))
                    
                    tukey_result = tukey_hsd(*groups)
                    
                    results['tukey_hsd'] = {
                        'test_name': 'Tukey HSD',
                        'result': tukey_result,
                        'confidence_interval': tukey_result.confidence_interval(),
                        'pvalue': tukey_result.pvalue,
                        'interpretation': 'Post-hoc pairwise comparisons completed'
                    }
                except Exception as e:
                    logger.warning(f"Tukey HSD failed: {e}")
                    results['tukey_hsd'] = {'error': f'Post-hoc test failed: {str(e)}'}
            
            return results
            
        except Exception as e:
            logger.error(f"Parametric tests failed: {e}")
            return {'error': f'Parametric tests failed: {str(e)}'}
    
    def perform_non_parametric_tests(self, data: pd.DataFrame, score_column: str, 
                                   group_column: str) -> Dict[str, Any]:
        """Perform non-parametric statistical tests"""
        results = {}
        
        try:
            # Prepare group data
            groups = []
            group_names = []
            
            for group_name in data[group_column].unique():
                if pd.notna(group_name):
                    group_data = data[data[group_column] == group_name][score_column].dropna()
                    if len(group_data) >= 1:
                        groups.append(group_data)
                        group_names.append(str(group_name))
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for non-parametric tests'}
            
            # Kruskal-Wallis H-test
            try:
                h_stat, p_value = kruskal(*groups)
                results['kruskal_wallis'] = {
                    'test_name': 'Kruskal-Wallis H-test',
                    'h_statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'degrees_freedom': len(groups) - 1,
                    'interpretation': f"Groups {'have' if p_value < 0.05 else 'do not have'} significantly different medians (H={h_stat:.4f}, p={p_value:.6f})"
                }
            except Exception as e:
                results['kruskal_wallis'] = {'error': f'Kruskal-Wallis test failed: {str(e)}'}
            
            # Mann-Whitney U tests (pairwise comparisons)
            if len(groups) >= 2:
                pairwise_results = []
                
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        try:
                            u_stat, p_value = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                            pairwise_results.append({
                                'group1': group_names[i],
                                'group2': group_names[j],
                                'u_statistic': u_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except Exception as e:
                            logger.warning(f"Mann-Whitney test failed for {group_names[i]} vs {group_names[j]}: {e}")
                
                results['mann_whitney'] = {
                    'test_name': 'Mann-Whitney U test (pairwise)',
                    'pairwise_results': pairwise_results,
                    'interpretation': f'Performed {len(pairwise_results)} pairwise comparisons'
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Non-parametric tests failed: {e}")
            return {'error': f'Non-parametric tests failed: {str(e)}'}
    
    def calculate_effect_sizes(self, data: pd.DataFrame, score_column: str, 
                             group_column: str) -> Dict[str, Any]:
        """Calculate effect sizes for group differences"""
        try:
            # Prepare group data
            groups = []
            group_names = []
            
            for group_name in data[group_column].unique():
                if pd.notna(group_name):
                    group_data = data[data[group_column] == group_name][score_column].dropna()
                    if len(group_data) >= 2:
                        groups.append(group_data)
                        group_names.append(str(group_name))
            
            if len(groups) < 2:
                return {'error': 'Need at least 2 groups for effect size calculation'}
            
            # Calculate various effect sizes
            effect_sizes = {}
            
            # Eta squared (η²) - for ANOVA
            try:
                # Calculate sum of squares
                all_data = np.concatenate(groups)
                grand_mean = np.mean(all_data)
                
                # Between-group sum of squares
                ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
                
                # Total sum of squares
                ss_total = sum((x - grand_mean)**2 for x in all_data)
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                effect_sizes['eta_squared'] = {
                    'value': eta_squared,
                    'interpretation': self._interpret_eta_squared(eta_squared),
                    'description': 'Proportion of variance explained by group differences'
                }
            except Exception as e:
                logger.warning(f"Eta squared calculation failed: {e}")
            
            # Cohen's d for pairwise comparisons
            if len(groups) == 2:
                try:
                    group1, group2 = groups[0], groups[1]
                    pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                                        (len(group2) - 1) * np.var(group2, ddof=1)) / 
                                       (len(group1) + len(group2) - 2))
                    
                    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
                    
                    effect_sizes['cohens_d'] = {
                        'value': cohens_d,
                        'interpretation': self._interpret_cohens_d(abs(cohens_d)),
                        'description': f'Standardized mean difference between {group_names[0]} and {group_names[1]}'
                    }
                except Exception as e:
                    logger.warning(f"Cohen's d calculation failed: {e}")
            
            return effect_sizes
            
        except Exception as e:
            logger.error(f"Effect size calculation failed: {e}")
            return {'error': f'Effect size calculation failed: {str(e)}'}
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta squared effect size"""
        if eta_squared < 0.01:
            return 'Negligible effect'
        elif eta_squared < 0.06:
            return 'Small effect'
        elif eta_squared < 0.14:
            return 'Medium effect'
        else:
            return 'Large effect'
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return 'Negligible effect'
        elif cohens_d < 0.5:
            return 'Small effect'
        elif cohens_d < 0.8:
            return 'Medium effect'
        else:
            return 'Large effect'
    
    def perform_correlation_analysis(self, data: pd.DataFrame, 
                                   score_columns: List[str]) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        if len(score_columns) < 2:
            return {'error': 'Need at least 2 score columns for correlation analysis'}
        
        results = {}
        
        for method in ['pearson', 'spearman', 'kendall']:
            try:
                if method == 'pearson':
                    corr_matrix = data[score_columns].corr(method='pearson')
                elif method == 'spearman':
                    corr_matrix = data[score_columns].corr(method='spearman')
                elif method == 'kendall':
                    corr_matrix = data[score_columns].corr(method='kendall')
                
                # Calculate p-values for correlations
                p_values = pd.DataFrame(index=score_columns, columns=score_columns)
                
                for i, col1 in enumerate(score_columns):
                    for j, col2 in enumerate(score_columns):
                        if i != j:
                            clean_data = data[[col1, col2]].dropna()
                            if len(clean_data) > 3:
                                if method == 'pearson':
                                    _, p_val = pearsonr(clean_data[col1], clean_data[col2])
                                elif method == 'spearman':
                                    _, p_val = spearmanr(clean_data[col1], clean_data[col2])
                                elif method == 'kendall':
                                    _, p_val = kendalltau(clean_data[col1], clean_data[col2])
                                p_values.loc[col1, col2] = p_val
                        else:
                            p_values.loc[col1, col2] = 0.0
                
                results[method] = {
                    'correlation_matrix': corr_matrix,
                    'p_values': p_values.astype(float),
                    'significant_correlations': (p_values.astype(float) < 0.05).sum().sum() - len(score_columns)
                }
                
            except Exception as e:
                logger.error(f"{method} correlation analysis failed: {e}")
                results[method] = {'error': f'{method} correlation failed: {str(e)}'}
        
        return results
    
    def generate_comprehensive_report(self, data: pd.DataFrame, score_column: str,
                                    group_column: str, score_columns: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis report"""
        
        report = {
            'data_summary': {
                'n_samples': len(data),
                'n_groups': data[group_column].nunique(),
                'missing_scores': data[score_column].isnull().sum(),
                'missing_groups': data[group_column].isnull().sum()
            }
        }
        
        # Normality tests
        report['normality'] = {}
        for test in ['shapiro_wilk', 'dagostino_pearson']:
            report['normality'][test] = self.check_normality(data[score_column], test)
        
        # Variance homogeneity
        report['variance_homogeneity'] = {}
        for test in ['levene', 'bartlett']:
            report['variance_homogeneity'][test] = self.check_variance_homogeneity(
                data, score_column, group_column, test
            )
        
        # Parametric tests
        report['parametric_tests'] = self.perform_parametric_tests(data, score_column, group_column)
        
        # Non-parametric tests
        report['non_parametric_tests'] = self.perform_non_parametric_tests(data, score_column, group_column)
        
        # Effect sizes
        report['effect_sizes'] = self.calculate_effect_sizes(data, score_column, group_column)
        
        # Correlation analysis (if multiple scores provided)
        if score_columns and len(score_columns) > 1:
            report['correlations'] = self.perform_correlation_analysis(data, score_columns)
        
        return report


def create_advanced_statistics_ui(data: pd.DataFrame, score_column: str, 
                                group_column: str) -> Dict[str, Any]:
    """Create advanced statistics UI and return analysis results"""
    
    analyzer = AdvancedStatisticalAnalyzer()
    
    st.markdown("### 🧮 Advanced Statistical Analysis")
    
    # Analysis options
    st.markdown("**Select Statistical Analyses:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        check_assumptions = st.checkbox("📊 Check Assumptions", value=True,
                                      help="Test normality and variance homogeneity")
        parametric_tests = st.checkbox("🔢 Parametric Tests", value=True,
                                     help="ANOVA, t-tests, post-hoc tests")
    
    with col2:
        non_parametric_tests = st.checkbox("📈 Non-parametric Tests", value=True,
                                         help="Kruskal-Wallis, Mann-Whitney U")
        effect_sizes = st.checkbox("📏 Effect Sizes", value=True,
                                 help="Calculate effect size measures")
    
    with col3:
        # Multi-score correlation analysis option
        all_numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        correlation_analysis = st.checkbox("🔗 Correlation Analysis", value=False,
                                         help="Analyze correlations between multiple scores")
        
        if correlation_analysis:
            selected_scores = st.multiselect(
                "Select scores for correlation analysis",
                options=all_numeric_cols,
                default=[score_column] + [col for col in all_numeric_cols[:3] if col != score_column],
                help="Choose multiple scores for correlation analysis"
            )
        else:
            selected_scores = [score_column]
    
    # Significance level
    alpha_level = st.selectbox(
        "Significance Level (α)",
        options=[0.05, 0.01, 0.001],
        index=0,
        help="Statistical significance threshold"
    )
    
    # Run analysis button
    if st.button("🚀 Run Advanced Statistical Analysis", type="primary"):
        
        with st.spinner("Performing comprehensive statistical analysis..."):
            # Generate comprehensive report
            analysis_results = analyzer.generate_comprehensive_report(
                data, score_column, group_column, selected_scores if correlation_analysis else None
            )
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Advanced Statistical Results")
        
        # Data summary
        summary = analysis_results['data_summary']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", summary['n_samples'])
        with col2:
            st.metric("Groups", summary['n_groups'])
        with col3:
            st.metric("Missing Scores", summary['missing_scores'])
        with col4:
            st.metric("Missing Groups", summary['missing_groups'])
        
        # Assumption testing
        if check_assumptions:
            st.markdown("### 📋 Statistical Assumptions")
            
            # Normality tests
            st.markdown("**Normality Tests**")
            normality_results = analysis_results['normality']
            
            normality_summary = []
            for test_name, result in normality_results.items():
                if 'error' not in result:
                    normality_summary.append({
                        'Test': result['test_name'],
                        'Statistic': f"{result['statistic']:.4f}",
                        'p-value': f"{result['p_value']:.6f}",
                        'Normal?': '✅ Yes' if result['is_normal'] else '❌ No',
                        'Interpretation': result['interpretation']
                    })
            
            if normality_summary:
                st.dataframe(pd.DataFrame(normality_summary), use_container_width=True)
            
            # Variance homogeneity tests
            st.markdown("**Variance Homogeneity Tests**")
            variance_results = analysis_results['variance_homogeneity']
            
            variance_summary = []
            for test_name, result in variance_results.items():
                if 'error' not in result:
                    variance_summary.append({
                        'Test': result['test_name'],
                        'Statistic': f"{result['statistic']:.4f}",
                        'p-value': f"{result['p_value']:.6f}",
                        'Homogeneous?': '✅ Yes' if result['homogeneous'] else '❌ No',
                        'Interpretation': result['interpretation']
                    })
            
            if variance_summary:
                st.dataframe(pd.DataFrame(variance_summary), use_container_width=True)
        
        # Parametric tests
        if parametric_tests:
            st.markdown("### 🔢 Parametric Tests")
            
            param_results = analysis_results['parametric_tests']
            
            if 'one_way_anova' in param_results and 'error' not in param_results['one_way_anova']:
                anova_result = param_results['one_way_anova']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("F-statistic", f"{anova_result['f_statistic']:.4f}")
                    st.metric("p-value", f"{anova_result['p_value']:.6f}")
                with col2:
                    st.metric("Significant?", "✅ Yes" if anova_result['significant'] else "❌ No")
                    st.metric("DF", f"{anova_result['degrees_freedom']}")
                
                st.info(anova_result['interpretation'])
                
                # Post-hoc results
                if 'tukey_hsd' in param_results and 'error' not in param_results['tukey_hsd']:
                    with st.expander("📊 Post-hoc Test Results", expanded=False):
                        st.write("Tukey HSD test completed - see detailed results in exported data")
        
        # Non-parametric tests
        if non_parametric_tests:
            st.markdown("### 📈 Non-parametric Tests")
            
            nonparam_results = analysis_results['non_parametric_tests']
            
            # Kruskal-Wallis results
            if 'kruskal_wallis' in nonparam_results and 'error' not in nonparam_results['kruskal_wallis']:
                kw_result = nonparam_results['kruskal_wallis']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("H-statistic", f"{kw_result['h_statistic']:.4f}")
                    st.metric("p-value", f"{kw_result['p_value']:.6f}")
                with col2:
                    st.metric("Significant?", "✅ Yes" if kw_result['significant'] else "❌ No")
                    st.metric("DF", kw_result['degrees_freedom'])
                
                st.info(kw_result['interpretation'])
            
            # Mann-Whitney pairwise results
            if 'mann_whitney' in nonparam_results and 'pairwise_results' in nonparam_results['mann_whitney']:
                with st.expander("🔍 Pairwise Comparisons (Mann-Whitney U)", expanded=False):
                    pairwise_data = nonparam_results['mann_whitney']['pairwise_results']
                    
                    if pairwise_data:
                        pairwise_df = pd.DataFrame(pairwise_data)
                        pairwise_df['Significant?'] = pairwise_df['significant'].apply(lambda x: '✅ Yes' if x else '❌ No')
                        pairwise_df['p-value'] = pairwise_df['p_value'].apply(lambda x: f"{x:.6f}")
                        pairwise_df['u-statistic'] = pairwise_df['u_statistic'].apply(lambda x: f"{x:.2f}")
                        
                        display_df = pairwise_df[['group1', 'group2', 'u-statistic', 'p-value', 'Significant?']]
                        display_df.columns = ['Group 1', 'Group 2', 'U-statistic', 'p-value', 'Significant?']
                        
                        st.dataframe(display_df, use_container_width=True)
        
        # Effect sizes
        if effect_sizes:
            st.markdown("### 📏 Effect Sizes")
            
            effect_results = analysis_results['effect_sizes']
            
            if effect_results and 'error' not in effect_results:
                effect_summary = []
                
                for effect_name, effect_data in effect_results.items():
                    effect_summary.append({
                        'Effect Size': effect_name.replace('_', ' ').title(),
                        'Value': f"{effect_data['value']:.4f}",
                        'Interpretation': effect_data['interpretation'],
                        'Description': effect_data['description']
                    })
                
                if effect_summary:
                    st.dataframe(pd.DataFrame(effect_summary), use_container_width=True)
        
        # Correlation analysis
        if correlation_analysis and 'correlations' in analysis_results:
            st.markdown("### 🔗 Correlation Analysis")
            
            corr_results = analysis_results['correlations']
            
            # Display correlation matrices
            for method, results in corr_results.items():
                if 'error' not in results:
                    with st.expander(f"📊 {method.title()} Correlation", expanded=(method == 'pearson')):
                        
                        corr_matrix = results['correlation_matrix']
                        p_matrix = results['p_values']
                        
                        # Create heatmap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Create annotations with correlation and significance
                        annotations = corr_matrix.round(3).astype(str)
                        for i in range(len(corr_matrix.index)):
                            for j in range(len(corr_matrix.columns)):
                                if i != j:
                                    p_val = p_matrix.iloc[i, j]
                                    if p_val < 0.001:
                                        sig = '***'
                                    elif p_val < 0.01:
                                        sig = '**'
                                    elif p_val < 0.05:
                                        sig = '*'
                                    else:
                                        sig = ''
                                    annotations.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.3f}{sig}"
                        
                        sns.heatmap(corr_matrix, annot=annotations, fmt='', cmap='RdBu_r', center=0,
                                  square=True, ax=ax, cbar_kws={'shrink': 0.8})
                        ax.set_title(f'{method.title()} Correlation Matrix\n(*p<0.05, **p<0.01, ***p<0.001)')
                        
                        st.pyplot(fig)
                        
                        # Summary statistics
                        n_significant = results['significant_correlations']
                        total_comparisons = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1)
                        
                        st.write(f"**{n_significant}** out of **{total_comparisons}** correlations are significant (p < 0.05)")
        
        # Recommendations
        st.markdown("### 💡 Statistical Recommendations")
        
        recommendations = []
        
        # Based on normality
        if check_assumptions and 'normality' in analysis_results:
            normal_tests = [r for r in analysis_results['normality'].values() if 'is_normal' in r]
            if normal_tests:
                normal_count = sum(1 for r in normal_tests if r['is_normal'])
                if normal_count == 0:
                    recommendations.append("🔸 **Non-normal data detected** - prefer non-parametric tests (Kruskal-Wallis, Mann-Whitney)")
                elif normal_count == len(normal_tests):
                    recommendations.append("🔸 **Normal distribution confirmed** - parametric tests (ANOVA, t-tests) are appropriate")
                else:
                    recommendations.append("🔸 **Mixed normality results** - consider both parametric and non-parametric approaches")
        
        # Based on variance homogeneity
        if check_assumptions and 'variance_homogeneity' in analysis_results:
            variance_tests = [r for r in analysis_results['variance_homogeneity'].values() if 'homogeneous' in r]
            if variance_tests:
                homogeneous_count = sum(1 for r in variance_tests if r['homogeneous'])
                if homogeneous_count == 0:
                    recommendations.append("🔸 **Unequal variances detected** - consider Welch's ANOVA or non-parametric alternatives")
        
        # Based on sample size
        if summary['n_samples'] < 30:
            recommendations.append("🔸 **Small sample size** - interpret results cautiously and prefer exact tests")
        elif summary['n_samples'] > 1000:
            recommendations.append("🔸 **Large sample size** - statistical tests have high power, focus on effect sizes")
        
        # Based on missing data
        missing_percent = (summary['missing_scores'] + summary['missing_groups']) / (summary['n_samples'] * 2) * 100
        if missing_percent > 10:
            recommendations.append("🔸 **Substantial missing data** - consider imputation or sensitivity analysis")
        
        for rec in recommendations:
            st.markdown(rec)
        
        return analysis_results
    
    return {}


# Export key functions
__all__ = [
    'AdvancedStatisticalAnalyzer',
    'create_advanced_statistics_ui'
]
