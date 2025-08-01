# advanced_visualizations.py
"""
Advanced Visualization System for TaxoConserv
Enhanced plotting capabilities with interactive features
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Optional, Tuple, Any, Union
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class AdvancedVisualizationSystem:
    """
    Advanced visualization system with enhanced plotting capabilities
    """
    
    def __init__(self):
        self.plot_categories = {
            'distribution': {
                'name': 'Distribution Plots',
                'plots': ['histogram', 'density', 'violin', 'boxplot', 'ridgeline', 'raincloud']
            },
            'comparison': {
                'name': 'Group Comparison',
                'plots': ['grouped_boxplot', 'swarmplot', 'stripplot', 'barplot', 'errorbar']
            },
            'correlation': {
                'name': 'Correlation & Relationships',
                'plots': ['scatterplot', 'correlation_heatmap', 'pairplot', 'regression']
            },
            'advanced': {
                'name': 'Advanced Statistical',
                'plots': ['forest_plot', 'funnel_plot', 'qq_plot', 'residual_plot']
            },
            'interactive': {
                'name': 'Interactive Visualizations',
                'plots': ['3d_scatter', 'animated_plot', 'dashboard', 'sunburst']
            }
        }
        
        self.color_schemes = {
            'categorical': ['Set1', 'Set2', 'Set3', 'Dark2', 'Paired', 'Accent', 'Pastel1', 'Pastel2'],
            'sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Blues', 'Reds', 'Greens'],
            'diverging': ['RdBu', 'RdYlBu', 'coolwarm', 'bwr', 'seismic', 'PiYG', 'PRGn']
        }
        
        self.plot_themes = {
            'scientific': 'Publication ready scientific plots',
            'modern': 'Modern clean design',
            'dark': 'Dark theme for presentations',
            'colorful': 'Vibrant colors and gradients',
            'minimal': 'Minimalist design'
        }
    
    def create_ridgeline_plot(self, data: pd.DataFrame, score_column: str, 
                            group_column: str, interactive: bool = True) -> Any:
        """Create ridgeline/joyplot for distribution comparison"""
        try:
            if interactive:
                # Plotly ridgeline plot
                fig = go.Figure()
                
                groups = data[group_column].unique()
                colors = px.colors.qualitative.Set3
                
                for i, group in enumerate(groups):
                    if pd.notna(group):
                        group_data = data[data[group_column] == group][score_column].dropna()
                        
                        if len(group_data) > 0:
                            fig.add_trace(go.Violin(
                                y=group_data,
                                name=str(group),
                                side='positive',
                                orientation='v',
                                width=3,
                                points=False,
                                meanline_visible=True,
                                line_color=colors[i % len(colors)],
                                fillcolor=colors[i % len(colors)],
                                opacity=0.7
                            ))
                
                fig.update_layout(
                    title="Ridgeline Plot - Distribution Comparison",
                    xaxis_title="Groups",
                    yaxis_title=score_column.replace('_', ' ').title(),
                    height=600,
                    showlegend=True
                )
                
                return fig
            else:
                # Matplotlib ridgeline using multiple subplots
                groups = [g for g in data[group_column].unique() if pd.notna(g)]
                n_groups = len(groups)
                
                fig, axes = plt.subplots(n_groups, 1, figsize=(10, 2*n_groups), sharex=True)
                if n_groups == 1:
                    axes = [axes]
                
                for i, group in enumerate(groups):
                    group_data = data[data[group_column] == group][score_column].dropna()
                    
                    if len(group_data) > 0:
                        axes[i].fill_between(
                            np.linspace(group_data.min(), group_data.max(), 100),
                            0,
                            np.histogram(group_data, bins=50, density=True)[0][:100],
                            alpha=0.7,
                            label=str(group)
                        )
                        axes[i].set_ylabel(str(group), rotation=0, ha='right')
                        axes[i].set_ylim(0, None)
                
                axes[-1].set_xlabel(score_column.replace('_', ' ').title())
                plt.suptitle('Ridgeline Plot - Distribution Comparison', fontsize=16)
                plt.tight_layout()
                
                return fig
                
        except Exception as e:
            logger.error(f"Ridgeline plot creation failed: {e}")
            return None
    
    def create_raincloud_plot(self, data: pd.DataFrame, score_column: str, 
                            group_column: str, interactive: bool = True) -> Any:
        """Create raincloud plot (combination of violin, box, and strip plots)"""
        try:
            if interactive:
                fig = go.Figure()
                
                groups = data[group_column].unique()
                colors = px.colors.qualitative.Set3
                
                for i, group in enumerate(groups):
                    if pd.notna(group):
                        group_data = data[data[group_column] == group][score_column].dropna()
                        
                        if len(group_data) > 0:
                            # Violin plot (cloud)
                            fig.add_trace(go.Violin(
                                x=[str(group)] * len(group_data),
                                y=group_data,
                                name=f"{group} (distribution)",
                                side='negative',
                                line_color=colors[i % len(colors)],
                                fillcolor=colors[i % len(colors)],
                                opacity=0.6,
                                points=False,
                                showlegend=False
                            ))
                            
                            # Box plot
                            fig.add_trace(go.Box(
                                x=[str(group)],
                                y=group_data,
                                name=f"{group} (quartiles)",
                                line_color=colors[i % len(colors)],
                                fillcolor='rgba(255,255,255,0)',
                                showlegend=False,
                                width=0.3
                            ))
                            
                            # Strip plot (rain)
                            fig.add_trace(go.Scatter(
                                x=[str(group)] * len(group_data),
                                y=group_data,
                                mode='markers',
                                name=f"{group} (data points)",
                                marker=dict(
                                    size=4,
                                    color=colors[i % len(colors)],
                                    opacity=0.6
                                ),
                                showlegend=False
                            ))
                
                fig.update_layout(
                    title="Raincloud Plot - Comprehensive Distribution View",
                    xaxis_title="Groups",
                    yaxis_title=score_column.replace('_', ' ').title(),
                    height=600
                )
                
                return fig
            else:
                # Matplotlib raincloud plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create violin plot
                parts = ax.violinplot([data[data[group_column] == g][score_column].dropna() 
                                     for g in data[group_column].unique() if pd.notna(g)],
                                    positions=range(len(data[group_column].unique())),
                                    widths=0.8,
                                    showmeans=True,
                                    showmedians=True)
                
                # Add strip plot overlay
                sns.stripplot(data=data, x=group_column, y=score_column, 
                            size=3, alpha=0.6, ax=ax)
                
                ax.set_title('Raincloud Plot - Comprehensive Distribution View')
                ax.set_xlabel(group_column.replace('_', ' ').title())
                ax.set_ylabel(score_column.replace('_', ' ').title())
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                return fig
                
        except Exception as e:
            logger.error(f"Raincloud plot creation failed: {e}")
            return None
    
    def create_forest_plot(self, data: pd.DataFrame, score_column: str, 
                          group_column: str) -> Any:
        """Create forest plot showing means and confidence intervals"""
        try:
            # Calculate statistics for each group
            group_stats = []
            
            for group in data[group_column].unique():
                if pd.notna(group):
                    group_data = data[data[group_column] == group][score_column].dropna()
                    
                    if len(group_data) >= 2:
                        mean = group_data.mean()
                        std = group_data.std()
                        n = len(group_data)
                        se = std / np.sqrt(n)
                        
                        # 95% confidence interval
                        ci_lower = mean - 1.96 * se
                        ci_upper = mean + 1.96 * se
                        
                        group_stats.append({
                            'group': str(group),
                            'mean': mean,
                            'ci_lower': ci_lower,
                            'ci_upper': ci_upper,
                            'n': n
                        })
            
            if not group_stats:
                return None
            
            # Create forest plot
            fig = go.Figure()
            
            y_positions = list(range(len(group_stats)))
            
            # Add confidence intervals
            for i, stats in enumerate(group_stats):
                fig.add_trace(go.Scatter(
                    x=[stats['ci_lower'], stats['ci_upper']],
                    y=[i, i],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False
                ))
                
                # Add mean points
                fig.add_trace(go.Scatter(
                    x=[stats['mean']],
                    y=[i],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name=f"{stats['group']} (n={stats['n']})",
                    showlegend=True
                ))
            
            # Add vertical line at overall mean
            overall_mean = data[score_column].mean()
            fig.add_vline(x=overall_mean, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Forest Plot - Group Means with 95% Confidence Intervals",
                xaxis_title=score_column.replace('_', ' ').title(),
                yaxis=dict(
                    tickmode='array',
                    tickvals=y_positions,
                    ticktext=[stats['group'] for stats in group_stats]
                ),
                height=max(400, len(group_stats) * 50)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Forest plot creation failed: {e}")
            return None
    
    def create_3d_scatter(self, data: pd.DataFrame, score_columns: List[str], 
                         group_column: str) -> Any:
        """Create 3D scatter plot for multi-dimensional data"""
        try:
            if len(score_columns) < 3:
                return None
            
            x_col, y_col, z_col = score_columns[:3]
            
            fig = px.scatter_3d(
                data, 
                x=x_col, 
                y=y_col, 
                z=z_col,
                color=group_column,
                title="3D Conservation Score Visualization",
                width=800,
                height=600
            )
            
            fig.update_traces(marker_size=3)
            
            return fig
            
        except Exception as e:
            logger.error(f"3D scatter plot creation failed: {e}")
            return None
    
    def create_animated_plot(self, data: pd.DataFrame, score_column: str, 
                           group_column: str, time_column: Optional[str] = None) -> Any:
        """Create animated plot showing changes over time or conditions"""
        try:
            if time_column and time_column in data.columns:
                # Animated by time column
                fig = px.box(
                    data, 
                    x=group_column, 
                    y=score_column,
                    animation_frame=time_column,
                    title=f"Animated {score_column.replace('_', ' ').title()} Over {time_column.replace('_', ' ').title()}"
                )
            else:
                # Create animated plot by splitting data into chunks
                data_copy = data.copy()
                n_chunks = min(5, len(data) // 20)  # Create up to 5 frames
                
                if n_chunks > 1:
                    chunk_size = len(data) // n_chunks
                    data_copy['frame'] = data_copy.index // chunk_size
                    
                    fig = px.box(
                        data_copy, 
                        x=group_column, 
                        y=score_column,
                        animation_frame='frame',
                        title=f"Animated {score_column.replace('_', ' ').title()} Distribution"
                    )
                else:
                    return None
            
            fig.update_layout(height=600)
            
            return fig
            
        except Exception as e:
            logger.error(f"Animated plot creation failed: {e}")
            return None
    
    def create_sunburst_plot(self, data: pd.DataFrame, hierarchy_columns: List[str],
                           score_column: str) -> Any:
        """Create sunburst plot for hierarchical data"""
        try:
            if len(hierarchy_columns) < 2:
                return None
            
            # Prepare hierarchical data
            hierarchy_data = data[hierarchy_columns + [score_column]].dropna()
            
            # Create path column for sunburst
            hierarchy_data['path'] = hierarchy_data[hierarchy_columns].apply(
                lambda x: ' / '.join(x.astype(str)), axis=1
            )
            
            # Aggregate scores by hierarchy
            agg_data = hierarchy_data.groupby(hierarchy_columns)[score_column].agg(['mean', 'count']).reset_index()
            
            # Create sunburst plot
            fig = px.sunburst(
                agg_data,
                path=hierarchy_columns,
                values='count',
                color='mean',
                color_continuous_scale='viridis',
                title="Hierarchical Distribution of Conservation Scores"
            )
            
            fig.update_layout(height=600)
            
            return fig
            
        except Exception as e:
            logger.error(f"Sunburst plot creation failed: {e}")
            return None
    
    def create_qq_plot(self, data: pd.DataFrame, score_column: str) -> Any:
        """Create Q-Q plot for normality assessment"""
        try:
            from scipy import stats
            
            clean_data = data[score_column].dropna()
            
            if len(clean_data) < 10:
                return None
            
            # Create Q-Q plot
            fig = go.Figure()
            
            # Calculate quantiles
            sorted_data = np.sort(clean_data)
            n = len(sorted_data)
            theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
            
            # Add scatter points
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sorted_data,
                mode='markers',
                name='Data points',
                marker=dict(size=4, opacity=0.7)
            ))
            
            # Add reference line
            slope, intercept = np.polyfit(theoretical_quantiles, sorted_data, 1)
            line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
            line_y = slope * line_x + intercept
            
            fig.add_trace(go.Scatter(
                x=line_x,
                y=line_y,
                mode='lines',
                name='Reference line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title="Q-Q Plot - Normality Assessment",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles",
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Q-Q plot creation failed: {e}")
            return None
    
    def apply_theme(self, fig: Any, theme: str, plot_type: str = 'plotly') -> Any:
        """Apply visual theme to plot"""
        try:
            if plot_type == 'plotly':
                if theme == 'scientific':
                    fig.update_layout(
                        template='plotly_white',
                        font=dict(family='Arial', size=12),
                        title_font=dict(size=16, family='Arial'),
                        plot_bgcolor='white',
                        paper_bgcolor='white'
                    )
                elif theme == 'dark':
                    fig.update_layout(
                        template='plotly_dark',
                        font=dict(color='white'),
                        paper_bgcolor='#2e2e2e',
                        plot_bgcolor='#2e2e2e'
                    )
                elif theme == 'modern':
                    fig.update_layout(
                        template='simple_white',
                        font=dict(family='Helvetica', size=11),
                        title_font=dict(size=18, family='Helvetica'),
                    )
                elif theme == 'minimal':
                    fig.update_layout(
                        template='none',
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
                    )
            
            return fig
            
        except Exception as e:
            logger.error(f"Theme application failed: {e}")
            return fig


def create_advanced_visualization_ui(data: pd.DataFrame, score_column: str, 
                                   group_column: str) -> Dict[str, Any]:
    """Create advanced visualization UI"""
    
    viz_system = AdvancedVisualizationSystem()
    
    st.markdown("### 🎨 Advanced Visualization Options")
    
    # Plot category selection
    st.markdown("**Select Visualization Category:**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox(
            "Visualization Category",
            options=list(viz_system.plot_categories.keys()),
            format_func=lambda x: viz_system.plot_categories[x]['name'],
            help="Choose the type of visualization you want to create"
        )
        
        # Plot type selection within category
        available_plots = viz_system.plot_categories[selected_category]['plots']
        selected_plot = st.selectbox(
            "Specific Plot Type",
            options=available_plots,
            help="Select the specific plot type to generate"
        )
    
    with col2:
        # Visual options
        st.markdown("**Visual Options:**")
        
        interactive_mode = st.checkbox("Interactive Plot", value=True)
        
        theme = st.selectbox(
            "Theme",
            options=list(viz_system.plot_themes.keys()),
            format_func=lambda x: viz_system.plot_themes[x]
        )
        
        # Color scheme selection
        color_type = st.selectbox(
            "Color Scheme Type",
            options=['categorical', 'sequential', 'diverging']
        )
        
        color_scheme = st.selectbox(
            "Color Scheme",
            options=viz_system.color_schemes[color_type]
        )
    
    # Additional options based on plot type
    st.markdown("---")
    st.markdown("**Plot-Specific Options:**")
    
    additional_options = {}
    
    if selected_plot in ['3d_scatter', 'pairplot']:
        # Multi-score selection for 3D and pairplot
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        additional_options['score_columns'] = st.multiselect(
            "Select Multiple Scores",
            options=numeric_cols,
            default=[score_column] + [col for col in numeric_cols[:2] if col != score_column],
            help="Select multiple score columns for multi-dimensional visualization"
        )
    
    elif selected_plot == 'sunburst':
        # Hierarchy columns for sunburst
        text_cols = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        additional_options['hierarchy_columns'] = st.multiselect(
            "Hierarchy Columns",
            options=text_cols,
            default=[group_column] + [col for col in text_cols[:2] if col != group_column],
            help="Select columns to create hierarchical structure"
        )
    
    elif selected_plot == 'animated_plot':
        # Time column for animation
        all_cols = list(data.columns)
        additional_options['time_column'] = st.selectbox(
            "Animation Variable (optional)",
            options=[None] + all_cols,
            help="Select a column to animate over (optional)"
        )
    
    # Generate visualization button
    if st.button("🎨 Generate Advanced Visualization", type="primary"):
        
        with st.spinner(f"Creating {selected_plot} visualization..."):
            
            fig = None
            
            try:
                # Generate the selected plot
                if selected_plot == 'ridgeline':
                    fig = viz_system.create_ridgeline_plot(data, score_column, group_column, interactive_mode)
                
                elif selected_plot == 'raincloud':
                    fig = viz_system.create_raincloud_plot(data, score_column, group_column, interactive_mode)
                
                elif selected_plot == 'forest_plot':
                    fig = viz_system.create_forest_plot(data, score_column, group_column)
                
                elif selected_plot == '3d_scatter':
                    if 'score_columns' in additional_options and len(additional_options['score_columns']) >= 3:
                        fig = viz_system.create_3d_scatter(data, additional_options['score_columns'], group_column)
                    else:
                        st.error("3D scatter plot requires at least 3 score columns")
                
                elif selected_plot == 'animated_plot':
                    fig = viz_system.create_animated_plot(
                        data, score_column, group_column, 
                        additional_options.get('time_column', None)  # type: ignore
                    )
                
                elif selected_plot == 'sunburst':
                    if 'hierarchy_columns' in additional_options and len(additional_options['hierarchy_columns']) >= 2:
                        fig = viz_system.create_sunburst_plot(
                            data, additional_options['hierarchy_columns'], score_column
                        )
                    else:
                        st.error("Sunburst plot requires at least 2 hierarchy columns")
                
                elif selected_plot == 'qq_plot':
                    fig = viz_system.create_qq_plot(data, score_column)
                
                # Standard plots with enhanced styling
                elif selected_plot == 'histogram':
                    if interactive_mode:
                        fig = px.histogram(data, x=score_column, color=group_column, 
                                         title=f"Enhanced Histogram - {score_column.replace('_', ' ').title()}")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=data, x=score_column, hue=group_column, ax=ax)
                        ax.set_title(f"Enhanced Histogram - {score_column.replace('_', ' ').title()}")
                
                elif selected_plot == 'violin':
                    if interactive_mode:
                        fig = px.violin(data, x=group_column, y=score_column, box=True,
                                       title=f"Enhanced Violin Plot - {score_column.replace('_', ' ').title()}")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.violinplot(data=data, x=group_column, y=score_column, ax=ax)
                        ax.set_title(f"Enhanced Violin Plot - {score_column.replace('_', ' ').title()}")
                
                # Apply theme if figure was created
                if fig is not None:
                    if hasattr(fig, 'update_layout'):  # Plotly figure
                        fig = viz_system.apply_theme(fig, theme, 'plotly')
                    
                    # Display the figure
                    st.markdown("---")
                    st.markdown(f"## 🎨 {selected_plot.replace('_', ' ').title()} Visualization")
                    
                    if hasattr(fig, 'update_layout'):  # Plotly figure
                        st.plotly_chart(fig, use_container_width=True)
                    else:  # Matplotlib figure
                        st.pyplot(fig)  # type: ignore
                    
                    # Provide download option
                    st.markdown("### 💾 Download Options")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if hasattr(fig, 'to_html'):  # type: ignore
                            html_str = fig.to_html(include_plotlyjs='cdn')  # type: ignore
                            st.download_button(
                                "📄 Download as HTML",
                                data=html_str,
                                file_name=f"{selected_plot}_{score_column}.html",
                                mime="text/html"
                            )
                    
                    with col2:
                        if hasattr(fig, 'write_image'):  # type: ignore
                            try:
                                img_bytes = fig.to_image(format="png", width=1200, height=800)  # type: ignore
                                st.download_button(
                                    "🖼️ Download as PNG",
                                    data=img_bytes,
                                    file_name=f"{selected_plot}_{score_column}.png",
                                    mime="image/png"
                                )
                            except:
                                st.info("PNG download requires kaleido package")
                        else:
                            # Matplotlib figure
                            buf = BytesIO()
                            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')  # type: ignore
                            st.download_button(
                                "🖼️ Download as PNG",
                                data=buf.getvalue(),
                                file_name=f"{selected_plot}_{score_column}.png",
                                mime="image/png"
                            )
                    
                    with col3:
                        # Save plot configuration
                        plot_config = {
                            'plot_type': selected_plot,
                            'score_column': score_column,
                            'group_column': group_column,
                            'theme': theme,
                            'color_scheme': color_scheme,
                            'interactive': interactive_mode,
                            'additional_options': additional_options
                        }
                        
                        config_str = str(plot_config)
                        st.download_button(
                            "⚙️ Download Config",
                            data=config_str,
                            file_name=f"{selected_plot}_config.txt",
                            mime="text/plain"
                        )
                    
                    # Interpretation and insights
                    st.markdown("### 💡 Visualization Insights")
                    
                    insights = []
                    
                    if selected_plot in ['ridgeline', 'raincloud', 'violin']:
                        insights.append("🔍 **Distribution Shape**: Examine the shape of distributions across groups")
                        insights.append("📊 **Outliers**: Look for data points that fall outside typical patterns")
                        insights.append("📈 **Variability**: Compare the spread of data between groups")
                    
                    elif selected_plot == 'forest_plot':
                        insights.append("🎯 **Confidence Intervals**: Wider intervals indicate more uncertainty")
                        insights.append("📊 **Effect Comparison**: Groups whose intervals don't overlap may differ significantly")
                        insights.append("📍 **Reference Line**: Shows overall mean for comparison")
                    
                    elif selected_plot == '3d_scatter':
                        insights.append("🌐 **Multi-dimensional Patterns**: Explore relationships between three scores simultaneously")
                        insights.append("🔍 **Clustering**: Look for natural groupings in 3D space")
                        insights.append("📐 **Correlation Structure**: Examine how multiple variables relate")
                    
                    elif selected_plot == 'qq_plot':
                        insights.append("📏 **Normality Assessment**: Points close to the line suggest normal distribution")
                        insights.append("📊 **Distribution Tails**: Deviations at ends indicate heavy or light tails")
                        insights.append("🔍 **Systematic Patterns**: S-curves or other patterns reveal distribution characteristics")
                    
                    for insight in insights:
                        st.markdown(insight)
                
                else:
                    st.error(f"Failed to create {selected_plot} visualization. Please check your data and options.")
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                logger.error(f"Visualization creation failed: {e}")
        
        return {'plot_type': selected_plot, 'figure': fig, 'config': additional_options}
    
    return {}


# Export key functions
__all__ = [
    'AdvancedVisualizationSystem',
    'create_advanced_visualization_ui'
]
