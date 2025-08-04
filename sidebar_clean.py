# Clean sidebar implementation for TaxoConserv
import streamlit as st
import pandas as pd

def create_clean_sidebar(data):
    """Create a clean, non-duplicate sidebar for TaxoConserv"""
    
    if data is not None:
        # Single unified configuration sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏷️ Analysis Configuration")
        
        # Group column selection
        group_options = [col for col in data.columns if not pd.api.types.is_numeric_dtype(data[col])]
        if not group_options:
            group_options = list(data.columns)
        
        group_column = st.sidebar.selectbox(
            "Grouping Column",
            options=group_options,
            index=group_options.index(st.session_state.get('group_column', group_options[0])) if st.session_state.get('group_column') in group_options else 0,
            help="Select column for taxonomic grouping",
            key="main_group_column"
        )
        
        # Score column selection
        score_options = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        if score_options:
            score_column = st.sidebar.selectbox(
                "Conservation Score Column",
                options=score_options,
                index=score_options.index(st.session_state.get('score_column', score_options[0])) if st.session_state.get('score_column') in score_options else 0,
                help="Select conservation score to analyze",
                key="main_score_column"
            )
        else:
            st.sidebar.error("No numeric columns found for analysis!")
            return None, None

        # Visualization Configuration
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎨 Visualization")
        
        # Get recommended plot types
        def suggest_plot_types(score_column: str) -> tuple:
            col_lower = score_column.lower()
            if any(x in col_lower for x in ["phylop", "phastcons", "conservation"]):
                return ["boxplot", "violin", "kde"], "These scores are continuous and best visualized with distribution plots."
            elif "gerp" in col_lower:
                return ["histogram", "barplot"], "Gerp scores are typically summarized with histograms and barplots."
            elif any(x in col_lower for x in ["revel", "cadd"]):
                return ["swarm", "kde"], "REVEL/CADD scores are suitable for individual and group density visualizations."
            else:
                return ["boxplot"], "Boxplot is recommended by default for general score visualization."

        # Only proceed with visualization if we have a score column
        if score_column:
            recommended_plots, tooltip_text = suggest_plot_types(score_column)
            st.sidebar.markdown("**Recommended plot types:** " + ", ".join([f"`{plot}`" for plot in recommended_plots]))
            
            # Plot type selection
            plot_type = st.sidebar.selectbox(
                "📈 Plot Type",
                options=recommended_plots,
                index=0,
                key='plot_type',
                help="Auto-selected based on your data type"
            )
        else:
            plot_type = None
        
        # Fine Settings
        with st.sidebar.expander("⚙️ Fine Settings", expanded=False):
            # Plot rendering mode
            plot_mode = st.radio(
                "🖥️ Plot Rendering Mode",
                options=["Interactive (Plotly)", "Static (Matplotlib)"],
                index=0,
                key='plot_mode',
                help="Interactive: Web-based plots with zoom/hover\nStatic: Traditional publication-ready plots"
            )
            
            # Color palette selection
            palette_options = ["Set1", "Set2", "Set3", "tab10", "Dark2", "Pastel1", "Accent", "viridis", "plasma", "magma", "inferno", "cividis", "colorblind"]
            color_palette = st.selectbox(
                "🎨 Color Palette",
                options=palette_options,
                index=2,  # Set3
                key='color_palette',
                help="Select color palette for visualizations"
            )
            
            # Statistical display options
            show_statistics = st.checkbox(
                "📊 Show Statistical Results",
                value=True,
                key='show_statistics',
                help="Display statistical analysis results in sidebar"
            )
            
            # Advanced analysis options
            st.markdown("**🔬 Advanced Analysis:**")
            use_multi_score = st.checkbox(
                "Multi-Score Analysis",
                value=False,
                key='use_multi_score',
                help="Compare multiple conservation scores (PhyloP, GERP, etc.)"
            )
            
            # Statistical analysis level
            advanced_stats_level = st.selectbox(
                "Statistical Analysis Level",
                options=["Basic", "Detailed", "Expert"],
                index=0,
                key='advanced_stats_level',
                help="Choose the level of statistical analysis detail"
            )
            
            # Chart quality settings
            st.markdown("**Chart Quality:**")
            chart_dpi = st.slider("Resolution (DPI)", min_value=100, max_value=300, value=150, step=50, key='chart_dpi')
            chart_size = st.selectbox("Chart Size", ["Small", "Medium", "Large"], index=1, key='chart_size')

        # Analysis button
        st.sidebar.markdown("---")
        analysis_enabled = data is not None and group_column and score_column
        
        run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", disabled=not analysis_enabled, key="clean_run_analysis")
        
        # Return all necessary values
        settings = {
            'plot_mode': st.session_state.get('plot_mode', 'Interactive (Plotly)'),
            'color_palette': st.session_state.get('color_palette', 'Set3'),
            'show_statistics': st.session_state.get('show_statistics', True),
            'chart_dpi': st.session_state.get('chart_dpi', 150),
            'chart_size': st.session_state.get('chart_size', 'Medium')
        }
        
        return group_column, score_column, plot_type, run_analysis, settings
        
    return None, None, None, False, {}
