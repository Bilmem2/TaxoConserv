# score_plot_mapping.py
"""
Advanced score-to-plot mapping system for TaxoConserv
Automatically suggests optimal visualization based on conservation score type
"""

from typing import Dict, List, Tuple
import re


class ScorePlotMapper:
    """
    Intelligent mapping system that suggests optimal plots for conservation scores
    """
    
    def __init__(self):
        # Core mapping: score_pattern -> (recommended_plots, explanation)
        self.score_mappings = {
            # Phylogenetic Conservation Scores
            'phylop': {
                'plots': ['violin', 'boxplot', 'kde'],
                'primary': 'violin',
                'explanation': 'PhyloP scores show continuous phylogenetic conservation with good distribution spread - violin plots reveal the full data distribution.',
                'score_type': 'continuous',
                'typical_range': '(-20, 10)',
                'interpretation': 'Positive values indicate conservation, negative indicate acceleration'
            },
            
            'phastcons': {
                'plots': ['boxplot', 'violin', 'histogram'],
                'primary': 'boxplot',
                'explanation': 'PhastCons scores are probabilities (0-1) showing conserved elements - boxplots effectively show quartiles and outliers.',
                'score_type': 'probability',
                'typical_range': '[0, 1]',
                'interpretation': 'Higher values indicate stronger conservation probability'
            },
            
            # GERP Scores
            'gerp': {
                'plots': ['histogram', 'barplot', 'boxplot'],
                'primary': 'histogram',
                'explanation': 'GERP scores often have bimodal distributions - histograms best reveal the underlying distribution patterns.',
                'score_type': 'continuous',
                'typical_range': '(-12, 6)',
                'interpretation': 'Positive values indicate constraint (conservation)'
            },
            
            # Variant Impact Scores
            'revel': {
                'plots': ['swarm', 'kde', 'violin'],
                'primary': 'swarm',
                'explanation': 'REVEL scores (0-1) for pathogenicity prediction - swarm plots show individual variants and clustering patterns.',
                'score_type': 'probability',
                'typical_range': '[0, 1]',
                'interpretation': 'Higher values indicate higher pathogenicity prediction'
            },
            
            'cadd': {
                'plots': ['kde', 'violin', 'histogram'],
                'primary': 'kde',
                'explanation': 'CADD scores show variant deleteriousness with skewed distributions - KDE plots smooth the distribution for better interpretation.',
                'score_type': 'continuous',
                'typical_range': '[0, 40+]',
                'interpretation': 'Higher scores indicate more deleterious variants'
            },
            
            # SIFT Scores
            'sift': {
                'plots': ['boxplot', 'histogram', 'violin'],
                'primary': 'boxplot',
                'explanation': 'SIFT scores (0-1) predict functional impact - boxplots show the tolerance distribution effectively.',
                'score_type': 'probability',
                'typical_range': '[0, 1]',
                'interpretation': 'Lower values indicate deleterious variants (< 0.05 typically)'
            },
            
            # PolyPhen Scores
            'polyphen': {
                'plots': ['barplot', 'histogram', 'boxplot'],
                'primary': 'barplot',
                'explanation': 'PolyPhen scores often cluster in discrete categories - bar plots show categorical distributions well.',
                'score_type': 'probability',
                'typical_range': '[0, 1]',
                'interpretation': 'Higher values indicate more damaging variants'
            },
            
            # Generic Conservation
            'conservation': {
                'plots': ['boxplot', 'violin', 'kde'],
                'primary': 'boxplot',
                'explanation': 'Generic conservation scores typically show normal-like distributions - boxplots provide clear statistical summaries.',
                'score_type': 'continuous',
                'typical_range': 'varies',
                'interpretation': 'Higher values typically indicate stronger conservation'
            },
            
            # Evolutionary Scores
            'evolutionary': {
                'plots': ['violin', 'kde', 'boxplot'],
                'primary': 'violin',
                'explanation': 'Evolutionary scores often have complex distributions - violin plots reveal multimodal patterns.',
                'score_type': 'continuous',
                'typical_range': 'varies',
                'interpretation': 'Depends on specific evolutionary metric'
            }
        }
        
        # Fallback for unknown scores
        self.default_mapping = {
            'plots': ['boxplot', 'violin', 'histogram'],
            'primary': 'boxplot',
            'explanation': 'Boxplot provides a robust visualization for unknown score types, showing quartiles and outliers.',
            'score_type': 'unknown',
            'typical_range': 'unknown',
            'interpretation': 'Interpretation depends on the specific score definition'
        }
    
    def detect_score_type(self, column_name: str) -> str:
        """
        Detect score type from column name using pattern matching
        """
        col_lower = column_name.lower().replace('_', '').replace(' ', '')
        
        for score_type in self.score_mappings.keys():
            if score_type in col_lower:
                return score_type
        
        # Additional pattern matching
        if any(pattern in col_lower for pattern in ['conserv', 'cons']):
            return 'conservation'
        elif any(pattern in col_lower for pattern in ['evol', 'phylo']):
            return 'evolutionary'
        
        return 'unknown'
    
    def get_recommendations(self, column_name: str) -> Dict:
        """
        Get plot recommendations for a given score column
        """
        score_type = self.detect_score_type(column_name)
        
        if score_type in self.score_mappings:
            mapping = self.score_mappings[score_type].copy()
        else:
            mapping = self.default_mapping.copy()
        
        mapping['detected_type'] = score_type
        mapping['column_name'] = column_name
        
        return mapping
    
    def get_plot_explanation(self, column_name: str, plot_type: str) -> str:
        """
        Get detailed explanation for why a specific plot is good for this score
        """
        recommendations = self.get_recommendations(column_name)
        base_explanation = recommendations['explanation']
        
        plot_explanations = {
            'violin': 'Shows full distribution shape including density and quartiles',
            'boxplot': 'Displays quartiles, median, and outliers clearly',
            'histogram': 'Reveals underlying distribution patterns and modality',
            'kde': 'Provides smooth density estimation for continuous data',
            'swarm': 'Shows individual data points and clustering patterns',
            'barplot': 'Effective for categorical or discrete value distributions',
            'heatmap': 'Good for visualizing patterns across multiple dimensions',
            'correlation': 'Shows relationships between multiple variables'
        }
        
        plot_specific = plot_explanations.get(plot_type, 'General visualization approach')
        
        return f"{base_explanation} {plot_specific}."
    
    def suggest_best_plot(self, column_name: str) -> str:
        """
        Get the single best plot recommendation
        """
        recommendations = self.get_recommendations(column_name)
        return recommendations['primary']
    
    def get_all_suggested_plots(self, column_name: str) -> List[str]:
        """
        Get all recommended plots in order of preference
        """
        recommendations = self.get_recommendations(column_name)
        return recommendations['plots']


# Global instance
score_plot_mapper = ScorePlotMapper()


def suggest_plot_for_score(score_column: str) -> Tuple[List[str], str]:
    """
    Main function to get plot suggestions for a score column
    Returns: (recommended_plots, explanation)
    """
    recommendations = score_plot_mapper.get_recommendations(score_column)
    return recommendations['plots'], recommendations['explanation']


def get_enhanced_plot_info(score_column: str) -> Dict:
    """
    Get comprehensive information about a score column
    """
    return score_plot_mapper.get_recommendations(score_column)


if __name__ == "__main__":
    # Test the mapper
    test_columns = ['phyloP_score', 'phastCons', 'GERP_RS', 'REVEL_score', 'conservation_score']
    
    for col in test_columns:
        recommendations = score_plot_mapper.get_recommendations(col)
        print(f"\n{col}:")
        print(f"  Detected type: {recommendations['detected_type']}")
        print(f"  Primary plot: {recommendations['primary']}")
        print(f"  All plots: {recommendations['plots']}")
        print(f"  Explanation: {recommendations['explanation']}")
