"""
score_utils.py
TaxoConserv - Conservation Score Detection and Utilities
"""

from typing import Dict, List

# Conservation Score Detection and Description Dictionary
CONSERVATION_SCORE_PATTERNS = {
    # PhyloP scores
    'phyloP': 'PhyloP conservation score from multiple alignments (e.g. 100-way vertebrates)',
    'phyloP_score': 'PhyloP conservation score from multiple alignments (e.g. 100-way vertebrates)',
    'phylop': 'PhyloP conservation score from multiple alignments (e.g. 100-way vertebrates)',
    'phylop_score': 'PhyloP conservation score from multiple alignments (e.g. 100-way vertebrates)',
    # PhastCons scores
    'phastCons': 'Phylogenetic conservation score (e.g. 100-way, 30-way vertebrate alignments)',
    'phastCons_score': 'Phylogenetic conservation score (e.g. 100-way, 30-way vertebrate alignments)',
    'phastcons': 'Phylogenetic conservation score (e.g. 100-way, 30-way vertebrate alignments)',
    'phastcons_score': 'Phylogenetic conservation score (e.g. 100-way, 30-way vertebrate alignments)',
    # GERP scores
    'GERP': 'Genomic Evolutionary Rate Profiling score',
    'GERP++': 'Genomic Evolutionary Rate Profiling score (enhanced version)',
    'GERP_score': 'Genomic Evolutionary Rate Profiling score',
    'gerp': 'Genomic Evolutionary Rate Profiling score',
    'gerp_score': 'Genomic Evolutionary Rate Profiling score',
    # SiPhy scores
    'SiPhy': 'Site-specific phylogenetic analysis score',
    'SiPhy_score': 'Site-specific phylogenetic analysis score',
    'siphy': 'Site-specific phylogenetic analysis score',
    'siphy_score': 'Site-specific phylogenetic analysis score',
    # Generic conservation scores
    'conservation_score': 'Generic conservation score (evolutionary conservation measure)',
    'conservation': 'Generic conservation score (evolutionary conservation measure)',
    'score': 'Generic score column',
    # Additional common patterns
    'phylop_cons': 'PhyloP conservation score',
    'phastcons_cons': 'PhastCons conservation score',
    'evolutionary_score': 'Evolutionary conservation score',
    'evo_score': 'Evolutionary conservation score',
    'cons_score': 'Conservation score',
    'conservation_measure': 'Conservation measure score'
}

def detect_conservation_scores(data) -> Dict[str, str]:
    """
    Detect conservation score columns in the dataset.
    Args:
        data (pd.DataFrame): Input dataset
    Returns:
        dict: Dictionary with detected scores and their descriptions
    """
    detected_scores = {}
    for col in data.columns:
        col_lower = col.lower()
        # Check for exact matches first
        if col in CONSERVATION_SCORE_PATTERNS:
            detected_scores[col] = CONSERVATION_SCORE_PATTERNS[col]
            continue
        # Check for partial matches
        for pattern, description in CONSERVATION_SCORE_PATTERNS.items():
            if pattern.lower() in col_lower:
                detected_scores[col] = description
                break
    return detected_scores

def get_score_description(score_name: str) -> str:
    """
    Get description for a conservation score.
    Args:
        score_name (str): Name of the score column
    Returns:
        str: Description of the score
    """
    return CONSERVATION_SCORE_PATTERNS.get(score_name, "Numeric score column")

def prioritize_conservation_scores(score_options: List[str], detected_scores: Dict[str, str]) -> List[str]:
    """
    Prioritize conservation score columns in the options list.
    Args:
        score_options (list): List of all numeric columns
        detected_scores (dict): Dictionary of detected conservation scores
    Returns:
        list: Reordered list with conservation scores first
    """
    if not detected_scores:
        return score_options
    conservation_scores = list(detected_scores.keys())
    other_scores = [col for col in score_options if col not in detected_scores]
    return conservation_scores + other_scores 