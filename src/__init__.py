"""
TaxoConserv - Taxonomic Conservation Analysis Tool
Source code modules
"""

__version__ = "1.0.0-alpha"
__author__ = "TaxoConserv Development Team"

# Import core modules
from .input_parser import parse_input, validate_input
from .taxon_grouping import group_by_taxon, calculate_stats, get_group_summary

# Export main functions
__all__ = [
    "parse_input",
    "validate_input",
    "group_by_taxon",
    "calculate_stats",
    "get_group_summary",
]
