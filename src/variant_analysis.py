"""
Variant Conservation Analysis Module

This module processes VCF files and provides detailed conservation analysis
for individual variants across taxonomic groups.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Optional VCF parsing - try to import PyVCF, fall back to simple parsing
try:
    import vcf
    HAS_PYVCF = True
except ImportError:
    try:
        # Try alternative import names
        import pyvcf as vcf
        HAS_PYVCF = True
    except ImportError:
        HAS_PYVCF = False
        print("PyVCF not available. Using simple VCF parsing.")

class VariantConservationAnalyzer:
    """
    Analyzes conservation scores for individual variants from VCF files
    """
    
    def __init__(self, conservation_data: Optional[pd.DataFrame] = None):
        """
        Initialize with conservation score database
        
        Parameters:
        -----------
        conservation_data : pd.DataFrame
            DataFrame containing genomic positions and conservation scores
            Expected columns: position, phyloP_score, phastCons_score, GERP++, taxon_group
        """
        self.conservation_data = conservation_data
        self.logger = logging.getLogger(__name__)
        
    def load_conservation_database(self, file_path: str) -> pd.DataFrame:
        """
        Load conservation score database from file
        
        Parameters:
        -----------
        file_path : str
            Path to conservation score database file
            
        Returns:
        --------
        pd.DataFrame
            Loaded conservation data
        """
        try:
            if file_path.endswith('.csv'):
                self.conservation_data = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                self.conservation_data = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError("Unsupported file format. Use CSV or TSV.")
                
            self.logger.info(f"Loaded conservation database with {len(self.conservation_data)} entries")
            return self.conservation_data
            
        except Exception as e:
            self.logger.error(f"Error loading conservation database: {e}")
            raise
    
    def parse_vcf_file(self, vcf_path: str) -> List[Dict]:
        """
        Parse VCF file and extract variant information
        
        Parameters:
        -----------
        vcf_path : str
            Path to VCF file
            
        Returns:
        --------
        List[Dict]
            List of variant dictionaries with position, ref, alt, and other info
        """
        variants = []
        
        try:
            if HAS_PYVCF:
                # Use PyVCF if available
                with open(vcf_path, 'r') as vcf_file:
                    vcf_reader = vcf.Reader(vcf_file)
                    
                    for record in vcf_reader:
                        variant_info = {
                            'chromosome': record.CHROM,
                            'position': record.POS,
                            'ref': record.REF,
                            'alt': [str(alt) for alt in record.ALT],
                            'quality': record.QUAL,
                            'filter': record.FILTER,
                            'info': record.INFO,
                            'variant_id': f"{record.CHROM}:{record.POS}:{record.REF}:{','.join([str(alt) for alt in record.ALT])}"
                        }
                        variants.append(variant_info)
            else:
                # Use simple parser
                variants = self._parse_vcf_simple(vcf_path)
                    
            self.logger.info(f"Parsed {len(variants)} variants from VCF file")
            return variants
            
        except Exception as e:
            self.logger.error(f"Error parsing VCF file: {e}")
            # Fallback to simple text parsing
            return self._parse_vcf_simple(vcf_path)
    
    def _parse_vcf_simple(self, vcf_path: str) -> List[Dict]:
        """
        Simple VCF parser as fallback
        """
        variants = []
        
        with open(vcf_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    variant_info = {
                        'chromosome': parts[0],
                        'position': int(parts[1]),
                        'ref': parts[3],
                        'alt': parts[4].split(','),
                        'quality': parts[5] if parts[5] != '.' else None,
                        'variant_id': f"{parts[0]}:{parts[1]}:{parts[3]}:{parts[4]}"
                    }
                    variants.append(variant_info)
        
        return variants
    
    def get_conservation_for_variant(self, variant: Dict) -> Dict:
        """
        Get conservation scores for a specific variant
        
        Parameters:
        -----------
        variant : Dict
            Variant information including position
            
        Returns:
        --------
        Dict
            Conservation analysis results for the variant
        """
        if self.conservation_data is None:
            raise ValueError("Conservation database not loaded")
        
        position = variant['position']
        
        # Find conservation scores for this position
        position_data = self.conservation_data[
            self.conservation_data['position'] == position
        ].copy()
        
        if position_data.empty:
            return {
                'variant_id': variant['variant_id'],
                'position': position,
                'conservation_available': False,
                'message': 'No conservation data available for this position'
            }
        
        # Analyze conservation across taxonomic groups
        conservation_analysis = self._analyze_position_conservation(position_data)
        conservation_analysis.update({
            'variant_id': variant['variant_id'],
            'chromosome': variant.get('chromosome'),
            'position': position,
            'ref': variant['ref'],
            'alt': variant['alt'],
            'conservation_available': True
        })
        
        return conservation_analysis
    
    def _analyze_position_conservation(self, position_data: pd.DataFrame) -> Dict:
        """
        Analyze conservation patterns for a genomic position
        """
        analysis = {
            'taxonomic_groups': {},
            'overall_statistics': {},
            'conservation_interpretation': {}
        }
        
        # Group by taxonomic groups
        for taxon in position_data['taxon_group'].unique():
            taxon_data = position_data[position_data['taxon_group'] == taxon]
            
            analysis['taxonomic_groups'][taxon] = {
                'count': len(taxon_data),
                'phyloP_mean': taxon_data['phyloP_score'].mean() if 'phyloP_score' in taxon_data.columns else None,
                'phyloP_std': taxon_data['phyloP_score'].std() if 'phyloP_score' in taxon_data.columns else None,
                'phastCons_mean': taxon_data['phastCons_score'].mean() if 'phastCons_score' in taxon_data.columns else None,
                'phastCons_std': taxon_data['phastCons_score'].std() if 'phastCons_score' in taxon_data.columns else None,
                'GERP_mean': taxon_data['GERP++'].mean() if 'GERP++' in taxon_data.columns else None,
                'GERP_std': taxon_data['GERP++'].std() if 'GERP++' in taxon_data.columns else None
            }
        
        # Overall statistics
        for score_col in ['phyloP_score', 'phastCons_score', 'GERP++']:
            if score_col in position_data.columns:
                analysis['overall_statistics'][score_col] = {
                    'mean': position_data[score_col].mean(),
                    'median': position_data[score_col].median(),
                    'std': position_data[score_col].std(),
                    'min': position_data[score_col].min(),
                    'max': position_data[score_col].max(),
                    'percentile_25': position_data[score_col].quantile(0.25),
                    'percentile_75': position_data[score_col].quantile(0.75)
                }
        
        # Conservation interpretation
        analysis['conservation_interpretation'] = self._interpret_conservation_scores(
            analysis['overall_statistics']
        )
        
        return analysis
    
    def _interpret_conservation_scores(self, stats: Dict) -> Dict:
        """
        Provide interpretation of conservation scores
        """
        interpretation = {
            'conservation_level': 'unknown',
            'evidence_strength': 'insufficient',
            'acmg_relevance': 'unclear',
            'interpretation_notes': []
        }
        
        # PhyloP interpretation (typical range: -20 to +10)
        if 'phyloP_score' in stats and stats['phyloP_score']['mean'] is not None:
            phylop_mean = stats['phyloP_score']['mean']
            
            if phylop_mean > 2.0:
                interpretation['conservation_level'] = 'high'
                interpretation['evidence_strength'] = 'strong'
                interpretation['acmg_relevance'] = 'supports_PP3'
                interpretation['interpretation_notes'].append(
                    f"PhyloP score ({phylop_mean:.2f}) indicates strong conservation"
                )
            elif phylop_mean > 0.5:
                interpretation['conservation_level'] = 'moderate'
                interpretation['evidence_strength'] = 'moderate'
                interpretation['acmg_relevance'] = 'supports_PP3_weak'
                interpretation['interpretation_notes'].append(
                    f"PhyloP score ({phylop_mean:.2f}) indicates moderate conservation"
                )
            elif phylop_mean < -1.0:
                interpretation['conservation_level'] = 'low'
                interpretation['evidence_strength'] = 'moderate'
                interpretation['acmg_relevance'] = 'supports_BP4'
                interpretation['interpretation_notes'].append(
                    f"PhyloP score ({phylop_mean:.2f}) indicates low conservation"
                )
        
        # phastCons interpretation (range: 0 to 1)
        if 'phastCons_score' in stats and stats['phastCons_score']['mean'] is not None:
            phastcons_mean = stats['phastCons_score']['mean']
            
            if phastcons_mean > 0.8:
                interpretation['interpretation_notes'].append(
                    f"phastCons score ({phastcons_mean:.3f}) indicates high conservation"
                )
            elif phastcons_mean < 0.2:
                interpretation['interpretation_notes'].append(
                    f"phastCons score ({phastcons_mean:.3f}) indicates low conservation"
                )
        
        # GERP interpretation (typical range: -12 to +6)
        if 'GERP++' in stats and stats['GERP++']['mean'] is not None:
            gerp_mean = stats['GERP++']['mean']
            
            if gerp_mean > 2.0:
                interpretation['interpretation_notes'].append(
                    f"GERP++ score ({gerp_mean:.2f}) indicates constraint/conservation"
                )
            elif gerp_mean < -2.0:
                interpretation['interpretation_notes'].append(
                    f"GERP++ score ({gerp_mean:.2f}) indicates accelerated evolution"
                )
        
        return interpretation
    
    def analyze_variant_batch(self, variants: List[Dict]) -> pd.DataFrame:
        """
        Analyze conservation for multiple variants
        
        Parameters:
        -----------
        variants : List[Dict]
            List of variant dictionaries
            
        Returns:
        --------
        pd.DataFrame
            Conservation analysis results for all variants
        """
        results = []
        
        for variant in variants:
            try:
                conservation_result = self.get_conservation_for_variant(variant)
                results.append(conservation_result)
            except Exception as e:
                self.logger.error(f"Error analyzing variant {variant.get('variant_id', 'unknown')}: {e}")
                results.append({
                    'variant_id': variant.get('variant_id', 'unknown'),
                    'position': variant.get('position'),
                    'conservation_available': False,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def export_conservation_report(self, results: pd.DataFrame, output_path: str):
        """
        Export conservation analysis results to file
        """
        try:
            if output_path.endswith('.csv'):
                results.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                results.to_excel(output_path, index=False)
            else:
                # Default to CSV
                results.to_csv(output_path + '.csv', index=False)
                
            self.logger.info(f"Conservation analysis results exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            raise

# Usage example and testing functions
def analyze_vcf_conservation(vcf_path: str, conservation_db_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Complete workflow for VCF conservation analysis
    """
    # Initialize analyzer
    analyzer = VariantConservationAnalyzer()
    
    # Load conservation database
    analyzer.load_conservation_database(conservation_db_path)
    
    # Parse VCF file
    variants = analyzer.parse_vcf_file(vcf_path)
    
    # Analyze conservation for all variants
    results = analyzer.analyze_variant_batch(variants)
    
    # Export results if output path provided
    if output_path:
        analyzer.export_conservation_report(results, output_path)
    
    return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    print("Variant Conservation Analyzer - Ready for VCF analysis")
    print("Usage: analyze_vcf_conservation('variants.vcf', 'conservation_data.csv', 'results.csv')")
