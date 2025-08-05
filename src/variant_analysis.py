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

# VCF parsing strategy - prioritize robust built-in parser
HAS_PYVCF = False
vcf = None

# Note: External VCF libraries can be problematic on some systems
# Our built-in parser is robust and handles most VCF formats correctly
try:
    # Try PyVCF3 if specifically installed
    import vcf as pyvcf_module
    HAS_PYVCF = True
    vcf = pyvcf_module
    print("PyVCF library detected - using enhanced VCF parsing")
except ImportError:
    # Use robust built-in parser (recommended)
    HAS_PYVCF = False
    print("Using robust built-in VCF parser (recommended for compatibility)")

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
            if HAS_PYVCF and vcf is not None:
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
                # Use robust built-in parser (recommended)
                variants = self._parse_vcf_simple(vcf_path)
                    
            self.logger.info(f"Parsed {len(variants)} variants from VCF file")
            return variants
            
        except Exception as e:
            self.logger.error(f"Error parsing VCF file: {e}")
            # Fallback to simple text parsing
            return self._parse_vcf_simple(vcf_path)
    
    def _parse_vcf_simple(self, vcf_path: str) -> List[Dict]:
        """
        Robust VCF parser as fallback with comprehensive error handling
        """
        variants = []
        line_number = 0
        
        try:
            with open(vcf_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line_number += 1
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        parts = line.split('\t')
                        
                        # Validate minimum VCF columns
                        if len(parts) < 8:
                            self.logger.warning(f"Line {line_number}: Insufficient columns ({len(parts)} < 8)")
                            continue
                        
                        # Extract and validate core fields
                        chrom = parts[0].strip()
                        pos_str = parts[1].strip()
                        id_field = parts[2].strip() if parts[2] != '.' else None
                        ref = parts[3].strip()
                        alt = parts[4].strip()
                        qual_str = parts[5].strip()
                        filter_field = parts[6].strip()
                        info_field = parts[7].strip()
                        
                        # Validate and convert position
                        try:
                            position = int(pos_str)
                            if position <= 0:
                                self.logger.warning(f"Line {line_number}: Invalid position {position}")
                                continue
                        except ValueError:
                            self.logger.warning(f"Line {line_number}: Cannot convert position '{pos_str}' to integer")
                            continue
                        
                        # Validate nucleotides
                        valid_bases = set('ATCG')
                        if not all(base in valid_bases for base in ref.upper()):
                            self.logger.warning(f"Line {line_number}: Invalid REF allele '{ref}'")
                            continue
                        
                        # Process ALT alleles
                        alt_alleles = []
                        for alt_allele in alt.split(','):
                            alt_allele = alt_allele.strip()
                            if alt_allele == '.':
                                continue
                            if all(base in valid_bases for base in alt_allele.upper()):
                                alt_alleles.append(alt_allele)
                            else:
                                self.logger.warning(f"Line {line_number}: Invalid ALT allele '{alt_allele}'")
                        
                        if not alt_alleles:
                            continue
                        
                        # Process quality score
                        quality = None
                        if qual_str != '.':
                            try:
                                quality = float(qual_str)
                            except ValueError:
                                self.logger.warning(f"Line {line_number}: Invalid quality score '{qual_str}'")
                        
                        # Parse INFO field for additional annotations
                        info_dict = {}
                        if info_field != '.':
                            for info_item in info_field.split(';'):
                                if '=' in info_item:
                                    key, value = info_item.split('=', 1)
                                    info_dict[key] = value
                                else:
                                    info_dict[info_item] = True
                        
                        # Create variant object
                        variant_info = {
                            'chromosome': chrom,
                            'position': position,
                            'id': id_field,
                            'ref': ref,
                            'alt': alt_alleles,
                            'quality': quality,
                            'filter': filter_field if filter_field != '.' else None,
                            'info': info_dict,
                            'variant_id': f"{chrom}:{position}:{ref}:{','.join(alt_alleles)}",
                            'line_number': line_number
                        }
                        variants.append(variant_info)
                        
                    except Exception as e:
                        self.logger.error(f"Line {line_number}: Error parsing variant - {e}")
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading VCF file: {e}")
            raise
        
        self.logger.info(f"Successfully parsed {len(variants)} variants from {line_number} lines")
        return variants
    
    def get_conservation_for_variant(self, variant: Dict) -> Dict:
        """
        Get conservation scores for a specific variant with enhanced position matching
        
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
        chromosome = variant.get('chromosome', '')
        
        # Enhanced position matching strategy
        conservation_result = {
            'variant_id': variant['variant_id'],
            'chromosome': chromosome,
            'position': position,
            'ref': variant['ref'],
            'alt': variant['alt'],
            'conservation_available': False,
            'conservation_scores': {},
            'taxonomic_analysis': {},
            'acmg_interpretation': {},
            'search_strategy': 'direct_match'
        }
        
        # Strategy 1: Direct position match
        position_data = self.conservation_data[
            self.conservation_data['position'] == position
        ].copy()
        
        # Strategy 2: If no direct match, try chromosome-specific matching
        if position_data.empty and 'chromosome' in self.conservation_data.columns:
            # Try matching with chromosome prefix
            chrom_variants = [chromosome, chromosome.replace('chr', ''), f'chr{chromosome.replace("chr", "")}']
            for chrom_variant in chrom_variants:
                position_data = self.conservation_data[
                    (self.conservation_data['chromosome'] == chrom_variant) & 
                    (self.conservation_data['position'] == position)
                ].copy()
                if not position_data.empty:
                    conservation_result['search_strategy'] = f'chromosome_match_{chrom_variant}'
                    break
        
        # Strategy 3: If still no match, try nearby positions (±5 bp window)
        if position_data.empty:
            nearby_window = 5
            position_data = self.conservation_data[
                (self.conservation_data['position'] >= position - nearby_window) &
                (self.conservation_data['position'] <= position + nearby_window)
            ].copy()
            
            if not position_data.empty:
                conservation_result['search_strategy'] = f'nearby_window_±{nearby_window}bp'
                # Calculate distance-weighted scores
                position_data['distance'] = abs(position_data['position'] - position)
                position_data['weight'] = 1 / (position_data['distance'] + 1)
        
        # If no conservation data found at all
        if position_data.empty:
            conservation_result.update({
                'message': f'No conservation data found for position {position} (searched direct, chromosome-specific, and ±5bp window)',
                'suggestions': [
                    'Check if conservation database covers this genomic region',
                    'Verify chromosome naming convention (chr1 vs 1)',
                    'Consider using a more comprehensive conservation database'
                ]
            })
            return conservation_result
        
        # Analyze available conservation data
        conservation_result['conservation_available'] = True
        conservation_result['data_points'] = len(position_data)
        
        # Extract conservation scores
        score_columns = ['phyloP_score', 'phastCons_score', 'GERP++', 'GERP_score', 'conservation_score']
        available_scores = {}
        
        for score_col in score_columns:
            if score_col in position_data.columns:
                scores = position_data[score_col].dropna()
                if len(scores) > 0:
                    if 'weight' in position_data.columns:
                        # Distance-weighted average for nearby positions
                        weights = position_data.loc[scores.index, 'weight']
                        weighted_mean = (scores * weights).sum() / weights.sum()
                        available_scores[score_col] = {
                            'mean': weighted_mean,
                            'raw_values': scores.tolist(),
                            'weights': weights.tolist(),
                            'calculation': 'distance_weighted'
                        }
                    else:
                        # Direct statistics
                        available_scores[score_col] = {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            'min': scores.min(),
                            'max': scores.max(),
                            'median': scores.median(),
                            'count': len(scores),
                            'raw_values': scores.tolist(),
                            'calculation': 'direct_statistics'
                        }
        
        conservation_result['conservation_scores'] = available_scores
        
        # Taxonomic analysis if taxon_group data available
        if 'taxon_group' in position_data.columns:
            taxonomic_analysis = self._analyze_position_conservation(position_data)
            conservation_result['taxonomic_analysis'] = taxonomic_analysis
        
        # ACMG interpretation
        conservation_result['acmg_interpretation'] = self._generate_acmg_interpretation(available_scores)
        
        return conservation_result
    
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
    
    def _generate_acmg_interpretation(self, conservation_scores: Dict) -> Dict:
        """
        Generate ACMG PP3/BP4 interpretation based on conservation scores
        
        Parameters:
        -----------
        conservation_scores : Dict
            Dictionary of conservation scores with statistics
            
        Returns:
        --------
        Dict
            ACMG interpretation with evidence strength
        """
        interpretation = {
            'acmg_criteria': [],
            'evidence_strength': 'insufficient',
            'conservation_level': 'unknown',
            'supporting_evidence': [],
            'recommendations': [],
            'score_thresholds': {
                'phyloP': {'high': 2.0, 'moderate': 0.5, 'low': -1.0},
                'phastCons': {'high': 0.8, 'moderate': 0.5, 'low': 0.2},
                'GERP': {'high': 4.0, 'moderate': 2.0, 'low': -2.0}
            }
        }
        
        # Count evidence for high conservation
        high_conservation_evidence = 0
        moderate_conservation_evidence = 0
        low_conservation_evidence = 0
        
        # Analyze phyloP scores
        phylop_scores = [v for k, v in conservation_scores.items() if 'phylop' in k.lower()]
        if phylop_scores:
            phylop_mean = phylop_scores[0]['mean']
            if phylop_mean > 2.0:
                high_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"PhyloP score ({phylop_mean:.3f}) indicates strong evolutionary conservation"
                )
            elif phylop_mean > 0.5:
                moderate_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"PhyloP score ({phylop_mean:.3f}) indicates moderate evolutionary conservation"
                )
            elif phylop_mean < -1.0:
                low_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"PhyloP score ({phylop_mean:.3f}) indicates accelerated evolution/low conservation"
                )
        
        # Analyze phastCons scores
        phastcons_scores = [v for k, v in conservation_scores.items() if 'phastcons' in k.lower()]
        if phastcons_scores:
            phastcons_mean = phastcons_scores[0]['mean']
            if phastcons_mean > 0.8:
                high_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"phastCons score ({phastcons_mean:.3f}) indicates high conservation probability"
                )
            elif phastcons_mean > 0.5:
                moderate_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"phastCons score ({phastcons_mean:.3f}) indicates moderate conservation probability"
                )
            elif phastcons_mean < 0.2:
                low_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"phastCons score ({phastcons_mean:.3f}) indicates low conservation probability"
                )
        
        # Analyze GERP scores
        gerp_scores = [v for k, v in conservation_scores.items() if 'gerp' in k.lower()]
        if gerp_scores:
            gerp_mean = gerp_scores[0]['mean']
            if gerp_mean > 4.0:
                high_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"GERP++ score ({gerp_mean:.3f}) indicates strong evolutionary constraint"
                )
            elif gerp_mean > 2.0:
                moderate_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"GERP++ score ({gerp_mean:.3f}) indicates moderate evolutionary constraint"
                )
            elif gerp_mean < -2.0:
                low_conservation_evidence += 1
                interpretation['supporting_evidence'].append(
                    f"GERP++ score ({gerp_mean:.3f}) indicates accelerated evolution"
                )
        
        # Determine overall ACMG interpretation
        total_scores = len([s for s in conservation_scores.keys()])
        
        if high_conservation_evidence >= 2 or (high_conservation_evidence >= 1 and total_scores >= 2):
            interpretation['acmg_criteria'].append('PP3')
            interpretation['evidence_strength'] = 'supporting'
            interpretation['conservation_level'] = 'high'
            interpretation['recommendations'].append(
                "Strong conservation evidence supports PP3 (multiple computational evidence supporting deleterious effect)"
            )
        elif moderate_conservation_evidence >= 2:
            interpretation['acmg_criteria'].append('PP3_weak')
            interpretation['evidence_strength'] = 'weak_supporting'
            interpretation['conservation_level'] = 'moderate'
            interpretation['recommendations'].append(
                "Moderate conservation evidence provides weak support for PP3"
            )
        elif low_conservation_evidence >= 2:
            interpretation['acmg_criteria'].append('BP4')
            interpretation['evidence_strength'] = 'supporting'
            interpretation['conservation_level'] = 'low'
            interpretation['recommendations'].append(
                "Low conservation evidence supports BP4 (multiple computational evidence suggesting no impact)"
            )
        else:
            interpretation['evidence_strength'] = 'insufficient'
            interpretation['conservation_level'] = 'unclear'
            interpretation['recommendations'].append(
                "Insufficient conservation evidence for reliable ACMG classification"
            )
        
        # Add general recommendations
        if total_scores < 2:
            interpretation['recommendations'].append(
                "Consider obtaining additional conservation scores (phyloP, phastCons, GERP) for more reliable assessment"
            )
        
        return interpretation

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
    
    def calculate_consensus_conservation_score(self, conservation_scores: Dict) -> Dict:
        """
        Calculate consensus conservation score from multiple metrics with confidence intervals
        
        Parameters:
        -----------
        conservation_scores : Dict
            Dictionary containing multiple conservation scores
            
        Returns:
        --------
        Dict
            Consensus score with confidence intervals and interpretation
        """
        consensus_result = {
            'consensus_score': None,
            'confidence_level': 'unknown',
            'contributing_scores': [],
            'score_agreement': 'unknown',
            'confidence_interval': {'lower': None, 'upper': None},
            'interpretation': 'insufficient_data',
            'evidence_strength': 'insufficient'
        }
        
        # Normalize scores to common scale (0-1, where 1 = highly conserved)
        normalized_scores = []
        score_weights = {
            'phyloP_score': 0.35,  # High weight for phyloP
            'phastCons_score': 0.30,  # High weight for phastCons
            'GERP_score': 0.25,    # Moderate weight for GERP
            'GERP++': 0.25,        # Same as GERP_score
            'conservation_score': 0.20  # Lower weight for generic score
        }
        
        for score_name, score_data in conservation_scores.items():
            if isinstance(score_data, dict) and 'mean' in score_data:
                raw_score = score_data['mean']
                weight = score_weights.get(score_name, 0.15)
                
                # Normalize different score types to 0-1 scale
                if 'phylop' in score_name.lower():
                    # PhyloP: -20 to +10, normalize around 0
                    normalized = min(1.0, max(0.0, (raw_score + 3) / 8))  # -3 to +5 range
                elif 'phastcons' in score_name.lower():
                    # phastCons: already 0-1
                    normalized = min(1.0, max(0.0, raw_score))
                elif 'gerp' in score_name.lower():
                    # GERP: -12 to +6, normalize around 0
                    normalized = min(1.0, max(0.0, (raw_score + 2) / 8))  # -2 to +6 range
                else:
                    # Generic conservation score: assume 0-1 range
                    normalized = min(1.0, max(0.0, raw_score))
                
                normalized_scores.append({
                    'name': score_name,
                    'raw_score': raw_score,
                    'normalized_score': normalized,
                    'weight': weight,
                    'count': score_data.get('count', 1)
                })
                
                consensus_result['contributing_scores'].append({
                    'score_type': score_name,
                    'raw_value': raw_score,
                    'normalized_value': normalized,
                    'weight': weight
                })
        
        if not normalized_scores:
            return consensus_result
        
        # Calculate weighted consensus score
        total_weight = sum(score['weight'] for score in normalized_scores)
        if total_weight > 0:
            weighted_sum = sum(score['normalized_score'] * score['weight'] for score in normalized_scores)
            consensus_score = weighted_sum / total_weight
            consensus_result['consensus_score'] = round(consensus_score, 4)
            
            # Calculate confidence level based on score agreement
            score_values = [score['normalized_score'] for score in normalized_scores]
            score_std = np.std(score_values) if len(score_values) > 1 else 0
            
            # Determine confidence level
            if len(normalized_scores) >= 3:
                if score_std < 0.15:
                    consensus_result['confidence_level'] = 'high'
                    consensus_result['score_agreement'] = 'strong_agreement'
                elif score_std < 0.25:
                    consensus_result['confidence_level'] = 'moderate'
                    consensus_result['score_agreement'] = 'moderate_agreement'
                else:
                    consensus_result['confidence_level'] = 'low'
                    consensus_result['score_agreement'] = 'poor_agreement'
            elif len(normalized_scores) == 2:
                if score_std < 0.2:
                    consensus_result['confidence_level'] = 'moderate'
                    consensus_result['score_agreement'] = 'good_agreement'
                else:
                    consensus_result['confidence_level'] = 'low'
                    consensus_result['score_agreement'] = 'disagreement'
            else:
                consensus_result['confidence_level'] = 'low'
                consensus_result['score_agreement'] = 'single_score'
            
            # Calculate confidence interval (bootstrap-style)
            if len(score_values) > 1:
                margin = 1.96 * score_std / np.sqrt(len(score_values))  # 95% CI
                consensus_result['confidence_interval'] = {
                    'lower': max(0.0, consensus_score - margin),
                    'upper': min(1.0, consensus_score + margin)
                }
            
            # Interpret consensus score
            if consensus_score > 0.8:
                consensus_result['interpretation'] = 'highly_conserved'
                consensus_result['evidence_strength'] = 'strong'
            elif consensus_score > 0.6:
                consensus_result['interpretation'] = 'moderately_conserved'
                consensus_result['evidence_strength'] = 'moderate'
            elif consensus_score > 0.4:
                consensus_result['interpretation'] = 'weakly_conserved'
                consensus_result['evidence_strength'] = 'weak'
            else:
                consensus_result['interpretation'] = 'poorly_conserved'
                consensus_result['evidence_strength'] = 'weak_against_conservation'
        
        return consensus_result
    
    def analyze_variant_batch(self, variants: List[Dict]) -> pd.DataFrame:
        """
        Analyze conservation for multiple variants with enhanced processing
        
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
        total_variants = len(variants)
        
        for i, variant in enumerate(variants):
            try:
                # Get basic conservation analysis
                conservation_result = self.get_conservation_for_variant(variant)
                
                # Add consensus scoring if conservation data available
                if conservation_result.get('conservation_available') and conservation_result.get('conservation_scores'):
                    consensus_score = self.calculate_consensus_conservation_score(conservation_result['conservation_scores'])
                    conservation_result['consensus_conservation'] = consensus_score
                    
                    # Enhanced ACMG interpretation using consensus score
                    if consensus_score.get('consensus_score') is not None:
                        enhanced_acmg = self._enhanced_acmg_interpretation(
                            conservation_result['conservation_scores'],
                            consensus_score
                        )
                        conservation_result['enhanced_acmg'] = enhanced_acmg
                
                conservation_result['processing_order'] = i
                results.append(conservation_result)
                
            except Exception as e:
                self.logger.error(f"Error analyzing variant {variant.get('variant_id', 'unknown')}: {e}")
                results.append({
                    'variant_id': variant.get('variant_id', 'unknown'),
                    'position': variant.get('position'),
                    'conservation_available': False,
                    'error': str(e),
                    'processing_order': i
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add batch-level statistics
        if len(results_df) > 0:
            results_df.attrs['batch_stats'] = self._calculate_batch_statistics(results_df)
        
        return results_df
    
    def _enhanced_acmg_interpretation(self, individual_scores: Dict, consensus_score: Dict) -> Dict:
        """
        Enhanced ACMG interpretation using both individual and consensus scores
        """
        interpretation = {
            'primary_criterion': None,
            'confidence': consensus_score.get('confidence_level', 'unknown'),
            'evidence_strength': 'insufficient',
            'supporting_details': [],
            'recommendations': []
        }
        
        consensus_value = consensus_score.get('consensus_score', 0)
        confidence_level = consensus_score.get('confidence_level', 'unknown')
        
        # Determine ACMG criterion based on consensus score and confidence
        if consensus_value > 0.8 and confidence_level in ['high', 'moderate']:
            interpretation['primary_criterion'] = 'PP3'
            interpretation['evidence_strength'] = 'supporting'
            interpretation['supporting_details'].append(
                f"High consensus conservation score ({consensus_value:.3f}) with {confidence_level} confidence"
            )
        elif consensus_value > 0.6 and confidence_level == 'high':
            interpretation['primary_criterion'] = 'PP3_moderate'
            interpretation['evidence_strength'] = 'moderate_supporting'
            interpretation['supporting_details'].append(
                f"Moderate consensus conservation score ({consensus_value:.3f}) with high confidence"
            )
        elif consensus_value < 0.3 and confidence_level in ['high', 'moderate']:
            interpretation['primary_criterion'] = 'BP4'
            interpretation['evidence_strength'] = 'supporting'
            interpretation['supporting_details'].append(
                f"Low consensus conservation score ({consensus_value:.3f}) with {confidence_level} confidence"
            )
        else:
            interpretation['primary_criterion'] = 'insufficient_evidence'
            interpretation['evidence_strength'] = 'insufficient'
            interpretation['supporting_details'].append(
                f"Consensus score {consensus_value:.3f} with {confidence_level} confidence - insufficient for ACMG classification"
            )
        
        # Add recommendations based on score quality
        if consensus_score.get('score_agreement') == 'poor_agreement':
            interpretation['recommendations'].append(
                "Consider additional conservation metrics due to poor agreement between existing scores"
            )
        
        if len(individual_scores) < 2:
            interpretation['recommendations'].append(
                "Obtain additional conservation scores (phyloP, phastCons, GERP) for more robust assessment"
            )
        
        return interpretation
    
    def _calculate_batch_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for the entire batch of analyzed variants
        """
        stats = {
            'total_variants': len(results_df),
            'conservation_coverage': 0,
            'acmg_distribution': {},
            'average_consensus_score': None,
            'high_confidence_variants': 0
        }
        
        if len(results_df) > 0:
            # Conservation coverage
            with_conservation = results_df['conservation_available'].sum()
            stats['conservation_coverage'] = with_conservation / len(results_df) * 100
            
            # Extract consensus scores where available
            consensus_scores = []
            high_confidence_count = 0
            
            for _, row in results_df.iterrows():
                if row.get('conservation_available'):
                    # Try to extract consensus score from the row data
                    if isinstance(row.get('consensus_conservation'), dict):
                        consensus_data = row['consensus_conservation']
                        if consensus_data.get('consensus_score') is not None:
                            consensus_scores.append(consensus_data['consensus_score'])
                            if consensus_data.get('confidence_level') == 'high':
                                high_confidence_count += 1
            
            if consensus_scores:
                stats['average_consensus_score'] = np.mean(consensus_scores)
                stats['consensus_score_std'] = np.std(consensus_scores)
                stats['high_confidence_variants'] = high_confidence_count
        
        return stats
    
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
