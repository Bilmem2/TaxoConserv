#!/usr/bin/env python3
"""
Test web interface conservation analysis workflow
"""

from src.input_parser import create_sample_vcf_data, create_demo_data
from src.variant_analysis import VariantConservationAnalyzer
import pandas as pd
import numpy as np

def test_web_interface_workflow():
    """Test the exact workflow used in web interface"""
    
    print("=== Testing Web Interface Conservation Workflow ===")
    
    try:
        # Simulate the exact steps from web_taxoconserv.py
        print("1. Simulating sample data loading (as in web interface)...")
        
        # Load sample VCF data (exactly as in web interface)
        variant_data = create_sample_vcf_data()
        
        # Also load compatible conservation database
        conservation_database = create_demo_data()
        
        # Ensure position compatibility (exactly as in web interface)
        if variant_data is not None and conservation_database is not None:
            # Align conservation data positions with VCF positions for demonstration
            vcf_positions = variant_data['POS'].tolist()[:20]  # Take first 20 VCF positions
            
            # Create aligned conservation data
            aligned_conservation = conservation_database.copy()
            aligned_conservation['position'] = np.tile(vcf_positions, 
                                                     (len(aligned_conservation) // len(vcf_positions)) + 1)[:len(aligned_conservation)]
            
            conservation_database = aligned_conservation
        
        print(f"   Loaded {len(variant_data)} sample variants with {len(conservation_database)} conservation data points")
        
        # Test filtering (as would happen in web interface)
        print("\n2. Testing data filtering...")
        filtered_data = variant_data.copy()  # No filters applied for test
        
        print(f"   Filtered data shape: {filtered_data.shape}")
        
        # Test conservation analysis button functionality
        print("\n3. Simulating 'Analyze Conservation Patterns' button click...")
        
        # Initialize variant analyzer for comprehensive analysis
        analyzer = VariantConservationAnalyzer(conservation_database)
        
        # Convert variant data to analyzer format (exactly as in web interface)
        variants_for_analysis = []
        for _, row in filtered_data.iterrows():
            variant = {
                'chromosome': row['CHROM'],
                'position': row['POS'],
                'ref': row['REF'],
                'alt': [row['ALT']] if isinstance(row['ALT'], str) else row['ALT'],
                'variant_id': f"{row['CHROM']}:{row['POS']}:{row['REF']}:{row['ALT']}"
            }
            variants_for_analysis.append(variant)
        
        print(f"   Converted {len(variants_for_analysis)} variants for analysis")
        
        # Perform batch conservation analysis (as in web interface)
        print("\n4. Performing conservation analysis...")
        conservation_results = []
        for i, variant in enumerate(variants_for_analysis[:10]):  # Test first 10 for speed
            result = analyzer.get_conservation_for_variant(variant)
            conservation_results.append(result)
            if i % 5 == 0:
                print(f"   Processed {i+1} variants...")
        
        print(f"   Analysis completed for {len(conservation_results)} variants")
        
        # Test ACMG interpretation summary (as in web interface)
        print("\n5. Testing ACMG interpretation summary...")
        
        acmg_summary = {'PP3': 0, 'PP3_weak': 0, 'BP4': 0, 'insufficient': 0}
        consensus_scores = []
        high_confidence_variants = 0
        
        for result in conservation_results:
            if result.get('conservation_available'):
                # Check for enhanced ACMG interpretation
                enhanced_acmg = result.get('enhanced_acmg', {})
                if enhanced_acmg:
                    criterion = enhanced_acmg.get('primary_criterion', 'insufficient_evidence')
                    if criterion == 'PP3':
                        acmg_summary['PP3'] += 1
                    elif criterion in ['PP3_moderate', 'PP3_weak']:
                        acmg_summary['PP3_weak'] += 1
                    elif criterion == 'BP4':
                        acmg_summary['BP4'] += 1
                    else:
                        acmg_summary['insufficient'] += 1
                else:
                    # Fallback to basic ACMG interpretation
                    acmg_criteria = result.get('acmg_interpretation', {}).get('acmg_criteria', [])
                    if 'PP3' in acmg_criteria:
                        acmg_summary['PP3'] += 1
                    elif 'PP3_weak' in acmg_criteria:
                        acmg_summary['PP3_weak'] += 1
                    elif 'BP4' in acmg_criteria:
                        acmg_summary['BP4'] += 1
                    else:
                        acmg_summary['insufficient'] += 1
                
                # Collect consensus scores
                consensus_data = result.get('consensus_conservation', {})
                if consensus_data.get('consensus_score') is not None:
                    consensus_scores.append(consensus_data['consensus_score'])
                    if consensus_data.get('confidence_level') == 'high':
                        high_confidence_variants += 1
            else:
                acmg_summary['insufficient'] += 1
        
        # Display results (as in web interface)
        print(f"\n6. ACMG Summary Results:")
        print(f"   🔴 PP3 (Strong Conservation): {acmg_summary['PP3']}")
        print(f"   🟡 PP3 Weak (Moderate): {acmg_summary['PP3_weak']}")
        print(f"   🟢 BP4 (Low Conservation): {acmg_summary['BP4']}")
        print(f"   ⚪ Insufficient Evidence: {acmg_summary['insufficient']}")
        
        # Additional consensus score metrics (as in web interface)
        if consensus_scores:
            print(f"\n7. Consensus Conservation Analysis:")
            avg_consensus = np.mean(consensus_scores)
            print(f"   Average Consensus Score: {avg_consensus:.3f}")
            print(f"   High Confidence Variants: {high_confidence_variants}")
            coverage = len(consensus_scores) / len(conservation_results) * 100
            print(f"   Analysis Coverage: {coverage:.1f}%")
        
        # Test conservation score distributions
        print(f"\n8. Testing conservation score availability...")
        conservation_cols = ['phyloP_score', 'GERP_score', 'phastCons_score']
        available_conservation_cols = [col for col in conservation_cols if col in filtered_data.columns]
        print(f"   Available conservation columns: {available_conservation_cols}")
        
        if available_conservation_cols:
            for score_col in available_conservation_cols:
                score_data = filtered_data[score_col].dropna()
                print(f"   {score_col}: mean={score_data.mean():.3f}, std={score_data.std():.3f}, n={len(score_data)}")
        
        print("\n✅ Web interface conservation workflow test completed successfully!")
        print(f"\nSummary:")
        print(f"- Total variants loaded: {len(variant_data)}")
        print(f"- Conservation database size: {len(conservation_database)}")
        print(f"- Variants analyzed: {len(conservation_results)}")
        print(f"- Success rate: {sum(1 for r in conservation_results if r['conservation_available'])/len(conservation_results)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in web interface workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_web_interface_workflow()
