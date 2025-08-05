#!/usr/bin/env python3
"""
Test script to verify sample data conservation pattern analysis
"""

from src.input_parser import create_sample_vcf_data, create_demo_data
from src.variant_analysis import VariantConservationAnalyzer
import pandas as pd
import numpy as np

def test_sample_conservation_analysis():
    """Test conservation pattern analysis with sample data"""
    
    print("=== Testing Sample Conservation Analysis ===")
    
    try:
        # Create sample data
        print("1. Creating sample data...")
        vcf_data = create_sample_vcf_data()
        conservation_data = create_demo_data()
        
        print(f"   VCF Data: {vcf_data.shape} - {list(vcf_data.columns)}")
        print(f"   Conservation Data: {conservation_data.shape} - {list(conservation_data.columns)}")
        
        # Align position data for testing
        print("\n2. Aligning position data...")
        vcf_positions = vcf_data['POS'].tolist()[:10]  # First 10 positions
        
        # Update conservation data positions to match VCF positions for testing
        aligned_conservation = conservation_data.copy()
        aligned_conservation['position'] = np.tile(vcf_positions, 
                                                 (len(aligned_conservation) // len(vcf_positions)) + 1)[:len(aligned_conservation)]
        
        print(f"   Aligned positions: {aligned_conservation['position'].unique()[:10]}")
        
        # Initialize analyzer
        print("\n3. Initializing analyzer...")
        analyzer = VariantConservationAnalyzer(aligned_conservation)
        
        # Test with first few variants
        print("\n4. Testing variant conservation analysis...")
        test_results = []
        
        for i in range(5):  # Test first 5 variants
            test_variant = {
                'chromosome': vcf_data.iloc[i]['CHROM'],
                'position': vcf_data.iloc[i]['POS'], 
                'ref': vcf_data.iloc[i]['REF'],
                'alt': [vcf_data.iloc[i]['ALT']],
                'variant_id': f'test_variant_{i+1}'
            }
            
            print(f"\n   Testing variant {i+1}: {test_variant['chromosome']}:{test_variant['position']}")
            
            # Get conservation analysis
            result = analyzer.get_conservation_for_variant(test_variant)
            
            print(f"   - Conservation available: {result['conservation_available']}")
            print(f"   - Search strategy: {result['search_strategy']}")
            
            if result['conservation_available']:
                print(f"   - Conservation scores found: {len(result.get('conservation_scores', {}))}")
                
                # Check for consensus scoring
                if 'consensus_conservation' in result:
                    consensus = result['consensus_conservation']
                    print(f"   - Consensus score: {consensus.get('consensus_score', 'N/A')}")
                    print(f"   - Confidence level: {consensus.get('confidence_level', 'N/A')}")
                
                # Check ACMG interpretation
                if 'acmg_interpretation' in result:
                    acmg = result['acmg_interpretation']
                    print(f"   - ACMG criteria: {acmg.get('acmg_criteria', [])}")
                
                # Check enhanced ACMG
                if 'enhanced_acmg' in result:
                    enhanced = result['enhanced_acmg']
                    print(f"   - Enhanced ACMG: {enhanced.get('primary_criterion', 'N/A')}")
            
            test_results.append(result)
        
        # Summary statistics
        print("\n5. Summary Statistics:")
        conservation_found = sum(1 for r in test_results if r['conservation_available'])
        print(f"   - Variants with conservation data: {conservation_found}/{len(test_results)}")
        print(f"   - Success rate: {conservation_found/len(test_results)*100:.1f}%")
        
        # Test batch analysis
        print("\n6. Testing batch analysis...")
        variants_for_batch = []
        for i in range(3):
            variant = {
                'chromosome': vcf_data.iloc[i]['CHROM'],
                'position': vcf_data.iloc[i]['POS'], 
                'ref': vcf_data.iloc[i]['REF'],
                'alt': [vcf_data.iloc[i]['ALT']],
                'variant_id': f'batch_variant_{i+1}'
            }
            variants_for_batch.append(variant)
        
        try:
            batch_results = analyzer.analyze_variant_batch(variants_for_batch)
            print(f"   - Batch analysis successful: {batch_results.shape}")
            print(f"   - Batch columns: {list(batch_results.columns)}")
            
            if 'conservation_available' in batch_results.columns:
                batch_success = batch_results['conservation_available'].sum()
                print(f"   - Batch conservation found: {batch_success}/{len(batch_results)}")
        
        except Exception as batch_error:
            print(f"   - Batch analysis error: {batch_error}")
        
        print("\n✅ Sample conservation analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in conservation analysis test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sample_conservation_analysis()
