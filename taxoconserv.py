#!/usr/bin/env python3
"""
TaxoConserv - Taxonomic Conservation Score Analysis Platform
Main CLI Entry Point

Copyright 2025 Can Sevilmiş

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

DISCLAIMER:
This software is provided "AS IS" and any express or implied warranties,
including, but not limited to, the implied warranties of merchantability
and fitness for a particular purpose are disclaimed. For research and 
educational purposes only. Not intended for clinical or diagnostic use.

Usage:
    python taxoconserv.py --input data/input.csv --output output/
"""

import argparse
import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Version information (Semantic Versioning)
__version__ = "2.1.0"
__version_info__ = (2, 1, 0)
__author__ = "Can Sevilmiş"
__email__ = ""
__license__ = "APACHE 2.0"
__description__ = "Taxonomic Conservation Analysis Tool for Research Applications"

# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Background colors
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

def print_colored(text, color=Colors.ENDC, bold=False, bg_color=None):
    """Print text with color and formatting."""
    if not sys.stdout.isatty():  # Skip colors if output is redirected
        print(text)
        return
    
    format_str = ""
    if bold:
        format_str += Colors.BOLD
    if bg_color:
        format_str += bg_color
    format_str += color
    
    print(f"{format_str}{text}{Colors.ENDC}")

def print_banner():
    """Print a professional and colorful welcome banner."""
    banner = f"""
{Colors.OKCYAN}================================================================================{Colors.ENDC}
{Colors.HEADER}                        🌿 TaxoConserv - v{__version__}                        {Colors.ENDC}
{Colors.OKCYAN}        Taxonomic Conservation Analysis Tool for Research Applications{Colors.ENDC}
{Colors.OKCYAN}================================================================================{Colors.ENDC}

    {Colors.OKGREEN}Advanced Statistical Analysis for Conservation Biology{Colors.ENDC}
    {Colors.OKBLUE}Phylogenetic Conservation Score Analysis & Visualization{Colors.ENDC}
    
    {Colors.WARNING}Created by: {__author__}{Colors.ENDC}
    
{Colors.OKCYAN}================================================================================{Colors.ENDC}
"""
    print(banner)
    print_colored(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.OKBLUE)
    print_colored("-" * 80, Colors.OKCYAN)

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import our modules
try:
    from src.input_parser import parse_input, validate_input
    from src.taxon_grouping import group_by_taxon, calculate_stats, get_group_summary
    from src.stats_tests import run_kruskal_wallis, format_test_results
    from src.visualization import generate_visualization, generate_boxplot, generate_summary_plot
    # from report_generator import export_summary_csv
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("📝 Some modules may not be available yet.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TaxoConserv - Taxonomic Conservation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    Basic usage:
        python taxoconserv.py --input data/conservation_scores.csv --output results/
    
    Advanced usage:
        python taxoconserv.py -i data/phyloP_scores.tsv -o analysis_results/ --format tsv --verbose
        python taxoconserv.py -i data/custom_data.csv -o output/ --score-column "phyloP" --taxon-column "species"
    
SUPPORTED FORMATS:
    • CSV (Comma-separated values)
    • TSV (Tab-separated values)
    
REQUIRED COLUMNS:
    • Conservation scores (numeric values)
    • Taxonomic groups (categorical data)
    
OUTPUT INCLUDES:
    • Statistical analysis results
    • Boxplot visualizations
    • Detailed summary reports
    
NOTE: Run without arguments to access demo mode with sample data.
        """
    )
    
    # Required arguments
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="FILE",
        help="Input CSV/TSV file containing conservation scores and taxonomic data"
    )
    
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="DIR",
        help="Output directory where results will be saved"
    )
    
    # Optional arguments
    parser.add_argument(
        "--format",
        choices=["csv", "tsv"],
        default="csv",
        metavar="FORMAT",
        help="Input file format: 'csv' or 'tsv' (default: csv)"
    )
    
    parser.add_argument(
        "--score-column",
        default="phyloP_score",
        metavar="COLUMN",
        help="Column name containing conservation scores (default: phyloP_score)"
    )
    
    parser.add_argument(
        "--taxon-column",
        default="taxon_group",
        metavar="COLUMN",
        help="Column name containing taxonomic group labels (default: taxon_group)"
    )
    
    parser.add_argument(
        "--plot-type",
        choices=["boxplot", "violin", "swarm", "heatmap"],
        default="boxplot",
        metavar="TYPE",
        help="Type of plot to generate: boxplot, violin, swarm, or heatmap (default: boxplot)"
    )
    
    parser.add_argument(
        "--color-palette",
        choices=["viridis", "plasma", "inferno", "magma", "cividis", "colorblind"],
        default="colorblind",
        metavar="PALETTE",
        help="Color palette for visualizations (default: colorblind)"
    )
    
    parser.add_argument(
        "--output-format",
        choices=["png", "pdf", "svg"],
        default="png",
        metavar="FORMAT",
        help="Output format for visualizations (default: png)"
    )
    
    parser.add_argument(
        "--show-pvalue",
        action="store_true",
        help="Show p-value in plot title"
    )
    
    parser.add_argument(
        "--show-gene",
        action="store_true",
        help="Show gene information in plot title (if available)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Generate interactive HTML plots using Plotly"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed output and progress information"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"🌿 TaxoConserv v{__version__} - {__description__}"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    print_colored("\nValidating input parameters...", Colors.OKBLUE)
    
    # Check input file exists
    if not os.path.exists(args.input):
        print_colored(f"ERROR: Input file '{args.input}' does not exist.", Colors.FAIL, bold=True)
        sys.exit(1)
    else:
        print_colored(f"Input file: {args.input}", Colors.OKGREEN)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    print_colored(f"Output directory: {args.output}", Colors.OKGREEN)
    
    if args.verbose:
        print_colored("\nConfiguration Summary:", Colors.OKCYAN, bold=True)
        print_colored(f"  Input file: {args.input}", Colors.OKBLUE)
        print_colored(f"  Output directory: {args.output}", Colors.OKBLUE)
        print_colored(f"  Format: {args.format}", Colors.OKBLUE)
        print_colored(f"  Score column: {args.score_column}", Colors.OKBLUE)
        print_colored(f"  Taxon column: {args.taxon_column}", Colors.OKBLUE)
        print_colored(f"  Plot type: {args.plot_type}", Colors.OKBLUE)
        print_colored(f"  Color palette: {args.color_palette}", Colors.OKBLUE)
        print_colored(f"  Output format: {args.output_format}", Colors.OKBLUE)
        print_colored(f"  Show p-value: {args.show_pvalue}", Colors.OKBLUE)
        print_colored(f"  Interactive mode: {args.interactive}", Colors.OKBLUE)


def show_progress(step, total_steps, message):
    """Show progress with a clean progress bar."""
    progress = int((step / total_steps) * 40)
    bar = "=" * progress + "-" * (40 - progress)
    percentage = int((step / total_steps) * 100)
    print_colored(f"\n[{bar}] {percentage}% - {message}", Colors.OKCYAN)

def show_interactive_help():
    """Show interactive help menu."""
    help_banner = f"""
{Colors.OKCYAN}================================================================================{Colors.ENDC}
{Colors.HEADER}                            📚 INTERACTIVE HELP 📚                           {Colors.ENDC}
{Colors.OKCYAN}================================================================================{Colors.ENDC}
"""
    print(help_banner)
    
    print_colored("🎯 What TaxoConserv can do for you:", Colors.OKCYAN, bold=True)
    print_colored("-" * 50, Colors.OKCYAN)
    print_colored("• 📊 Analyze conservation scores across taxonomic groups", Colors.OKBLUE)
    print_colored("• 🧮 Perform statistical tests (Kruskal-Wallis)", Colors.OKBLUE)
    print_colored("• 📈 Generate multiple visualization types", Colors.OKBLUE)
    print_colored("• 📄 Export comprehensive analysis reports", Colors.OKBLUE)
    
    print_colored("\n📋 Required data format:", Colors.OKCYAN, bold=True)
    print_colored("-" * 30, Colors.OKCYAN)
    print_colored("Your input file should contain:", Colors.OKBLUE)
    print_colored("  📊 Conservation scores (numeric column)", Colors.OKGREEN)
    print_colored("  🏷️  Taxonomic groups (categorical column)", Colors.OKGREEN)
    print_colored("  📁 Supported formats: CSV, TSV", Colors.OKGREEN)
    
    print_colored("\n💡 Quick start examples:", Colors.OKCYAN, bold=True)
    print_colored("-" * 30, Colors.OKCYAN)
    print_colored("1. Basic analysis:", Colors.OKBLUE)
    print_colored("   python taxoconserv.py -i data.csv -o results/", Colors.OKGREEN)
    print_colored("2. With violin plot:", Colors.OKBLUE)
    print_colored("   python taxoconserv.py -i data.csv -o results/ --plot-type violin", Colors.OKGREEN)
    print_colored("3. Interactive mode:", Colors.OKBLUE)
    print_colored("   python taxoconserv.py -i data.csv -o results/ --interactive", Colors.OKGREEN)
    
    print_colored("\n🆘 Need more help?", Colors.OKCYAN, bold=True)
    print_colored("-" * 20, Colors.OKCYAN)
    print_colored("• Run: python taxoconserv.py --help", Colors.OKBLUE)
    print_colored("• Try demo mode: python taxoconserv.py (no arguments)", Colors.OKBLUE)
    print_colored("• Check documentation in README.md", Colors.OKBLUE)
    
    print_colored("\n" + "-" * 80, Colors.OKCYAN)
    print_colored("Press Enter to continue...", Colors.OKBLUE)
    input()

def show_welcome_menu():
    """Show welcome menu for first-time users."""
    menu_banner = f"""
{Colors.OKCYAN}================================================================================{Colors.ENDC}
{Colors.HEADER}                            WELCOME TO TAXOCONSERV{Colors.ENDC}
{Colors.OKCYAN}================================================================================{Colors.ENDC}
"""
    print(menu_banner)
    
    print_colored("Available Options:", Colors.OKCYAN, bold=True)
    print_colored("-" * 30, Colors.OKCYAN)
    print_colored("1. 🚀 Run demo with sample data", Colors.OKGREEN)
    print_colored("2. 📚 View interactive help", Colors.OKBLUE)
    print_colored("3. 🏃 Continue with current analysis", Colors.OKGREEN)
    print_colored("4. 🚪 Exit program", Colors.FAIL)
    
    while True:
        choice = input(f"\n{Colors.OKCYAN}Enter your choice (1-4): {Colors.ENDC}")
        
        if choice == "1":
            return "demo"
        elif choice == "2":
            show_interactive_help()
            continue
        elif choice == "3":
            return "continue"
        elif choice == "4":
            print_colored("👋 Thank you for using TaxoConserv!", Colors.HEADER, bold=True)
            sys.exit(0)
        else:
            print_colored("❌ Invalid choice. Please enter 1, 2, 3, or 4.", Colors.FAIL)

def main():
    """Main pipeline orchestration."""
    print_banner()
    
    # Parse arguments - use test mode if no arguments provided
    try:
        args = parse_arguments()
    except SystemExit:
        # If no arguments provided, show welcome menu
        choice = show_welcome_menu()
        
        if choice == "demo":
            print_colored("\n🚀 Launching DEMO MODE with sample data...", Colors.OKGREEN, bold=True)
            
            # Show demo mode banner
            demo_banner = f"""
{Colors.OKCYAN}================================================================================{Colors.ENDC}
{Colors.HEADER}                            🎯 DEMO MODE ACTIVATED 🎯                          {Colors.ENDC}
{Colors.OKCYAN}                                                                               {Colors.ENDC}
{Colors.OKBLUE}  This demonstration will analyze sample conservation data                     {Colors.ENDC}
{Colors.OKBLUE}  Using: data/example_conservation_scores.csv                                 {Colors.ENDC}
{Colors.OKCYAN}                                                                               {Colors.ENDC}
{Colors.OKCYAN}================================================================================{Colors.ENDC}
"""
            print(demo_banner)
            
            # Create a test args object
            class TestArgs:
                def __init__(self):
                    self.input = "data/example_conservation_scores.csv"
                    self.output = "output/"
                    self.format = "csv"
                    self.score_column = "conservation_score"
                    self.taxon_column = "taxon_group"
                    self.plot_type = "boxplot"
                    self.color_palette = "colorblind"
                    self.output_format = "png"
                    self.show_pvalue = True
                    self.show_gene = False
                    self.interactive = False
                    self.verbose = True
            
            args = TestArgs()
            
            # Check if test file exists
            if not os.path.exists(args.input):
                print_colored(f"\nERROR: Demo file '{args.input}' not found!", Colors.FAIL, bold=True)
                print_colored("Please create sample data or provide proper arguments.", Colors.WARNING)
                print_colored("\nYou can create sample data by running:", Colors.OKBLUE)
                print_colored("   python -c \"import pandas as pd; import numpy as np; pd.DataFrame({'taxon_group': ['Mammals']*50 + ['Birds']*50 + ['Reptiles']*30, 'conservation_score': np.random.normal(0.5, 0.2, 130)}).to_csv('data/example_conservation_scores.csv', index=False)\"", Colors.OKGREEN)
                print_colored("\nPress Enter to exit...", Colors.OKBLUE)
                input()
                return
        elif choice == "continue":
            # User wants to continue but didn't provide arguments
            print_colored("\nPlease run the program with proper arguments:", Colors.OKBLUE)
            print_colored("   python taxoconserv.py --help", Colors.OKGREEN)
            print_colored("\nPress Enter to exit...", Colors.OKBLUE)
            input()
            return
    
    validate_arguments(args)
    
    try:
        # Pipeline steps with progress tracking
        total_steps = 5
        
        show_progress(1, total_steps, "Parsing and validating input data...")
        time.sleep(0.5)  # Brief pause for visual effect
        
        # Parse and validate input
        data = parse_input(args.input)
        validated_data = validate_input(data)
        print_colored("Data successfully parsed and validated!", Colors.OKGREEN)
        
        show_progress(2, total_steps, "Grouping data by taxonomy and calculating statistics...")
        time.sleep(0.5)
        
        # Group data and calculate statistics
        grouped_data = group_by_taxon(validated_data)
        stats_summary = calculate_stats(grouped_data, args.score_column)
        print_colored("Taxonomic grouping and statistics completed!", Colors.OKGREEN)
        
        show_progress(3, total_steps, "Running statistical tests...")
        time.sleep(0.5)
        
        # Run Kruskal-Wallis test
        test_results = run_kruskal_wallis(grouped_data, args.score_column)
        print_colored("Statistical analysis completed!", Colors.OKGREEN)
        
        show_progress(4, total_steps, "Generating visualizations...")
        time.sleep(0.5)
        
        # Generate visualization
        plot_path = generate_visualization(validated_data,
                                         plot_type=args.plot_type,
                                         group_column=args.taxon_column,
                                         score_column=args.score_column,
                                         output_path=f"{args.output}/conservation_plot",
                                         color_palette=args.color_palette,
                                         output_format=args.output_format,
                                         show_pvalue=args.show_pvalue,
                                         show_gene=args.show_gene,
                                         interactive=args.interactive,
                                         test_results=test_results)
        print_colored("Visualizations generated successfully!", Colors.OKGREEN)
        
        show_progress(5, total_steps, "Compiling final report...")
        time.sleep(0.5)
        
        # Print enhanced summary results
        results_banner = f"""
{Colors.OKCYAN}================================================================================{Colors.ENDC}
{Colors.HEADER}                            📊 ANALYSIS RESULTS 📊                           {Colors.ENDC}
{Colors.OKCYAN}================================================================================{Colors.ENDC}
"""
        print(results_banner)
        
        # Statistical test results
        print_colored("🧮 Statistical Test Results:", Colors.OKCYAN, bold=True)
        print_colored("-" * 40, Colors.OKCYAN)
        print_colored(f"📈 Test Type: {test_results['test_type']}", Colors.OKBLUE)
        print_colored(f"🔢 H-statistic: {test_results['H_statistic']:.4f}", Colors.OKBLUE)
        print_colored(f"📊 p-value: {test_results['p_value']:.6f}", Colors.OKBLUE)
        
        # Significance with color coding
        if test_results['significant']:
            print_colored("✅ Result: SIGNIFICANT difference detected!", Colors.OKGREEN, bold=True)
        else:
            print_colored("❌ Result: No significant difference found", Colors.WARNING, bold=True)
        
        # Group statistics with better formatting
        print_colored("\n📋 Taxonomic Group Statistics:", Colors.OKCYAN, bold=True)
        print_colored("-" * 60, Colors.OKCYAN)
        
        for _, row in stats_summary.iterrows():
            group_info = f"🏷️  {row['taxon_group']:<15} | n={row['sample_size']:<4} | μ={row['mean_score']:.3f} | median={row['median_score']:.3f}"
            print_colored(group_info, Colors.OKBLUE)
        
        # Success message
        success_banner = f"""
{Colors.OKGREEN}================================================================================{Colors.ENDC}
{Colors.HEADER}                            🎉 ANALYSIS COMPLETE! 🎉                         {Colors.ENDC}
{Colors.OKGREEN}================================================================================{Colors.ENDC}
"""
        print(success_banner)
        
        print_colored(f"Results saved to: {args.output}", Colors.OKGREEN)
        print_colored(f"Visualization: {plot_path}", Colors.OKGREEN)
        print_colored(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", Colors.OKBLUE)
        
        # Interactive exit
        print_colored("\n🎯 Analysis Summary:", Colors.HEADER, bold=True)
        print_colored("   • Data successfully processed and analyzed", Colors.OKGREEN)
        print_colored("   • Statistical tests completed", Colors.OKGREEN)
        print_colored("   • Visualizations generated", Colors.OKGREEN)
        print_colored("   • Results exported to output directory", Colors.OKGREEN)
        
        print_colored("\n" + "-" * 80, Colors.OKCYAN)
        print_colored("Thank you for using TaxoConserv! 🌿", Colors.HEADER, bold=True)
        print_colored("Press Enter to exit...", Colors.OKBLUE)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        
    except Exception as e:
        error_banner = f"""
{Colors.FAIL}================================================================================{Colors.ENDC}
{Colors.HEADER}                              ❌ ERROR OCCURRED ❌                           {Colors.ENDC}
{Colors.FAIL}================================================================================{Colors.ENDC}
"""
        print(error_banner)
        print_colored(f"💥 Error during analysis: {e}", Colors.FAIL, bold=True)
        
        if args.verbose:
            print_colored("\n🔍 Detailed error information:", Colors.WARNING)
            import traceback
            traceback.print_exc()
        
        print_colored("\n🆘 Troubleshooting tips:", Colors.OKCYAN)
        print_colored("   • Check your input file format and columns", Colors.OKBLUE)
        print_colored("   • Ensure all required dependencies are installed", Colors.OKBLUE)
        print_colored("   • Try running with --verbose flag for more details", Colors.OKBLUE)
        
        print_colored("\nPress Enter to exit...", Colors.OKBLUE)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
