import subprocess
import sys
import os

# Test script for TaxoConserv CLI visualization
# This script runs the CLI with all plot types automatically

data_file = os.path.join('data', 'example_conservation_scores.csv')
score_column = 'conservation_score'
group_column = 'taxon_group'
output_path = 'test_cli_plot'
color_palette = 'colorblind'
output_format = 'png'

plot_types = [
    "boxplot", "violin", "swarm", "strip", "heatmap", "barplot", "histogram", "kde", "density", "pairplot", "correlation"
]

for pt in plot_types:
    print(f"\n--- Testing plot type: {pt} ---")
    cmd = [
        sys.executable, 'src/visualization.py'
    ]
    # Prepare input sequence for CLI
    inputs = f"{data_file}\n{score_column}\n{group_column}\n{pt}\nn\n{output_path}_{pt}\n{color_palette}\n{output_format}\n"
    try:
        result = subprocess.run(cmd, input=inputs.encode(), capture_output=True, timeout=60)
        print(result.stdout.decode())
        if result.stderr:
            print("STDERR:", result.stderr.decode())
    except Exception as e:
        print(f"❌ Error running CLI for {pt}: {e}")
