#!/bin/bash
# Generate all publication-ready figures for Neuro-CXG paper
# Usage: bash scripts/figures/generate_all.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Neuro-CXG: Generating Paper Figures"
echo "========================================"

# Ensure output directory exists
mkdir -p results/paper_figures/

# Run main figure generation script
echo "[1/4] Running generate_paper_figures.py..."
python3 src/analysis/generate_paper_figures.py --output results/paper_figures/

# Generate architecture diagram (from Mermaid in docs/architecture.md)
echo "[2/4] Generating architecture diagram..."
if command -v mermaid &> /dev/null; then
    mermaid docs/architecture.md -o results/paper_figures/
elif [ -f "results/paper_figures/architecture_diagram.png" ]; then
    echo "  Architecture diagram already exists"
fi

# Generate causal graph with ASD vs Control split
echo "[3/4] Generating causal graph visualizations..."
python3 -c "
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from src.analysis.diagnostics import CausalGraphAnalyzer
from src.core.config import CAUSAL_GRAPHS_DIR, MASTER_MANIFEST
import pandas as pd

manifest = pd.read_csv(MASTER_MANIFEST)
print('  DX_GROUP values:', manifest['DX_GROUP'].unique())

analyzer = CausalGraphAnalyzer(CAUSAL_GRAPHS_DIR, manifest)

# Generate ASD vs Control average causal graphs
output_dir = Path('results/paper_figures/causal_graphs')
output_dir.mkdir(parents=True, exist_ok=True)

# ASD group
try:
    result = analyzer.visualize_average_causal_graph(output_path=output_dir / 'causal_asd.png', group='ASD')
    if result:
        print('  Generated: causal_asd.png')
    else:
        print('  Skipped: causal_asd.png (no graphs)')
except Exception as e:
    print(f'  Warning: {e}')

# Control group  
try:
    result = analyzer.visualize_average_causal_graph(output_path=output_dir / 'causal_control.png', group='Control')
    if result:
        print('  Generated: causal_control.png')
    else:
        print('  Skipped: causal_control.png (no graphs)')
except Exception as e:
    print(f'  Warning: {e}')
" 2>&1 || echo "  Note: Causal graph generation requires processed graphs"

# Generate circular connectome plots
echo "[4/4] Generating circular connectome visualizations..."
python3 src/analysis/circular_connectome.py --output results/paper_figures/causal_graphs/ || echo "  Note: Circular connectome requires causal graphs"

echo ""
echo "========================================"
echo "Figure generation complete!"
echo "Output: results/paper_figures/"
echo "========================================"

# List generated files
echo ""
echo "Generated files:"
find results/paper_figures/ -type f \( -name "*.png" -o -name "*.pdf" \) | sort
