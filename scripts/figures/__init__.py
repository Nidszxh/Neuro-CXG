"""
Neuro-CXG Publication Figure Generation Pipeline
==================================================

Generates all paper-ready figures from trained model results.

Usage:
    python scripts/figures/generate_all.py          # Generate all figures
    python scripts/figures/generate_all.py --figures ablations training  # Specific figures
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.figures.plot_ablation_comparison import generate_ablation_figure
from scripts.figures.plot_consort_flow import generate_consort_diagram
from scripts.figures.plot_embedding_viz import generate_embedding_visualization
from scripts.figures.plot_training_curves import generate_training_curves

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "paper_figures"

def main():
    parser = argparse.ArgumentParser(description="Generate Neuro-CXG publication figures")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["all"],
        help="Figures to generate: ablations, training, consort, embedding, all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--style",
        type=str,
        default="paper",
        choices=["paper", "presentation", "poster"],
        help="Figure style preset"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output figures"
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figures_to_generate = args.figures if "all" not in args.figures else [
        "ablations", "training", "consort", "embedding"
    ]

    print(f"Generating figures: {figures_to_generate}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Style: {args.style}, DPI: {args.dpi}")
    print("-" * 50)

    success_count = 0
    failed_figures = []

    for fig_name in figures_to_generate:
        try:
            if fig_name == "ablations":
                output_path = generate_ablation_figure(args.output_dir / "ablations", args.dpi)
            elif fig_name == "training":
                output_path = generate_training_curves(args.output_dir / "training_curves", args.dpi)
            elif fig_name == "consort":
                output_path = generate_consort_diagram(args.output_dir, args.dpi)
            elif fig_name == "embedding":
                output_path = generate_embedding_visualization(args.output_dir / "embeddings", args.dpi)
            else:
                print(f"Unknown figure: {fig_name}")
                continue

            if output_path and output_path.exists():
                print(f"  ✓ {fig_name}: {output_path}")
                success_count += 1
            else:
                print(f"  ✗ {fig_name}: Failed to generate")
                failed_figures.append(fig_name)

        except Exception as e:
            print(f"  ✗ {fig_name}: Error - {e}")
            failed_figures.append(fig_name)

    print("-" * 50)
    print(f"Generated {success_count}/{len(figures_to_generate)} figures")

    if failed_figures:
        print(f"Failed figures: {', '.join(failed_figures)}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
