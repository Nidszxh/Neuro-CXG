# Figure Generation

This document tracks how each figure for the paper is generated.

## Publication Style Setup

- **DPI**: 300 (default in code)
- **Colorblind-safe**: Configured in `configs/matplotlib.rc`
- **Grayscale-compatible**: Hardcoded color hex codes work in B&W

```python
import matplotlib.pyplot as plt
plt.style.use('configs/matplotlib.rc')
```

## Figure Table

| Figure | Description | Generation Script | Output Location |
|--------|-------------|-------------------|-----------------|
| Fig. 1 | Pipeline architecture diagram | `docs/architecture.md` (Mermaid diagram) | Render via https://mermaid.live |
| Fig. 2 | Per-site AUC bar chart | `src/run_result_analysis.py` | `results/analysis/site_effects.png` |
| Fig. 3 | Causal graph visualization | `src/analysis/visualize_causal_graph.py` | `results/visualizations/` |
| Fig. 4 | Permutation test null distribution | `src/run_evaluation.py` | `results/evaluation/` |
| Fig. 5 | Subgroup analysis bars | `src/run_evaluation.py` | `results/evaluation/` |
| Fig. 6 | Baseline comparison | `src/run_evaluation.py` | `results/evaluation/` |
| Fig. 7 | Calibration Plot | `src/run_evaluation.py` | `results/paper_figures/` |
| Fig. 8 | Learning Curve | `src/experiments/run_learning_curve.py` | `results/paper_figures/` |

## Generate All Figures

```bash
# Run evaluation (generates Figs 4-6)
python src/run_evaluation.py

# Per-site AUC (Fig 2)
python src/run_result_analysis.py

# Causal graph (Fig 3) - requires a site ID
python -m src.analysis.visualize_causal_graph --auto-pair --site-id CMU
```
