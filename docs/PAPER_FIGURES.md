# Paper Figures and Code Mapping

This document maps each figure in the Neuro-CXG manuscript to the code that generates it.

## Publication Requirements Met

- **300 DPI** - Default in code
- **Colorblind-safe** - Configured in `configs/matplotlib.rc`
- **Grayscale-compatible** - Hardcoded color hex codes work in B&W

## Load Publication Style

```python
import matplotlib.pyplot as plt
plt.style.use('configs/matplotlib.rc')
```

## Main Paper Figures

| Figure | Description | Generation Code |
|--------|-------------|------------------|
| Fig. 1 | Pipeline architecture diagram | `docs/architecture.md` (Mermaid diagram) - render via https://mermaid.live |
| Fig. 2 | Per-site AUC bar chart | `src/run_result_analysis.py` |
| Fig. 3 | Causal graph visualization | `src/analysis/visualize_causal_graph.py` |
| Fig. 4 | Permutation test null distribution | `src/run_evaluation.py` |
| Fig. 5 | Subgroup analysis bars | `src/run_evaluation.py` |
| Fig. 6 | Baseline comparison | `src/run_evaluation.py` |

## Generate All Figures

```bash
# Run evaluation (generates Figs 4-6)
python src/run_evaluation.py

# Per-site AUC (Fig 2)
python src/run_result_analysis.py

# Causal graph (Fig 3) - requires a site ID
python -m src.analysis.visualize_causal_graph --auto-pair --site-id CMU
```

## Output Locations

- `results/evaluation/` - permutation_test.png, subgroup_analysis.png, baseline_comparison.png
- `results/analysis/` - site_effects.png, per_site_auc.png
- `results/visualizations/` - causal_graph_{site_id}.png