# Results — Neuro-CXG

**This file is deprecated.** All canonical metrics are maintained in `docs/paper/results.md`.

## Best Model (May 11, 2026)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Test AUC | **0.8810** | [0.8277, 0.9322] |
| CV AUC | 0.8168 ± 0.0488 | — |
| Test F1 | 0.8375 | [0.7785, 0.8903] |
| Accuracy | 83.12% | [77.27%, 88.33%] |

**Architecture**: 12-lobe GATv2, ridge_granger_hybrid (β=0.70), **48ch/4hd/3L/0.33**
**Provenance**: `src/core/hyperparams.py` — stable across 3 independent runs
**Full metrics**: See [`docs/paper/results.md`](paper/results.md)

## Canonical Baseline (May 2, 2026)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Test AUC | **0.8657** | [0.8017, 0.9185] |
| CV AUC | 0.8102 ± 0.0273 | — |

**Architecture**: 12-lobe GATv2, 32ch/2hd/2L/0.35

---

*Last updated: May 11, 2026*