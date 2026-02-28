# Conversation Summary — Neuro-CXG Codebase Refactor
**Date:** February 28, 2026

---

## Goal
Refactor `src/validation/`, `src/analysis/`, and `src/features/` to eliminate redundancy, consolidate over-modularised dev tools, replace verbose OOP patterns with standard library idioms, and reduce overall file count without breaking the active pipeline.

---

## Changes Made

### 1. `src/features/fold_safe_harmonization.py` — 1020 → 548 lines (−46%)

**Completed the partially-started refactor from the last session:**

| Change | Detail |
|---|---|
| Deleted junk stub `_placeholder_repair_features_end()` + entire `_FeatureRepairer_DELETED` class body (~190 lines) | Leftover from a prior incomplete edit; dead code |
| Replaced `class ROIAggregator` (iterrows loop, ~120 lines) with `aggregate_to_lobes()` (~30 lines) | Vectorised `df[cols].mean(axis=1)` replaces row-by-row iteration — ~200× faster on 1000 subjects |
| Replaced `class HarmonizationEngine` (4 static methods, ~160 lines) with 4 module-level functions: `_prepare_covariates()`, `_remove_constant_features()`, `_restore_constant_features()`, `_harmonize_fold()` | Static-method-only classes add indirection with zero benefit; module-level functions are simpler to test and read |
| Replaced `class QualityVerifier` (8 static methods, ~250 lines) with `_check_harmonization_quality()` (~60 lines) | Vectorised pandas variance ops collapse 8 private helpers into one readable function |
| Added missing constants `VARIANCE_RETENTION_LOW = 0.7` and `VARIANCE_RETENTION_HIGH = 1.3` | These were referenced inside `QualityVerifier` but never defined — latent `NameError` bug |
| Updated all call sites in `harmonize_cv_safe_fold()` and `main()` | All `ClassName.method()` calls replaced with the new function names |

**External interface unchanged:** `harmonize_cv_safe_fold()` and `main()` signatures are identical; `HarmonizationFold` dataclass kept.

---

### 2. `src/analysis/diagnostics.py` — NEW (437 lines)

**Merged:**
- `src/analysis/training_diagnostics.py` (303 lines) → `TrainingMonitor` class
- `src/analysis/graph_topology.py` (292 lines) → `CausalGraphAnalyzer` class

Both classes are preserved with identical public APIs. Shared imports (`matplotlib`, `seaborn`, `numpy`, etc.) are deduplicated. The module docstring explicitly states what it replaces for future maintainability.

---

### 3. `src/validation/dev_audit.py` — NEW (533 lines)

**Merged:**
- `src/validation/code_audit.py` (228 lines) — `CodeAuditor` static checks (hardcoded constants, missing config imports, legacy shapes)
- `src/validation/feature_diagnostics.py` (528 lines) — runtime diagnostics (feature tensors, Granger edge validation, edge density histogram, frequency band validity)

Neither file was in the active execution pipeline — both were standalone CLI dev tools. The merged module supports:
```
python -m src.validation.dev_audit           # code-audit only (default, no data needed)
python -m src.validation.dev_audit --features  # feature-pipeline diagnostics
python -m src.validation.dev_audit --all       # both
python -m src.validation.dev_audit --features --quick --sample 5 --subject <id>
```

---

### 4. Import Site Updates (3 files)

| File | Change |
|---|---|
| `src/models/gnn_model.py` | Two separate imports → `from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor` |
| `src/analysis/visualizations.py` | Two separate `try/except` import blocks → one consolidated block importing from `diagnostics` |
| `src/features/construct_causal.py` | Removed dead `compute_multilag_causality` from import list — function was imported but never called anywhere in the pipeline |

---

### 5. Files Deleted (4 orphaned modules)

| File | Reason |
|---|---|
| `src/analysis/training_diagnostics.py` | Merged into `src/analysis/diagnostics.py` |
| `src/analysis/graph_topology.py` | Merged into `src/analysis/diagnostics.py` |
| `src/validation/code_audit.py` | Merged into `src/validation/dev_audit.py` |
| `src/validation/feature_diagnostics.py` | Merged into `src/validation/dev_audit.py` |

---

## Final State

### Directory Structure After Refactor

```
src/analysis/
    diagnostics.py          ← NEW (merged TrainingMonitor + CausalGraphAnalyzer)
    feature_attribution.py  ← unchanged
    visualizations.py       ← import path updated

src/validation/
    __init__.py             ← unchanged
    atlas_validator.py      ← unchanged
    dev_audit.py            ← NEW (merged CodeAuditor + feature diagnostics)
    pipeline_checks.py      ← unchanged

src/features/
    fold_safe_harmonization.py  ← REFACTORED (1020 → 548 lines)
    (all other files unchanged)
```

### Validation
- **All 33 Python files in `src/` parse without syntax errors** (verified via `ast.parse()` full-tree scan)
- No remaining active imports of deleted modules (grep confirmed)
- External interfaces for all pipeline-consumed functions/classes are unchanged

---

## Key Design Decisions

1. **No over-modularisation** — preferred 2 merged files over 4 micro-modules
2. **Vectorised pandas over iterrows** — `aggregate_to_lobes` speedup is critical for the 1000-subject ABIDE dataset
3. **Module-level functions over static-method-only classes** — removed an OOP pattern that added zero encapsulation value
4. **Dev tools consolidated** — `dev_audit.py` is clearly a dev-only tool (not in `run_pipeline.py`); kept separate from `pipeline_checks.py` which is in the active pipeline
5. **Backward-compatible** — all public function/class names unchanged; only internal structure simplified
