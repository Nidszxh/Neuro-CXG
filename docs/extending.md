# Extending Neuro-CXG

## Goal

Use this guide when adding new capabilities while preserving existing pipeline contracts and reproducibility safeguards.

## Extension Surfaces

- stage orchestration (`src/pipeline/registry.py`, `src/run_pipeline.py`)
- feature schema (`src/core/feature_registry.py`, `src/features/*`, `src/features/graph_factory.py`)
- model/training behavior (`src/models/*.py`, `src/core/hyperparams.py`)
- reporting outputs (`src/run_*.py`)

## 1) Add a New Pipeline Stage

### Required Changes

1. Add stage metadata in `src/pipeline/registry.py`:
   - unique `key`
   - `module`
   - optional `function`
   - dependency list
   - reliable output sentinel
2. Ensure runner can execute it in context (`src/run_pipeline.py`):
   - reason text (optional but recommended)
   - skip behavior / mode interactions where needed
3. If the stage produces critical artifacts, add or update validation checks.

### Sentinel Rule

Pick an output sentinel that is both:

- deterministic for successful execution
- specific enough to avoid false positives

## 2) Add or Modify Feature Channels

### Required Changes

1. Update feature definitions in `src/core/feature_registry.py`.
2. Keep `ALL_FEATURE_NAMES` order stable and intentional.
3. Update extraction code to write compatible artifacts.
4. Update dataset assembly in `src/features/graph_factory.py`.
5. Verify model input dimensions (`GNN_IN_CHANNELS`) are still consistent.

### Critical Constraint

Do not reintroduce site-leaky spatial channels (`conf_std`, `detection_count`) into model input unless intentionally running a controlled experiment.

## 3) Add a New Causality or Graph Method

### Required Changes

1. Add method config in `src/core/hyperparams.py`.
2. Implement method path in `src/features/construct_causal.py` and/or `src/features/causal_inference.py`.
3. Preserve graph package contract expected by `ABIDECausalDataset`:
   - adjacency available and finite
   - internal features shape compatible
4. Validate downstream sparsity and degeneracy behavior.

### Safety Rule

If a new method can produce many zero-edge graphs, add diagnostics and gate checks before enabling it by default.

## 4) Extend Model Architecture or Training Objectives

### Required Changes

1. Introduce knobs in `src/core/hyperparams.py` with stable defaults.
2. Implement architecture changes in `src/models/causal_gnn.py` and construction path in `src/models/factory.py`.
3. Wire training objective changes in `src/models/gnn_model.py` / `src/models/training_utils.py`.
4. Ensure checkpoints remain loadable by evaluation/explainability scripts.

### Compatibility Rule

Keep backward-compatible checkpoint loading paths where practical (shape inference and `strict=False` loading are already used in evaluation scripts).

## 5) Extend Evaluation or Reporting

### Required Changes

1. Add computation and outputs in the relevant `src/run_*.py` script.
2. Preserve machine-readable summary artifacts:
   - `results/evaluation/comprehensive_results.json`
   - `results/explainability/summary.json`
   - `results/analysis/result_analysis_summary.json`
3. If output names/locations change, update stage sentinels in `src/pipeline/registry.py`.

### Drift Prevention

`run_result_analysis.py` uses evaluation metadata when available. Keep threshold-policy metadata consistent to avoid report divergence.

## 6) Extension Validation Checklist

Before considering an extension complete:

1. Dry-run orchestration plan:

```bash
python src/run_pipeline.py --dry-run
```

2. Rebuild only affected stages.
3. Validate inputs for graph build or training as needed:

```bash
python -c "from src.core.config import validate_graph_construction_inputs, validate_gnn_training_inputs; validate_graph_construction_inputs(); validate_gnn_training_inputs()"
```

4. Run targeted tests:

```bash
pytest tests/unit/
```

5. Verify new outputs are represented in docs and, if needed, in registry sentinels.

## 7) High-Risk Extension Areas

- changing feature order without updating all consumers
- altering fold logic without regenerating harmonized fold files
- changing threshold policy without regenerating evaluation outputs
- introducing optional branches without a quality gate

## 8) Documentation Update Rule

Any extension that changes runtime behavior should update these pages in the same change:

- `docs/architecture.md`
- `docs/components.md`
- `docs/configuration.md`
- `docs/usage.md`
- `docs/failure-modes.md`
