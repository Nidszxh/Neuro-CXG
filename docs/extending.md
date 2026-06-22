# Extending Neuro-CXG

## Goal

Use this guide when adding new capabilities while preserving existing pipeline contracts and reproducibility safeguards.

## Extension Surfaces

- Stage orchestration (`src/pipeline/registry.py`, `src/run_pipeline.py`)
- Feature schema (`src/core/feature_registry.py`, `src/features/*`, `src/features/graph_factory.py`)
- Model/training behavior (`src/models/*.py`, `src/core/hyperparams.py`)
- Reporting outputs (`src/run_*.py`)

## 1) Add a New Pipeline Stage

**Where to make the change:**
1. Add stage metadata in `src/pipeline/registry.py`:
   - Unique `key`
   - `module`
   - Optional `function`
   - Dependency list
   - Reliable output sentinel
2. Ensure runner can execute it in context (`src/run_pipeline.py`)
3. If the stage produces critical artifacts, add or update validation checks

**Contracts to update:**
- Registry sentinel in `src/pipeline/registry.py`
- Stage dependencies in runner logic

**Tests to add:**
- Verify stage execution in dry-run
- Verify sentinel creation

**Sentinel rule:** Pick an output sentinel that is both deterministic for successful execution and specific enough to avoid false positives.

---

## 2) Add or Modify Feature Channels

**Where to make the change:**
1. Update feature definitions in `src/core/feature_registry.py`
2. Keep `ALL_FEATURE_NAMES` order stable and intentional
3. Update extraction code to write compatible artifacts
4. Update dataset assembly in `src/features/graph_factory.py`
5. Verify model input dimensions (`GNN_IN_CHANNELS`) are still consistent

**Contracts to update:**
- Feature registry (`ALL_FEATURE_NAMES`, `FEATURE_GROUPS`)
- GNN_IN_CHANNELS computation
- Graph factory shape checks

**Tests to add:**
- Verify feature dimension alignment
- Verify no NaN/Inf in new features

⚠️ **Critical constraint:** Do not reintroduce site-leaky spatial channels (`conf_std`, `detection_count`) into model input unless intentionally running a controlled experiment.

---

## 3) Add a New Causality or Graph Method

**Where to make the change:**
1. Add method config in `src/core/hyperparams.py`
2. Implement method path in `src/features/construct_causal.py` and/or `src/features/causal_inference.py`
3. Preserve graph package contract expected by `ABIDECausalDataset`
4. Validate downstream sparsity and degeneracy behavior

**Contracts to update:**
- Graph package contract (adjacency, internal_features, edge_confidence, etc.)
- Degeneracy gate thresholds

**Tests to add:**
- Verify graph quality gates pass
- Verify edge sparsity within expected range

**Safety rule:** If a new method can produce many zero-edge graphs, add diagnostics and gate checks before enabling it by default.

---

## 4) Extend Model Architecture or Training Objectives

**Where to make the change:**
1. Introduce knobs in `src/core/hyperparams.py` with stable defaults
2. Implement architecture changes in `src/models/causal_gnn.py` and construction path in `src/models/factory.py`
3. Wire training objective changes in `src/models/gnn_model.py` / `src/models/training_utils.py`
4. Ensure checkpoints remain loadable by evaluation/explainability scripts

**Contracts to update:**
- Model architecture signatures
- Checkpoint metadata
- Evaluation script loading

**Tests to add:**
- Verify training completes
- Verify checkpoint loading works

**Compatibility rule:** Keep backward-compatible checkpoint loading paths where practical (shape inference and `strict=False` loading are already used in evaluation scripts).

---

## 5) Extend Evaluation or Reporting

**Where to make the change:**
1. Add computation and outputs in the relevant `src/run_*.py` script
2. Preserve machine-readable summary artifacts:
   - `results/evaluation/comprehensive_results.json`
   - `results/explainability/summary.json`
   - `results/analysis/result_analysis_summary.json`
3. If output names/locations change, update stage sentinels in `src/pipeline/registry.py`

**Contracts to update:**
- Output JSON schemas
- Stage sentinels

**Drift prevention:** `run_result_analysis.py` uses evaluation metadata when available. Keep threshold-policy metadata consistent to avoid report divergence.

---

## 6) High-Risk Extension Areas

- Changing feature order without updating all consumers
- Altering fold logic without regenerating harmonized fold files
- Changing threshold policy without regenerating evaluation outputs
- Introducing optional branches without a quality gate

---

## 7) Documentation Update Rule

Any extension that changes runtime behavior should update these files in the same change:

- `docs/architecture.md`
- `docs/configuration.md`
- `docs/setup.md`
- `docs/operations.md`
- `docs/paper/results.md`
- Root `README.md` (if important)

---

## 8) Extension Validation Checklist

Before considering an extension complete:

1. **Dry-run orchestration plan:**
   ```bash
   python src/run_pipeline.py --dry-run
   ```

2. Rebuild only affected stages

3. **Validate inputs for training:**
   ```bash
   python -c "from src.core.config import validate_gnn_training_inputs; validate_gnn_training_inputs()"
   ```

4. **Run targeted tests:**
   ```bash
   pytest tests/unit/
   ```

5. Verify new outputs are represented in docs and, if needed, in registry sentinels