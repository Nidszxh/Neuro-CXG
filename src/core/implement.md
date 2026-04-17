This is a substantial codebase with detailed documentation. Let me provide a comprehensive analysis across all four dimensions.

---

## Dimension 1: Accuracy Improvement

**Diagnosis:** The 0.0947 CV–test AUC gap is the primary problem, not the CV AUC itself (0.74 is respectable for ABIDE). Root causes are site leakage persisting through spatial features, the GRL being disabled after damaging results, and 12-region lobe aggregation losing within-lobe heterogeneity that discriminates ASD subtypes.

**Root Causes (ranked):**

1. **Site confound in graph structure.** Granger causality is computed from TR-specific time series. Different TRs → different effective lags → structurally different graphs for same diagnosis across sites. The GRL was disabled (DD-007) because alpha=1.0 was too aggressive—but a tuned GRL is the correct fix, not removal. The pending grid search `{0.05, 0.10}` in Phase 11.2 is the right direction.

2. **12-region aggregation loses discriminative within-lobe variance.** PCA eigenvariate captures only the dominant signal direction per lobe. For lobes like Cingulate (34 ROIs spanning ACC/PCC) and Subcortical (38 ROIs), the second and third PCs carry ASD-relevant heterogeneity that's discarded. This structurally caps the graph's information content.

3. **Single Granger estimate per subject is noisy without effective invariance training.** The multi-view infrastructure (DD-010) is built but `GNN_INVARIANCE_WEIGHT=0.0` by default—it never activates unless the multiview_graphs stage ran and passed the quality gate. This means the model trains on single noisy causal estimates.

**Concrete Changes:**

1. **Enable GRL with tuned alpha [quick win, expected +0.03 test AUC].** The grid search infrastructure exists. Run it:
```python
# In hyperparams.py
GNN_USE_GRL = True
GNN_AUTO_GRL_GRID_SEARCH = True
GRL_ALPHA_CANDIDATES = [0.05, 0.10, 0.15, 0.20]
GNN_GRL_ALPHA_MAX = 0.15  # was 0.3, which caused collapse
```
The key fix: `set_grl_alpha()` already implements warmup (p < 0.2 → alpha=0) and cosine schedule. The old crash was from static alpha=1.0, not from GRL per se.

2. **Activate multi-view invariance training [medium effort, expected +0.02 CV AUC stability].** The infrastructure is complete. Run `--multiview` then set `GNN_INVARIANCE_WEIGHT = 0.15` in hyperparams.py. The quality gate in `_assess_multiview_quality()` will validate before training activates it.

3. **Add per-ROI standard deviation as an internal feature [quick win, expected +0.01 AUC].** Current internal features are coherence + spatial_variance (mean). Adding per-lobe signal variance captures ASD-relevant hyperactivation patterns:
```python
# In construct_causal.aggregate_to_lobes(), after spatial_variance:
roi_max_std = valid_rois.std(dim=0).max()  # max ROI std in lobe
lobe_internal_features.append(torch.stack([coherence, spatial_variance, roi_max_std]))
```
This requires updating `FEATURE_GROUPS["internal"]` from 2→3 features and `GNN_IN_CHANNELS` accordingly. The `assert NUM_SPATIAL_FEATURES == 4` pattern in feature_registry.py should be mirrored here.

4. **Run site-stratified CV (DD-013) before next training run [medium effort, expected honest -0.02 CV AUC but +0.03 test AUC].** The implementation is complete in `split.py`. This gives a more honest CV estimate that will likely narrow the CV–test gap by surfacing true generalization. Run:
```bash
python src/data/split.py --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

5. **Replace global mean-pooling with hierarchical attention (already done as DD-011, but verify it's active).** `GNN_POOLING = "anatomical"` is already the default. Confirm checkpoints are using it by checking `model.anatomical_pool` exists in saved state_dict keys.

6. **Threshold policy: switch to Youden J [quick win, expected +0.05 specificity].** Current specificity is 0.41—the model is biased toward sensitivity. In hyperparams.py:
```python
EVAL_THRESHOLD_POLICY = "youden"  # was "f1" or unset
```
This is already implemented in `run_evaluation.py`; just changing the config activates it.

7. **Remove beta frequency features or zero them per-subject [medium effort, expected +0.01 AUC].** Beta (0.15-0.20 Hz) is at 3/4 of Nyquist for TR=2s but at Nyquist for TR=2.5s sites (KKI, OHSU, SBL). The per-subject TR is available. Extend `UNRELIABLE_FREQ_BANDS_AT_NYQUIST` to be dynamically computed:

```python
# In extract_temporal.py, extract_band_power():
def get_unreliable_bands(tr: float) -> set:
    nyquist = 0.5 / tr
    return {name for name, (_, hi) in FREQ_BANDS.items() if hi >= nyquist * 0.9}
```

8. **Per-site Platt calibration [quick win, existing infrastructure].** The calibration code is already in `run_evaluation.py` but only activates when `calibration_graphs` is non-empty. The `_load_last_fold_val_graphs()` function should reliably return data—check it returns graphs and the site_ids align.

**Recommended Experiment Sequence:**
1. Site-stratified CV + retrain (establishes honest baseline)
2. GRL grid search enabled (directly targets site confound)
3. Activate multi-view invariance (after verifying multiview graphs built correctly)
4. Youden threshold at evaluation (immediate specificity improvement, no retrain)

---

## Dimension 2: Performance and Optimization

**Diagnosis:** The main bottlenecks are Granger causality (sequential pairwise tests across 132 directed pairs per subject), repeated CSV loading across folds (ABIDECausalDataset re-reads CSVs on every fold iteration), and graph loading without effective caching.

**Bottleneck Map:**

| Stage | Bottleneck Type | Fix |
|---|---|---|
| `compute_granger_causality` | CPU-bound, sequential pairwise | Already uses joblib; increase `n_jobs=-1`, verify it's actually parallel |
| `ABIDECausalDataset.__init__` | I/O: re-reads CSVs per fold | Feather caching exists (`_load_csv_cached`) but only triggers on second run |
| `aggregate_to_lobes` per-subject SVD | CPU-bound, sequential | Vectorize across subjects in batch mode |
| GNN training DataLoader | Workers=4 but dataset is small | Use `num_workers=2` for <800 samples (already done in `make_loader`) |
| Multi-view graph construction | I/O: 6 Granger runs per subject | Bootstrap views are independent—parallelize with ProcessPoolExecutor |

**Concrete Optimizations:**

1. **Pre-warm the Feather cache before fold iteration [quick win, ~30% faster CSV reads].** Currently `_load_csv_cached` writes Feather on first load. For 5-fold training, 4 fold-specific CSV reads happen cold. Fix: add an explicit cache warmup in `gnn_model.py` before the fold loop:
```python
# Before fold loop in _run_training_once():
from src.features.graph_factory import _load_csv_cached
_ = _load_csv_cached(NODE_ATTRIBUTES_HARMONIZED, index_col="subject_id")
_ = _load_csv_cached(NODE_FEATURES_3D, index_col="subject_id")
logger.info("CSV cache warmed")
```

2. **Pin graph dict tensors in memory [quick win, ~20% faster graph loading on GPU training].** The `_graph_cache` in ABIDECausalDataset stores raw dicts. For GPU training, pre-pin the adj tensors:
```python
# In _validate_subjects(), after loading graph_dict:
if torch.cuda.is_available():
    graph_dict['adj'] = graph_dict['adj'].pin_memory()
```

3. **Parallelize multi-view construction across subjects [medium effort, ~5x speedup for Stage 15].** In `main_multiview()`, the subject loop is sequential. The current `construct_multiview_graphs()` function is self-contained per subject. Use `ProcessPoolExecutor` with the same `init_worker` pattern as `abide_download.py`:
```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=6) as exe:
    futures = {
        exe.submit(construct_multiview_graphs, sub_id, ...): sub_id
        for sub_id in all_subjects
    }
```

4. **Cache Granger results across overlapping fold-train sets [medium effort].** Folds 0–4 share ~80% of training subjects. Granger is computed once per subject in `construct_causal.py` and saved, so this is already effectively cached. The real redundancy is in `_harmonize_train_apply_pair()` which re-aggregates ROI→lobe features for each fold. This runs 5 times; pre-aggregate once:
```python
# In main() of fold_safe_harmonization.py, before fold loop:
aggregated_all = aggregate_to_lobes(features_safe)
# Then pass to _harmonize_train_apply_pair instead of raw features
```
This saves ~4× the aggregation time (currently done inside each fold call).

5. **Reduce DataLoader workers for small datasets [already done, verify].** `make_loader()` already caps at 2 workers for <800 subjects. Confirm `persistent_workers=True` is active (it is in current code)—this avoids process respawn between epochs, saving ~2s per fold.

---

## Dimension 3: Code Refactoring

**Diagnosis:** The config is cleanly split (DD-006 resolved the monolith), but there's coupling between `gnn_model.py` (800+ lines) and `training_utils.py` (600+ lines) with shared state through mutable defaults. The `run_evaluation.py` and `run_result_analysis.py` independently reimplement per-fold checkpoint loading, prediction collection, and ensemble weighting.

**Files to Refactor:**

**Problem 1: Duplicated ensemble prediction logic across 3 runners**

`run_evaluation.py`, `run_result_analysis.py`, and `gnn_model.evaluate_ensemble()` each implement `_load_model(fold_id)` + loop over folds + weighted average. This is ~150 lines duplicated 3 times with subtle differences (run_evaluation uses `get_active_checkpoint_dir()`, run_result_analysis uses a different fallback).

**Before (run_result_analysis.py ~line 100):**
```python
for fold_id in range(K_FOLDS):
    try:
        model = _load_model(fold_id)
    except FileNotFoundError:
        continue
    loader = make_loader(graphs, batch_size=1, shuffle=False)
    probs = []
    for batch in loader:
        ...
        probs.append(p)
    fold_probs_list.append(np.array(probs))
```

**After (add `src/models/ensemble.py`):**
```python
# src/models/ensemble.py
from src.models.factory import build_model
from src.models.training_utils import make_loader
from src.core.config import K_FOLDS, GNN_BATCH_SIZE
import torch, numpy as np

@torch.no_grad()
def collect_fold_predictions(
    graphs: list,
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int = GNN_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Returns (ensemble_probs, labels, fold_aucs) from all available checkpoints."""
    fold_probs, fold_aucs, labels = [], [], None
    loader = make_loader(graphs, batch_size=batch_size, shuffle=False)
    
    for fold_id in range(K_FOLDS):
        ckpt_path = checkpoint_dir / f"best_model_fold{fold_id}.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(device=device)
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        model.eval()
        
        probs, batch_labels = [], []
        for batch in loader:
            if batch is None: continue
            batch = batch.to(device)
            out = model.forward_batch(batch)
            probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
            batch_labels.extend(batch.y.cpu().numpy())
        
        fold_probs.append(np.array(probs))
        fold_aucs.append(float(ckpt.get("auc", 0.5)))
        if labels is None:
            labels = np.array(batch_labels)
    
    if not fold_probs:
        raise RuntimeError("No fold checkpoints found")
    
    weights = np.array(fold_aucs) / sum(fold_aucs)
    ens_probs = np.average(np.stack(fold_probs), axis=0, weights=weights)
    return ens_probs, labels, fold_aucs
```

All three runners then call `collect_fold_predictions()` and add their domain-specific logic on top.

**Problem 2: `gnn_model.py` is 800+ lines with training orchestration, loss classes, and utility functions co-mingled**

`FocalLoss`, `CausalInvarianceLoss`, and `SpatialInvarianceLoss` belong in `training_utils.py` (or a new `src/models/losses.py`) since they're consumed by the training loop, not by evaluation. `FocalLoss` is duplicated between `gnn_model.py` and `run_evaluation.py`'s `FlatMLP` section. The `compute_class_weights`, `find_optimal_threshold`, `evaluate`, and `evaluate_ensemble` helpers in `gnn_model.py` partially duplicate `src/models/evaluation.py`.

**Modularity Checklist:**
- [ ] `src/models/losses.py`: Move `FocalLoss`, `CausalInvarianceLoss`, `SpatialInvarianceLoss` here
- [ ] `src/models/ensemble.py`: Shared fold prediction collection (above)
- [ ] `gnn_model._run_training_once()`: Split into `_setup_training()` + `_run_fold()` + `_finalize()`
- [ ] `run_evaluation.py` → import `FlatMLP` from `src/models/factory.py` or `losses.py`, not redeclare inline
- [ ] `src/validation/pipeline_checks.py` has 1000+ lines; split `PipelineValidator` into `src/validation/validators/data_validator.py`, `feature_validator.py`, `graph_validator.py`
- [ ] `CausalInvarianceLoss._VIEW_ORDER` referenced in `gnn_model._assess_multiview_quality()` but `_VIEW_ORDER` is not defined on that class (it's `_MULTIVIEW_VIEW_ORDER` in `training_utils.py`)—this is a latent bug

---

## Dimension 4: Architecture and Design

**Diagnosis:** The pipeline has excellent separation of concerns at the stage level (registry-driven) but lacks experiment tracking integration—the `ExperimentTracker` is called but results aren't systematically comparable across runs. The fold-specific harmonization files are a required precondition for training that isn't enforced until training starts (causing a hard error deep in the loop).

**Target Architecture:**

```
src/
├── core/                    # Config, registry, paths [stable]
│   ├── config.py            # Re-export shim (keep)
│   ├── hyperparams.py
│   ├── feature_registry.py
│   ├── atlas_config.py
│   ├── paths.py
│   └── validators.py
├── data/                    # Pure I/O, no ML [stable]
├── features/                # Feature extraction, graph construction [stable]
├── models/
│   ├── losses.py            # FocalLoss, CausalInvarianceLoss, SpatialInvarianceLoss [NEW]
│   ├── ensemble.py          # Shared fold prediction collection [NEW]
│   ├── factory.py           # build_model() [stable]
│   ├── causal_gnn.py        # Architecture only [stable]
│   ├── evaluation.py        # Metrics computation [stable]
│   ├── training_utils.py    # Training loop primitives [stable]
│   └── gnn_model.py         # Orchestration only (~300 lines after extraction)
├── pipeline/
│   └── registry.py          # Stage metadata [stable]
└── run_pipeline.py          # Thin orchestrator [stable]
```

**Design Changes:**

1. **Precondition validation before fold loop, not inside it [medium effort].** The hard `FileNotFoundError` assertion for fold harmonization files fires inside the fold loop, wasting time if fold 0 succeeds but fold 3 is missing. Move to `validate_gnn_training_inputs()`:
```python
# In validators.py, validate_gnn_training_inputs():
from src.core.paths import HARMONIZED_FOLDS_DIR
from src.core.hyperparams import K_FOLDS
missing = [f for f in range(K_FOLDS) 
           if not (HARMONIZED_FOLDS_DIR / f"harmonized_fold_{f}.csv").exists()]
if missing:
    raise FileNotFoundError(
        f"Missing fold harmonization files for folds {missing}. "
        "Run: python -m src.features.fold_safe_harmonization"
    )
```

2. **Make `ExperimentTracker` queryable for run comparison [medium effort].** Currently each run writes a `run.json` but there's no aggregation. Add a `compare_runs()` function:
```python
# src/core/experiment_tracker.py
@classmethod
def compare_runs(cls, output_root: Path) -> pd.DataFrame:
    """Return DataFrame of all completed runs sorted by mean_auc."""
    import pandas as pd, json
    rows = []
    for run_json in output_root.glob("*/run.json"):
        rec = json.loads(run_json.read_text())
        summary = rec.get("summary", {})
        rows.append({
            "run_id": rec["run_id"],
            "mean_auc": summary.get("mean_auc", 0.0),
            "std_auc": summary.get("std_auc", 0.0),
            "grl_alpha": rec["notes"].get("grl_alpha"),
            "config_hash": rec.get("config_hash"),
        })
    return pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
```

3. **Stage dependency enforcement in registry [quick win].** The `Stage.is_complete()` checks sentinel files, but `run_pipeline.py` builds dependency flags manually (45+ `_ready_or_planned` variables). The registry `dependencies` field is declared but never enforced—`stage_should_run` ignores it. Fix by having the runner compute transitive completion:
```python
# In run_pipeline.py, after building stage_map():
def _deps_satisfied(stage_key: str, completed: set) -> bool:
    stage = registry_by_key[stage_key]
    return all(d in completed or stages.get(d, {}).get("should_run", False)
               for d in stage.dependencies)
```

4. **Artifact versioning for checkpoints [medium effort].** Currently `CheckpointManager` saves to `best_model_fold{N}.pt` without run ID. A second training run overwrites the canonical run (this is the documented "fold3 epoch=2 collapsed run" issue). Fix:
```python
# In CheckpointManager.save():
filename = (
    f"best_model_fold{fold}_{self.run_id}.pt" if fold is not None 
    else f"best_model_{self.run_id}.pt"
)
# Maintain symlink for current best
symlink = self.checkpoint_dir / f"best_model_fold{fold}.pt"
symlink.unlink(missing_ok=True)
symlink.symlink_to(filename)
```
This preserves history while keeping the `best_model_fold*.pt` pattern that `run_evaluation.py` expects.

5. **Onboarding: add a `--verify` mode to `run_pipeline.py` [quick win].** New contributors can't easily tell what data is needed vs. already generated. A lightweight verification pass that checks all sentinels without running would help:
```python
# In run_pipeline.py main():
parser.add_argument("--verify", action="store_true",
    help="Check which stages are complete without running anything")
if args.verify:
    snap = completion_snapshot()
    for key, done in snap.items():
        print(f"{'✓' if done else '○'} {key}")
    sys.exit(0)
```