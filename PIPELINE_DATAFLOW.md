# Pipeline Integration Diagram & Data Flow Visualization

## Complete End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURO-CXG COMPLETE PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

STAGE 1: DATA ACQUISITION
┌─────────────────────────┐
│ ABIDE Download          │  src/data/abide_download.py
│ (Optional)              │  - Downloads from S3 API
└──────────────┬──────────┘  - Generates download_log.csv
               │
          ✓ data/images/
          ✓ Phenotypic_V1_0b_preprocessed1.csv
               │
               ▼
┌─────────────────────────┐
│ Stratified Split        │  src/data/split.py
│ 2D Stratification       │  - By DX_GROUP + SITE_ID
│ (70/15/15)              │  - Preserves subject grouping
└──────────────┬──────────┘
               │
          ✓ data/final/train/{images,labels,time_series}
          ✓ data/final/val/{images,labels,time_series}
          ✓ data/final/test/{images,labels,time_series}
               │
               ▼
┌─────────────────────────┐
│ Master Manifest         │  src/utils/manifestor.py
│ Generation              │  - Subject-level metadata
└──────────────┬──────────┘  - Diagnosis, site, demographics
               │
          ✓ master_manifest.csv
               │
               ├──────────────────────┐
               │                      │
               ▼                      ▼

STAGE 2: VALIDATION & LABELING (PARALLEL PATH)
┌──────────────────────┐  ┌──────────────────────┐
│ Atlas Validation     │  │ Label Annotation     │
│ (Optional)           │  │ YOLO Format          │
│                      │  │                      │
└──────────────────────┘  └──────────┬───────────┘
       ✓ Validation OK         │
       ✓ Atlas verified    ✓ Bounding boxes
                           ✓ YOLO format *.txt
                               │
                               ▼

STAGE 3: OBJECT DETECTION
┌──────────────────────────────┐
│ YOLO Training                │  src/pipelines/roi_detection.py
│ (5 Brain Lobes)              │  - Uses labeled images
│                              │  - 100 epochs, batch 24
└──────────────────┬───────────┘  - yolo11s model
                   │
              ✓ best.pt weights
                   │
                   ▼

STAGE 4: FEATURE EXTRACTION (PARALLEL PATHS)

PATH A: SPATIAL FEATURES          PATH B: TEMPORAL FEATURES
────────────────────────          ─────────────────────────
┌─────────────────────────┐  ┌─────────────────────────┐
│ Extract Features        │  │ Compute ROI Stats       │
│ YOLO Inference          │  │ Time Series Analysis    │
│                         │  │                         │
│ - Detections per lobe   │  │ - 6 features per ROI    │
│ - Mean coordinates      │  │ - 164-170 ROIs          │
│ - 3D aggregation        │  │ - Aggregated to 5 lobes │
│ - Filter: all 5 lobes   │  │                         │
└──────────────┬──────────┘  └──────────┬──────────────┘
               │                        │
          ✓ node_features_3d.csv   ✓ node_attributes_temporal.csv
               │                        │
               └────────┬───────────────┘
                        │
                        ▼

STAGE 5: BATCH EFFECT REMOVAL
┌────────────────────────────────┐
│ Feature Harmonization          │  src/features/safe_harmonization.py
│ neuroCombat with NaN handling  │  - Remove scanner batch effects
│                                │  - Protect DX_GROUP (diagnosis)
│                                │  - Fill missing demographics
└───────────────────┬────────────┘  - Outlier capping (5σ)
                    │
             ✓ node_attributes_harmonized.csv
                    │
                    ▼

STAGE 6: COMPREHENSIVE VALIDATION & TUNING ✨ NEW
┌────────────────────────────────────────────────────┐
│ Comprehensive Validation Suite   │  src/validation/validator.py
│ Quality & Distribution Analysis  │  - YOLO detection quality checks
│                                  │  - Graph sparsity distribution
│                                  │  - Feature preprocessing validation
│                                  │  - Stratification correctness
└──────────────────────┬───────────────────────────────┘
               │
          ✓ Validation Report
               │
               ▼

STAGE 7: PRE-GNN INTEGRITY CHECK
┌────────────────────────────────────────────────────┐
│ Pre-GNN Integrity Check          │  src/validation/integrity.py
│ Dataset Completeness             │  - Verify split distribution
│                                  │  - Check label file matching
│                                  │  - Validate feature shapes
└──────────────────────┬───────────────────────────────┘
               │
          ✓ Dataset OK
               │
               ▼

STAGE 8: GRAPH CONSTRUCTION
┌──────────────────────────────────┐
│ Causal Graph Construction        │  src/features/construct_causal.py
│ Lagged Correlation (t-1 → t)     │  - Aggregate 170 AAL → 5 lobes
│                                  │  - Compute partial correlations
│                                  │  - Lag=1 TR for temporal precedence
│                                  │  - Sparsify top 20% edges
└──────────────┬───────────────────┘
               │
          ✓ graph_0.pt, graph_1.pt, ..., graph_N.pt
          ✓ Format: PyTorch Geometric Data objects
          ✓ Shape: (5 nodes, 6+3 node features, ~5 edges)
               │
               ▼

STAGE 9: GNN TRAINING
┌──────────────────────────────────┐
│ Graph Neural Network Training    │  src/models/gnn_model.py
│ 5-Fold Stratified Cross-Val      │  - GATv2 with 2 attention heads
│                                  │  - 4 residual layers
│                                  │  - Label smoothing (0.1)
│                                  │  - Early stopping on AUC
│                                  │  - Dropout 0.5
└──────────────┬───────────────────┘
               │
          ✓ best_model_fold0.pt
          ✓ best_model_fold1.pt
          ✓ best_model_fold2.pt
          ✓ best_model_fold3.pt
          ✓ best_model_fold4.pt
               │
               ▼
        📊 FINAL PREDICTIONS
        ├─ Mean AUC: 0.5354 ± 0.0562
        ├─ Mean F1: 0.6586 ± 0.0164
        ├─ Per-fold confusion matrices
        └─ Feature importance via gradients


┌────────────────────────────────────────────────────────────────────────┐
│                      OPTIONAL VALIDATION PATHS                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ AFTER DOWNLOAD: Post-Download Integrity Check                        │
│   src/validation/integrity.py → check_dataset_integrity()            │
│   - Verify PNG files valid                                           │
│   - Verify NPY files valid                                           │
│   - Check subject slice counts                                       │
│   └─→ (Currently integrated ✓)                                        │
│                                                                        │
│ AFTER FEATURE EXTRACTION: Comprehensive Validation & Tuning ✨ NEW   │
│   src/validation/validator.py                                         │
│   - Check YOLO detection quality (survival rate, confidence)         │
│   - Analyze graph sparsity distribution                              │
│   - Verify stratification correctness                                │
│   - Feature distribution analysis                                    │
│   - Setup evaluation metrics                                         │
│   └─→ (NOW INTEGRATED ✓ - stage 6, triggered by --run-diagnostics)   │
│                                                                        │
│ ANYTIME: Diagnostics & Health Check                                  │
│   src/validation/pipeline_diagnostics.py                              │
│   - Comprehensive health check for all stages                        │
│   - Actionable fix recommendations                                   │
│   └─→ (Currently integrated - optional flag ✓)                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘


KEY DECISION POINTS IN PIPELINE:
════════════════════════════════════════════════════════════════════════

1. YOLO Weight Strategy
   ├─ If best.pt exists: Skip training (default)
   ├─ If missing: Train automatically
   └─ Override: --force-reset or --skip-yolo

2. Feature Extraction Dependencies
   ├─ Spatial: REQUIRES yolo best.pt + PNG images
   ├─ Temporal: REQUIRES time_series/*.npy files
   ├─ Both must succeed before harmonization
   └─ Any failure → skip harmonization/graphs/training

3. Graph Construction Filter
   ├─ CRITICAL: Requires all 5 lobes detected for EACH subject
   ├─ If any subject missing lobes → filtered out
   ├─ This defines final training set size
   └─ Impact: ~1033/1035 subjects typically pass

4. GNN Training
   ├─ Always offered to user (--auto runs it)
   ├─ 5-fold CV on all available graphs
   ├─ Can be re-run without recomputing graphs
   └─ Outputs: 5 model checkpoints (best per fold)


DEPENDENCY CHAIN VISUALIZATION:
════════════════════════════════════════════════════════════════════════

Download
  └─→ Split
       ├─→ Manifest ────────────────────────────────────────┐
       ├─→ Label Annotation ────────────────────┐           │
       │   └─→ YOLO Training                    │           │
       │       ├─→ Spatial Features             │           │
       │       └────────────────┐               │           │
       ├─→ Temporal Features ─────┤             │           │
       │                          ▼             │           │
       │ Harmonization ◄────────────┘           │           │
       │   └─→ [Comprehensive Validation] ✨   │           │
       │       └─→ Causal Graphs                │           │
       │           └─→ GNN Training ◄───────────┴───────────┘
       │
       └─→ Post-Download Integrity ✓


VALIDATION MODULE INTEGRATION STATUS (JANUARY 20, 2026):
════════════════════════════════════════════════════════════════════════════

Module                              Invoked?  When?                        Status
─────────────────────────────────── ────────  ─────────────────────────────────────────
atlas_validator.py                  ✓ YES     Stage 4                    ✓ Integrated
pipeline_diagnostics.py             ✓ YES     Optional (--run-diagnostics) ✓ Integrated
integrity.py ✨ NEW                 ✓ YES     Stage 7 (pre-GNN check)    ✓ Consolidated*
  ├─ check_dataset_integrity()      ✓ YES     Stage 1 (post-download)    ✓ Combined module
  └─ check_distribution()           ✓ YES     Stage 7 (pre-GNN)          ✓ Combined module
validator.py [COMPREHENSIVE] ✨     ✓ YES     Stage 6 (after features)   ✓ NOW INTEGRATED!

* integrity.py consolidates integrity_check.py + integrity_check2.py into single module
  (Deleted: integrity_check.py, integrity_check2.py)
  (Created: integrity.py with both validation functions)


COMMAND EXAMPLES:
════════════════════════════════════════════════════════════════════════

# Full pipeline with diagnostics
python src/run_pipeline.py --run-diagnostics

# Skip slow stages, run rest
python src/run_pipeline.py --skip-download --skip-yolo

# Dry run (show plan, don't execute)
python src/run_pipeline.py --dry-run

# Force complete rebuild
python src/run_pipeline.py --force-reset

# Manual comprehensive validation (workaround for missing integration)
python src/validation/validator.py

```

---

## Module Dependency Graph

```
src.run_pipeline (ORCHESTRATOR)
├── src.data.abide_download
│   └─→ Outputs: phenotype CSV, PNG images
├── src.data.split
│   └─→ Outputs: train/val/test splits
├── src.utils.manifestor
│   └─→ Outputs: master_manifest.csv
├── src.validation.atlas_validator
│   └─→ Validates: atlas files
├── src.validation.pipeline_diagnostics
│   └─→ Checks: overall health
├── src.validation.integrity ✨ NEW (consolidated)
│   ├─→ check_dataset_integrity(): Validates PNG/NPY files
│   └─→ check_distribution(): Validates dataset distribution
├── src.utils.annotate
│   └─→ Outputs: YOLO labels
├── src.pipelines.roi_detection
│   └─→ Outputs: best.pt weights
├── src.features.extract_features
│   └─→ Outputs: node_features_3d.csv
│       └── imports: src.features.graph_factory
├── src.utils.compute_roi
│   └─→ Outputs: node_attributes_temporal.csv
├── src.features.safe_harmonization
│   └─→ Outputs: node_attributes_harmonized.csv
├── src.validation.validator ✨ NEW (now integrated!)
│   └─→ Comprehensive validation: YOLO quality, sparsity, stratification
├── src.features.construct_causal
│   └─→ Outputs: causal_graphs/*.pt
├── src.models.gnn_model
│   └─→ Outputs: best_model_fold*.pt
│       └── imports: src.features.graph_factory
│       └── imports: src.models.causal_gnn
│
├── [AVAILABLE BUT NOT REQUIRED] src.data.check_progress
│   └─→ Progress tracking
└── [AVAILABLE BUT NOT REQUIRED] src.data.class_distribution
    └─→ Class balance analysis

Legend:
  ✓ Integrated: Module is called by run_pipeline.py (or invoked as subprocess)
  ✗ Not Integrated: Module exists but is never called
  (imports X) Dependency not exposed through orchestrator
  ✨ NEW/UPDATED: Recently added or consolidated (January 20, 2026)
```

