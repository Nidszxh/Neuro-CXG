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
│ (12-Region Brain Model)      │  - Uses labeled images
│                              │  - 100 epochs, batch 32
└───────────────────┌───────────┘  - yolo26n model
                   │              - Results: mAP50-95=0.94073 (v26)
              ✓ best.pt weights  - Precision=0.98012, Recall=0.97754
                   │
                   ▼

STAGE 4: FEATURE EXTRACTION (PARALLEL PATHS)

PATH A: SPATIAL FEATURES          PATH B: TEMPORAL FEATURES
────────────────────────          ─────────────────────────
┌─────────────────────────┐  ┌─────────────────────────┐
│ Extract Features        │  │ Compute ROI Stats       │
│ YOLO Inference          │  │ Time Series Analysis    │
│                         │  │                         │
│ - Detections per region │  │ - 8 features per ROI    │
│ - Mean coordinates      │  │   (mean,std,skew,kurt,  │
│ - 3D aggregation        │  │   PSD,MSSD,range,auto)  │
│ - Filter: all 12 regions│  │ - 170 AAL ROIs          │
│ - 6 spatial features    │  │ - Aggregated to 12      │
│   per region            │  │   regions               │
└──────────────┬──────────┘  └──────────┬──────────────┘
               │                        │
          ✓ node_features_3d.csv   ✓ node_attributes_temporal.csv
               │                        │
               └────────┬───────────────┘
                        │
                        ▼

STAGE 5: BATCH EFFECT REMOVAL
┌────────────────────────────────┐
│ Fold-Safe Harmonization        │  src/features/fold_safe_harmonization.py
│ neuroHarmonize (CV-safe)       │  - Remove scanner batch effects
│                                │  - Protect DX_GROUP (diagnosis)
│                                │  - Fill missing demographics (age,sex)
└───────────────────┬────────────┘  - Outlier capping (5σ threshold)
                │                - 28 features per region maintained
           ✓ node_attributes_harmonized.csv   (20 temporal + 2 internal + 6 spatial)
           ✓ data/metadata/harmonized_folds_cv/fold*_*.csv
                │
                ▼

STAGE 6: COMPREHENSIVE VALIDATION & TUNING ✨ UPDATED (Feb 28)
┌────────────────────────────────────────────────────┐
│ Multi-Level Validation Suite     │  src/validation/
│ Quality & Distribution Analysis  │  - pipeline_checks.py: YOLO quality, sparsity
│                                  │  - dev_audit.py: Deep checks (--features flag)
│                                  │  - Feature distribution validation
│                                  │  - Stratification correctness
│                                  │  - Training readiness checks
└──────────────────────┬───────────────────────────────┘
               │
          ✓ Validation Reports
               │
               ▼

STAGE 7: PRE-GNN INTEGRITY CHECK
┌────────────────────────────────────────────────────┐
│ Pre-GNN Integrity Check          │  src/validation/pipeline_checks.py
│ Dataset Completeness             │  - Verify split distribution
│                                  │  - Check label file matching
│                                  │  - Validate 12-region graphs
└──────────────────────┬───────────────────────────────┘
               │
          ✓ Dataset OK
               │
               ▼

STAGE 8: GRAPH CONSTRUCTION
┌──────────────────────────────────┐
│ Causal Graph Construction        │  src/features/construct_causal.py
│ ✨ Granger Causality (default)   │  src/features/causal_inference.py
│ OR Lagged Pearson Correlation    │  - Aggregate 170 AAL → 12 regions
│ (multi-lag 1-5 TRs)              │  - PCA eigenvariate + ReHo aggregation
│                                  │  - Compute Granger causality or
│                                  │    lagged correlations
│                                  │  - Adaptive sparsification (0.70 quantile)
│                                  │  - Min 12 edges per graph
│                                  │  - 28 node features per region
└──────────────┬───────────────────┘
               │
          ✓ graph_0.pt, graph_1.pt, ..., graph_N.pt
          ✓ Format: PyTorch Geometric Data objects
          ✓ Shape: (12 nodes, 28 features, top 30% edges)
          ✓ Total graphs: 1035 subjects (702 train, 152 val, 152 test)
               │
               ▼

STAGE 9: GNN TRAINING
┌──────────────────────────────────┐
│ Graph Neural Network Training    │  src/models/gnn_model.py
│ 5-Fold Stratified Cross-Val      │  - ✨ GATv2Conv with 3 layers (current default)
│ ✨ 28 Input Features              │  - 4 attention heads per layer
│                                  │  - 128 hidden channels
│                                  │  - GELU activation (improved gradients)
│                                  │  - Attention pooling (GlobalAttention)
│                                  │  - Focal loss (α=0.62, γ=2.0)
│                                  │  - Early stopping on AUC (patience=20)
└──────────────┬───────────────────┘  - Dropout 0.45, L2 reg 1e-4, gradient clip 1.0
               │                       - Current: AUC=0.5593±0.0156 (Phase 3 stable)
          ✓ best_model_fold0.pt          ✅ Phase 3 Complete: Simplified architecture
          ✓ best_model_fold1.pt
          ✓ best_model_fold2.pt
          ✓ best_model_fold3.pt
                ✓ best_model_fold4.pt
                         │
                         ▼
            📊 FINAL PREDICTIONS & METRICS (Phase 3 Stable)
            ├─ YOLO mAP50-95: 0.94073 (v26, epoch 100) ✅
            ├─ Features: 28 dimensions (20 temporal + 2 internal + 6 spatial) ✨
            ├─ Causal: Granger + lagged correlation methods ✨
            ├─ GNN Baseline: AUC 0.5593 ± 0.0156 (28-feature model)
            ├─ Per-fold confusion matrices
            └─ Feature importance via gradients


┌────────────────────────────────────────────────────────────────────────┐
│                      OPTIONAL VALIDATION PATHS                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│ AFTER DOWNLOAD: Post-Download Integrity Check                        │
│   src/validation/pipeline_checks.py → check_dataset_integrity()      │
│   - Verify PNG files valid                                           │
│   - Verify NPY files valid                                           │
│   - Check subject slice counts                                       │
│   └─→ (Currently integrated ✓)                                        │
│                                                                        │
│ AFTER FEATURE EXTRACTION: Comprehensive Validation & Tuning ✨ NEW   │
│   src/validation/pipeline_checks.py                                   │
│   - Check YOLO detection quality (survival rate, confidence)         │
│   - Analyze graph sparsity distribution                              │
│   - Verify stratification correctness                                │
│   - Feature distribution analysis                                    │
│   - Setup evaluation metrics                                         │
│   └─→ (Integrated by default; skip with --skip-comprehensive-validation) │
│                                                                        │
│ ANYTIME: Diagnostics & Health Check                                  │
│   src/validation/pipeline_checks.py                                   │
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
     ├─ CRITICAL: Requires all 12 lobes detected for EACH subject
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
pipeline_checks.py ✨ NEW           ✓ YES     Stage 6/7                  ✓ Consolidated*
     ├─ check_dataset_integrity()      ✓ YES     Stage 1 (post-download)    ✓ Combined module
     └─ check_distribution()           ✓ YES     Stage 7 (pre-GNN)          ✓ Combined module
dev_audit.py                         ○ NO     Manual (--features flag)    ✓ Available

* pipeline_checks.py consolidates integrity_check.py + integrity_check2.py into single module
     (Deleted: integrity_check.py, integrity_check2.py)
     (Created: pipeline_checks.py with both validation functions)


COMMAND EXAMPLES:
════════════════════════════════════════════════════════════════════════

# Full pipeline (auto-run missing stages)
python src/run_pipeline.py --auto

# Skip slow stages, run rest
python src/run_pipeline.py --auto --skip-download --skip-yolo

# Dry run (show plan, don't execute)
python src/run_pipeline.py --dry-run

# Force complete rebuild
python src/run_pipeline.py --auto --force-reset

# Manual comprehensive validation (workaround for missing integration)
python src/validation/pipeline_checks.py --quality

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
├── src.validation.pipeline_checks ✨ UPDATED (consolidated, Feb 11, 2026)
│   ├─→ check_dataset_integrity(): Validates PNG/NPY files
│   └─→ check_distribution(): Validates dataset distribution
├── src.pipelines.generate_labels
│   └─→ Outputs: YOLO labels
├── src.pipelines.roi_detection
│   └─→ Outputs: best.pt weights
├── src.features.extract_spatial
│   └─→ Outputs: node_features_3d.csv
│       └── imports: src.features.graph_factory
├── src.features.extract_temporal
│   └─→ Outputs: node_attributes_temporal.csv
├── src.features.fold_safe_harmonization
│   └─→ Outputs: node_attributes_harmonized.csv
├── src.validation.pipeline_checks ✨ NEW (now integrated!)
│   └─→ Comprehensive validation: YOLO quality, sparsity, stratification
├── src.features.construct_causal
│   └─→ Outputs: causal_graphs/*.pt
├── src.models.gnn_model
│   └─→ Outputs: best_model_fold*.pt
│       └── imports: src.features.graph_factory
│       └── imports: src.models.causal_gnn
│
├── [VALIDATION SUITE] src.validation.* (3 modules)
│   ├── atlas_validator.py: Atlas file validation
│   ├── dev_audit.py: Deep validation checks (merged from code_audit + feature_diagnostics, Feb 28)
│   └── pipeline_checks.py: Post-download and pre-GNN checks (4 functions)

Legend:
  ✓ Integrated: Module is called by run_pipeline.py (or invoked as subprocess)
  ✗ Not Integrated: Module exists but is never called
  (imports X) Dependency not exposed through orchestrator
  ✨ NEW/UPDATED: Recently added or consolidated (Feb 11, 2026)
```

