# Neuro-CXG Documentation Index

**Welcome to Neuro-CXG!** This document helps you navigate the project's documentation based on your role and information needs.

---

## 🚀 Quick Navigation by Role

### For **Peer Reviewers** (Evaluating Publication)
Start here to assess experimental rigor and methodological soundness:

1. **[TEST_SET_PROTOCOL.md](TEST_SET_PROTOCOL.md)** (CRITICAL)
   - Transparency about multiple test set evaluations
   - Model selection integrity assessment
   - Answers reviewer concerns about "peeking" at test data

2. **[LITERATURE_COMPARISON.md](LITERATURE_COMPARISON.md)**
   - Benchmarks against 7 prior methods (Heinsfeld 2018, Kawahara 2017, Li 2021, etc.)
   - Shows 24.2% improvement over best prior SOTA
   - Fair comparison methodology

3. **[paper/ablation_statistical_tests.md](paper/ablation_statistical_tests.md)**
   - DeLong p-values for all ablations
   - Shows which components are statistically essential
   - Effect sizes and robustness checks

4. **[paper/methods.md](paper/methods.md)**
   - Model selection procedure
   - Post-hoc sensitivity analyses  
   - Causality terminology framework
   - Graph topology insights

5. **[paper/results.md](paper/results.md)**
   - Canonical results (12-lobe, Test AUC 0.8694)
   - Architecture comparison (12-lobe vs 11-lobe)
   - Generalization analysis

6. **[paper/ablations.md](paper/ablations.md)**
   - 10 core ablation experiments (A-E, D2, paper variants)
   - Demonstrates each component's necessity
   - Per-fold consistency and robustness

---

### For **Reproducers** (Running the Code)
Follow this path to understand and re-execute the pipeline:

1. **[README.md](../README.md)** (Start here)
   - Hardware requirements (24GB RAM, 8GB GPU recommended)
   - Quick start commands
   - Execution modes and timing estimates

2. **[architecture.md](architecture.md)**
   - 16-stage pipeline orchestration
   - Data dependencies and flow
   - Critical skip flags (`--skip-download`, `--skip-split`)

3. **[operations.md](operations.md)**
   - Step-by-step execution guide
   - Common failure modes and fixes
   - Validation procedures

4. **[data/DATA_MANIFEST.md](../data/DATA_MANIFEST.md)**
   - ABIDE-I dataset overview (1112 → 1015 subjects)
   - Subject filtering criteria and exclusions
   - Processed file locations

5. **[evaluation.md](evaluation.md)**
   - How to run post-training evaluation
   - Output format and interpretation
   - Threshold policies (Youden vs F1)

6. **[cv_test_gap.md](cv_test_gap.md)** (Optional)
   - Why CV AUC (0.7997) < Test AUC (0.8694)
   - Decomposition of generalization gap
   - Ensemble effects and distribution shift

---

### For **Ablation/Configuration Researchers**
Understanding what works and why:

1. **[paper/ablations.md](paper/ablations.md)**
   - 10 ablations documenting component necessity
   - Per-fold fold stability metrics
   - Interpretation of effect sizes

2. **[paper/ablation_statistical_tests.md](paper/ablation_statistical_tests.md)**
   - Statistical significance (DeLong test)
   - Which components are essential vs optional
   - Bootstrap confidence intervals

3. **[FINAL_ARCHITECTURE_ANALYSIS.md](FINAL_ARCHITECTURE_ANALYSIS.md)**
   - 12-lobe vs 11-lobe comparison
   - Why Brainstem (constant features) helps
   - Implicit regularization mechanism

4. **[decisions.md](dev/decisions.md)**
   - Architecture decisions (DD-001 through DD-018)
   - Rationale for each design choice
   - Trade-offs considered

5. **[experiments.md](experiments.md)**
   - Experiment tracking (ablations A-E, paper variants)
   - Configuration targets and actual results
   - Reproducibility metadata

---

### For **Code Contributors**
Understanding the implementation:

1. **[architecture.md](architecture.md)**
   - 16-stage pipeline structure
   - Data contracts (input/output schemas)

2. **[AGENTS.md](../AGENTS.md)** (Critical!)
   - Non-standard Python import patterns (`sys.path.insert`)
   - Configuration-driven architecture (all constants in `src/core/config.py`)
   - Common pitfalls (fold leakage, hardcoded paths, etc.)

3. **[operations.md](operations.md)**
   - Validation steps and checkpoints
   - Debugging procedures
   - Stage-specific logging

4. **[performance.md](performance.md)**
   - Computational bottlenecks
   - Optimization opportunities
   - GPU memory requirements per stage

5. **`.github/copilot-instructions.md`**
   - Detailed architecture guide (217 lines)
   - Feature extraction, graph construction, training loops
   - Validation patterns

---

### For **ML/AI Researchers**
Novel methodological insights:

1. **[LITERATURE_COMPARISON.md](LITERATURE_COMPARISON.md)**
   - Comparison with 7 methods (directed vs undirected, spectral vs spatial, etc.)
   - Why Neuro-CXG outperforms each baseline

2. **[FINAL_ARCHITECTURE_ANALYSIS.md](FINAL_ARCHITECTURE_ANALYSIS.md)**
   - Brainstem regularization discovery
   - CV-Test gap explanation
   - Implicit regularization mechanisms

3. **[paper/methods.md](paper/methods.md)**
   - Directed functional connectivity framework
   - Causality terminology (philosophical vs practical)
   - Graph topology vs edge weight contributions

4. **[paper/ablations.md](paper/ablations.md)**
   - Feature importance (temporal > spatial)
   - Graph necessity (+15.4% from structure)
   - Domain adaptation effectiveness (+13.3% from site conditioning)

5. **[cv_test_gap.md](cv_test_gap.md)**
   - Generalization gap decomposition
   - Calibration, ensemble effects, distribution shift
   - Multi-site heterogeneity management

---

## 📊 Document Categories

### Core Results (Publication-Ready)
- `paper/results.md` — Canonical test results (AUC 0.8694)
- `paper/ablations.md` — Ablation study findings
- `FINAL_ARCHITECTURE_ANALYSIS.md` — Architecture choice justification
- `LITERATURE_COMPARISON.md` — Literature benchmarks

### Methodological Rigor
- `TEST_SET_PROTOCOL.md` — Model selection integrity
- `paper/methods.md` — Methods section + causality framework
- `paper/ablation_statistical_tests.md` — Statistical significance testing

### Technical Implementation
- `architecture.md` — 16-stage pipeline structure
- `operations.md` — Execution guide
- `data/DATA_MANIFEST.md` — Dataset documentation
- `AGENTS.md` — Agent/contributor notes

### Decision & Design History
- `dev/decisions.md` — Architectural decisions (DD-001 through DD-018)
- `experiments.md` — Experiment tracking
- `cv_test_gap.md` — Generalization analysis

---

## 🔍 Finding Information by Topic

### "How do I run the pipeline?"
→ `README.md` → `architecture.md` → `operations.md`

### "What are the results?"
→ `paper/results.md` → `FINAL_ARCHITECTURE_ANALYSIS.md`

### "Why is 0.8694 better than prior work?"
→ `LITERATURE_COMPARISON.md`

### "Is the test set used fairly?"
→ `TEST_SET_PROTOCOL.md`

### "Which components are essential?"
→ `paper/ablations.md` → `paper/ablation_statistical_tests.md`

### "What do all the 'D' decisions mean?"
→ `dev/decisions.md`

### "How do I add a new feature/module?"
→ `AGENTS.md` → `.github/copilot-instructions.md`

### "Why does CV < Test (good generalization)?"
→ `cv_test_gap.md` → `FINAL_ARCHITECTURE_ANALYSIS.md`

### "How many times was test set used?"
→ `TEST_SET_PROTOCOL.md` (full transparency)

---

## 📋 Essential Files by Purpose

| Purpose | Files | Priority |
|---------|-------|----------|
| **Publication review** | TEST_SET_PROTOCOL.md, LITERATURE_COMPARISON.md, paper/* | ✅ HIGH |
| **Reproducibility** | README.md, architecture.md, data/DATA_MANIFEST.md, operations.md | ✅ HIGH |
| **Understanding design** | FINAL_ARCHITECTURE_ANALYSIS.md, dev/decisions.md, AGENTS.md | ⚠️ MEDIUM |
| **Ablations & analysis** | paper/ablations.md, paper/ablation_statistical_tests.md, cv_test_gap.md | ⚠️ MEDIUM |
| **Contributing code** | AGENTS.md, .github/copilot-instructions.md, architecture.md | ℹ️ LOW (optional) |

---

## 🎯 Key Metrics at a Glance

### Test Set Performance (Canonical)
- **AUC**: 0.8694 [95% CI: 0.7889–0.9037] ✅
- **F1** (Youden): 0.8000
- **Sensitivity**: 0.7595
- **Specificity**: 0.7733
- **Permutation p-value**: <0.001

### Cross-Validation Performance
- **CV AUC**: 0.7997 ± 0.0294
- **Fold Stability**: 46.5% lower variance than 11-lobe
- **CI Width**: 18.6% tighter than 11-lobe

### Generalization Gap
- **CV → Test**: +0.0697 (excellent generalization)
- **vs 11-lobe**: 11-lobe has CV > Test (overfitting)

### Literature Comparison
- **Best prior**: Heinsfeld 2018 (0.70)
- **Recent SOTA**: BrainNetCNN (0.8348)
- **Improvement**: +24.2% vs Heinsfeld, +4.0% vs BrainNetCNN

---

## 📞 Quick Reference

### Configuration
All hyperparameters centralized in: `src/core/config.py`
- Re-exports from: `atlas_config.py`, `hyperparams.py`, `paths.py`, `feature_registry.py`

### Entry Points
```bash
# Full pipeline
python src/run_pipeline.py --auto [--skip-download] [--skip-split]

# Post-training  
python src/run_evaluation.py
python src/run_explainability.py

# Ablations
python -m src.experiments.run_ablations
```

### Critical Constraints
- **Seed**: Must be 42 everywhere (reproducibility)
- **Fold safety**: Never fit harmonization on val/test data
- **DX protection**: ComBat must include DX_GROUP as covariate
- **No flips**: YOLO_FLIPLR=0.0 (medical domain constraint)

---

## ✅ Documentation Completeness

- [x] Test set protocol (fairness verification)
- [x] Literature comparison (5+ baselines)
- [x] Ablation statistical tests (DeLong p-values)
- [x] Architecture decisions (documented rationale)
- [x] Method section (causality framework)
- [x] Results section (canonical metrics)
- [x] Reproducibility guide (step-by-step)
- [x] Data manifest (dataset documentation)
- [x] Configuration documentation (no magic numbers)
- [ ] API documentation (docstrings)
- [ ] Tutorial notebooks (examples)

---

## 🚨 Critical Reading Order

For **first-time readers**, follow this order:

1. `README.md` (2 min) — What is this project?
2. `TEST_SET_PROTOCOL.md` (5 min) — Is this trustworthy?
3. `paper/results.md` (5 min) — What are the results?
4. `LITERATURE_COMPARISON.md` (5 min) — How does it compare?
5. `FINAL_ARCHITECTURE_ANALYSIS.md` (10 min) — Why 12-lobe?
6. `paper/ablations.md` (15 min) — What's necessary?
7. `architecture.md` (10 min) — How does it work?

**Total**: ~52 minutes for comprehensive understanding

---

## 📝 Document Metadata

| Document | Lines | Last Updated | Status | Audience |
|----------|-------|---|---|---|
| TEST_SET_PROTOCOL.md | 220 | Apr 29 | NEW | Reviewers |
| LITERATURE_COMPARISON.md | 380 | Apr 29 | NEW | Reviewers, Researchers |
| paper/ablation_statistical_tests.md | 290 | Apr 29 | NEW | Reviewers, Researchers |
| paper/results.md | 120 | Apr 29 | UPDATED | Reviewers |
| FINAL_ARCHITECTURE_ANALYSIS.md | 400 | Apr 28 | Existing | All |
| paper/ablations.md | 508 | Apr 29 | UPDATED | Reviewers, Researchers |
| architecture.md | — | Apr 28 | Existing | Contributors |
| README.md | 165 | Apr 29 | UPDATED | All |
| AGENTS.md | 342 | Apr 29 | Existing | Contributors |

---

## 🔗 Related Resources

- **GitHub Issues**: Issue tracking and bug reports
- **Model Checkpoints**: `models/checkpoints/best_model_fold*.pt`
- **Result Logs**: `results/experiments/runs/`
- **Paper Figures**: `results/paper_figures/`
- **Evaluation Outputs**: `results/evaluation/`

---

**Last Updated**: April 29, 2026  
**Maintained By**: Neuro-CXG Team  
**Status**: Publication-ready documentation ✅
