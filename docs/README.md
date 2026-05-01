# Documentation Index

Quick reference for navigating Neuro-CXG documentation. All files are publication-ready unless marked `[DRAFT]`.

---

## Quick Navigation

| What you need | Start here |
|--------------|------------|
| How to run the pipeline | `setup.md` |
| What each pipeline stage does | `architecture.md` |
| All configurable parameters | `configuration.md` |
| Why we made certain design choices | `decisions.md` |
| Current performance metrics | `evaluation.md` |
| Model limitations & known issues | `model_card.md` |
| Test set evaluation history | `test_set_protocol.md` |
| How to add new features | `extending.md` |
| Data schema & exclusions | `data.md` |
| Troubleshooting & failures | `operations.md` |
| Publication figures & changelog | `paper.md` |
| Comparison to prior work | `literature.md` |

---

## File Guide

### `architecture.md`
**Purpose**: Pipeline design, model architecture, training procedure
- Pipeline stage registry (29 stages, execution order)
- Model components: GATv2, GRL, AnatomicalHierarchyPool
- Data contracts per stage (input → output shapes)
- Mermaid flowchart of pipeline flow

**Audience**: Developers, new team members

---

### `setup.md`
**Purpose**: Installation, CLI usage, workflow walkthrough
- Environment setup & validation
- CLI flags table (`--auto`, `--skip-*`, `--multiview`, etc.)
- Typical workflows (full rebuild, iterate on features, etc.)
- ABIDE data acquisition methods

**Audience**: New users, reproduction

---

### `configuration.md`
**Purpose**: All tunable hyperparameters
- Parameter tables grouped by category: causal graph, GNN, site bias, loss, thresholds
- Config topology (which module defines what)
- Safe change workflow
- ⚠️ CRITICAL warnings for risky changes

**Audience**: Tuning experiments, reproducibility

---

### `data.md`
**Purpose**: Data schema, curation, features, reproducibility
- Feature artifact schema (24 features: temporal + frequency + spatial + internal)
- Exclusion criteria (1015 subjects after curation)
- Quality gates & validation
- Reproducibility notes (seed=42 everywhere)

**Audience**: Data scientists, audit compliance

---

### `decisions.md`
**Purpose**: Design decision log (DD-NNN format)
- Active decisions with rationale, rejected alternatives, trade-offs
- Status tracking (proposed → approved → implemented)
- Causality method rationale (Granger vs Pearson)
- Terminology policy

**Audience**: Architecture discussions, onboarding

---

### `evaluation.md`
**Purpose**: Results, ablations, statistics
- **§1** Canonical results (Test AUC 0.8651, CV AUC 0.8101 ± 0.0274)
- **§2** Ablation studies (A-E, D2)
- **§3** Graph topology analysis (ASD vs Control)
- **§4** Cross-site generalization (15/16 sites pass)
- **§5** Historical performance timeline
- **§6** Evaluation protocol
- **§7** Publication recommendations

**Audience**: Paper writing, results reporting

---

### `model_card.md`
**Purpose**: Model specifications, limitations
- Intended use & out-of-scope
- Training data (ABIDE-I, n=1015)
- Performance metrics (AUC, F1, Accuracy, Sensitivity, Specificity)
- Per-site performance table
- Known limitations (1 site fails, brainstem synthetic fallback)
- Hyperparameters

**Audience**: Model users, reviewers

---

### `test_set_protocol.md`
**Purpose**: Test set usage transparency
- Timeline of all test set evaluations (4 total)
- Model selection integrity assessment
- Canonical metric for publication
- Reviewer talking points
- CRITICAL DISCLOSURE language

**Audience**: Peer reviewers, reproducibility auditors

---

### `extending.md`
**Purpose**: How to add new capabilities safely
- Extension surface (new stages, features, models, metrics)
- Contracts to update
- Tests to add
- Backward compatibility notes

**Audience**: Developers adding features

---

### `operations.md`
**Purpose**: Failure modes, troubleshooting
- Quick triage order
- Performance benchmarks
- Known failure modes (with triggers, recovery, status)
- Post-fix audit checks

**Audience**: DevOps, debugging

---

### `paper.md`
**Purpose**: Publication figures & changelog
- Figure table (6 figures, generation scripts)
- Abridged changelog (durable architectural facts only)
- Key config changes over time

**Audience**: Paper writing

---

### `literature.md`
**Purpose**: Comparison to prior work
- ABIDE-I baseline comparison table
- Per-method analysis (Heinsfeld, Kawahara, etc.)
- Neuro-CXG's +23.5% improvement over best baseline
- Shuffled edges analysis

**Audience**: Literature review, related work

---

## Key Metrics (May 1, 2026 — Canonical)

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.8651 |
| **Test F1** | 0.7651 |
| **CV AUC** | 0.8101 ± 0.0274 |
| **Test Accuracy** | 77.27% |
| **Site Robustness** | 15/16 sites pass |

See `evaluation.md` §1 for full details.

---

## Version Info

| File | Last Updated |
|------|-------------|
| `test_set_protocol.md` | May 1, 2026 |
| `literature.md` | May 1, 2026 |
| `evaluation.md` | May 1, 2026 |
| `model_card.md` | April 30, 2026 |
| `architecture.md` | April 30, 2026 |
| `decisions.md` | April 30, 2026 |

---

*This index helps you find the right doc. For full context, start with `architecture.md` or `setup.md`.*