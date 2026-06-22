# Neuro-CXG Documentation

## ⚠️ Single Source of Truth

**All performance metrics are in `docs/paper/results.md`** — no other document should contain numerical results. All other docs should reference this file.

---

## Directory Structure

### `docs/paper/` — Publication Documents

| File | Purpose |
|------|---------|
| **[results.md](./paper/results.md)** | **SINGLE SOURCE OF TRUTH** — all metrics, CIs, ablations |
| [methods.md](./paper/methods.md) | Methodology, preregistration, CV-Test gap explanation |
| [figures.md](./paper/figures.md) | Figure generation guide |

### `docs/` — Developer & Reference Documents

| File | Purpose |
|------|---------|
| [architecture.md](./architecture.md) | System design, stage registry, data flow |
| [configuration.md](./configuration.md) | All config parameters reference |
| [decisions.md](./decisions.md) | Design decision log (DD-NNN series) |
| [test_set_protocol.md](./test_set_protocol.md) | Test set evaluation history |
| [operations.md](./operations.md) | Pipeline operations guide |
| [data.md](./data.md) | Data processing documentation |
| [literature.md](./literature.md) | Prior work and citations |

---

## Quick Links

- **Canonical Results**: [docs/paper/results.md](./paper/results.md)
- **Quick Start**: [README.md](../README.md)
- **Run Pipeline**: `python src/run_pipeline.py --auto`
- **Run Evaluation**: `python src/run_evaluation.py`
