# Documentation Index

## Start Here

1. [problem.md](problem.md) - Problem framing and project objective.
2. [setup.md](setup.md) - Environment setup and validation.
3. [usage.md](usage.md) - Primary command-line workflows.
4. [walkthrough.md](walkthrough.md) - End-to-end reproducible runbook.

## Maintained Core Docs

- [architecture.md](architecture.md) - Pipeline architecture, stage orchestration, and contracts.
- [components.md](components.md) - Module responsibilities and interfaces.
- [configuration.md](configuration.md) - Config modules, defaults, and high-impact knobs.
- [data.md](data.md) - Data artifacts, schemas, and quality checks.
- [evaluation.md](evaluation.md) - Evaluation, explainability, and result-analysis scripts.
- [decisions.md](decisions.md) - Active design decisions reflected in code.
- [performance.md](performance.md) - Runtime characteristics and performance interpretation.
- [failure-modes.md](failure-modes.md) - Operational failure signatures and recovery paths.
- [extending.md](extending.md) - Safe extension guide for stages/features/models.

## Additional Reference Docs

- [experiments.md](experiments.md) - Experiment and ablation notes.
- [results.md](results.md) - Result summaries and run reporting history.
- [training.md](training.md) - Training-focused implementation notes.
- [data-curation.md](data-curation.md) - Curation rationale and cohort filtering notes.
- [gpu-granger-testing.md](gpu-granger-testing.md) - GPU Granger test guidance.

## Draft/Internal Docs

- [structured-documentation-draft.md](structured-documentation-draft.md) - Internal draft used to assemble the structured documentation set.
- [DOCUMENTATION_AUDIT_REPORT.md](DOCUMENTATION_AUDIT_REPORT.md) - Prior audit snapshot.

## External Supplement

- [configs/README.md](../configs/README.md) - Configuration file notes under `configs/`.

## Maintenance Rules

- Treat code and config in `src/` as source of truth when docs drift.
- If output artifact names or paths change, update both docs and stage sentinels in `src/pipeline/registry.py`.
- Keep root [README.md](../README.md) and this index aligned.
- Update [CHANGELOG.md](../CHANGELOG.md) for user-visible documentation changes.
