"""Declarative stage registry for the Neuro-CXG pipeline runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.core.config import (
    ATLAS_METADATA,
    CAUSAL_GRAPHS_DIR,
    CAUSAL_GRAPHS_MULTIVIEW_DIR,
    CHECKPOINT_DIR,
    DATA_METADATA,
    FINAL_TRAIN,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    RESULTS_ABLATIONS_DIR,
    RESULTS_DATA_QUALITY_DIR,
    RESULTS_DIR,
    RESULTS_EVALUATION_DIR,
    YOLO_WEIGHTS_PATH,
)


@dataclass(frozen=True)
class Stage:
    """Static metadata for a single pipeline stage."""

    key: str
    name: str
    module: str
    output_sentinel: Optional[Path]
    dependencies: List[str] = field(default_factory=list)
    function: Optional[str] = None
    args: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """Return True when this stage appears complete from filesystem artifacts."""
        if self.output_sentinel is None:
            return False

        sentinel = self.output_sentinel
        if not sentinel.exists():
            return False

        if sentinel.is_dir():
            return any(sentinel.iterdir())

        return sentinel.exists()


STAGES: List[Stage] = [
    Stage("download", "ABIDE Download", "src.data.abide_download", DATA_METADATA / "download_log.csv"),
    Stage("split", "Train/Val/Test Split", "src.data.split", MASTER_MANIFEST, dependencies=["download"]),
    Stage("manifest", "Generate Master Manifest", "src.utils.manifestor", MASTER_MANIFEST, dependencies=["split"]),
    # Task 5 (DD-013): Opt-in site-stratified CV fold regeneration.
    # Run AFTER split and BEFORE harmonization/gnn_training.
    #   python src/run_pipeline.py --site-stratified-cv
    Stage(
        "site_stratified_cv",
        "Site-Stratified CV Fold Assignment",
        "src.data.split",
        MASTER_MANIFEST,
        dependencies=["split"],
        function="run_site_stratified_split",
        args=["--site-stratified-cv"],
    ),
    Stage("atlas_validation", "Atlas Validation", "src.validation.atlas_validator", ATLAS_METADATA),
    Stage("pipeline_validation", "Pipeline Validation", "src.validation.pipeline_checks", None),
    Stage(
        "post_download_integrity",
        "Post-Download Integrity",
        "src.validation.pipeline_checks",
        None,
        function="check_dataset_integrity",
    ),
    Stage("annotate", "Atlas-Based Label Annotation", "src.pipelines.generate_labels", FINAL_TRAIN / "labels", dependencies=["split"]),
    Stage("yolo", "YOLO Training", "src.pipelines.roi_detection", YOLO_WEIGHTS_PATH, dependencies=["annotate"]),
    Stage("spatial_features", "Spatial Feature Extraction", "src.features.extract_spatial", NODE_FEATURES_3D, dependencies=["split"]),
    Stage(
        "temporal_features",
        "Temporal Feature Extraction",
        "src.features.extract_temporal",
        NODE_ATTRIBUTES_TEMPORAL,
        dependencies=["split"],
        args=["--n-jobs", "-1"],
    ),
    Stage("harmonization", "Feature Harmonization", "src.features.fold_safe_harmonization", NODE_ATTRIBUTES_HARMONIZED, dependencies=["spatial_features", "temporal_features"]),
    Stage(
        "pre_gnn_integrity",
        "Pre-GNN Integrity",
        "src.validation.pipeline_checks",
        None,
        dependencies=["harmonization"],
        function="check_distribution",
    ),
    Stage("causal_graphs", "Causal Graph Construction", "src.features.construct_causal", CAUSAL_GRAPHS_DIR, dependencies=["harmonization"], function="main", args=["--n-jobs", "-1"]),
    # Task 2 (DD-010): Opt-in multi-view causal graph construction.
    # Run AFTER causal_graphs.  Activates CausalInvarianceLoss during gnn_training
    # when CAUSAL_GRAPHS_MULTIVIEW_DIR is populated.
    #   python src/run_pipeline.py --multiview
    Stage(
        "multiview_graphs",
        "Multi-View Causal Graph Construction",
        "src.features.construct_causal",
        CAUSAL_GRAPHS_MULTIVIEW_DIR,
        dependencies=["causal_graphs"],
        function="main_multiview",
        args=["--multiview"],
    ),
    Stage(
        "dead_lobe_diagnosis",
        "Dead-Lobe Diagnosis",
        "src.analysis.diagnose_dead_lobes",
        None,
        dependencies=["split"],
        args=["--split", "train"],
    ),
    Stage(
        "diagnostics",
        "Pipeline Diagnostics",
        "src.validation.pipeline_checks",
        None,
        dependencies=["causal_graphs"],
        function="generate_health_report",
    ),
    Stage(
        "quality_validation",
        "Quality Validation",
        "src.validation.pipeline_checks",
        None,
        dependencies=["causal_graphs"],
        function="run_quality_validation",
    ),
    Stage("gnn_training", "GNN Training", "src.models.gnn_model", CHECKPOINT_DIR, dependencies=["causal_graphs"]),
    Stage("visualizations", "Generate Visualizations", "src.analysis.visualizations", RESULTS_DIR / "visualizations", dependencies=["gnn_training"]),
    Stage(
        "graph_visualization",
        "Causal Graph Visualization",
        "src.analysis.visualize_causal_graph",
        RESULTS_DIR / "visualizations" / "causal_graph_comparison.png",
        dependencies=["causal_graphs"],
        args=["--auto-pair"],
    ),
    Stage("evaluation", "Comprehensive Evaluation", "src.run_evaluation", RESULTS_EVALUATION_DIR / "comprehensive_results.json", dependencies=["gnn_training"]),
    Stage("explainability", "Explainability", "src.run_explainability", RESULTS_DIR / "explainability" / "summary.json", dependencies=["gnn_training"]),
    Stage("result_analysis", "Result Analysis", "src.run_result_analysis", RESULTS_DIR / "analysis" / "result_analysis_summary.json", dependencies=["gnn_training"]),
    Stage("subject_analysis", "Subject Analysis", "src.analysis.subject_analysis", RESULTS_DIR / "subject_analysis", dependencies=["causal_graphs"]),
    Stage("audit_check", "Post-Fix Audit Check", "src.validation.audit_check", None),
    Stage("dev_audit", "Developer Audit", "src.validation.dev_audit", None),
    Stage("feature_diagnostics", "Feature Diagnostics", "src.validation.diagnose_features", None),
    Stage("data_quality_experiments", "Data Quality Experiments", "src.experiments.data_quality", RESULTS_DATA_QUALITY_DIR),
    Stage("ablation_studies", "Ablation Studies", "src.experiments.run_ablations", RESULTS_ABLATIONS_DIR),
]


def stage_map(stages: Iterable[Stage] = STAGES) -> Dict[str, Stage]:
    """Return stage metadata indexed by stage key."""
    return {stage.key: stage for stage in stages}


def completion_snapshot(stages: Iterable[Stage] = STAGES) -> Dict[str, bool]:
    """Return stage completion status keyed by stage key."""
    return {stage.key: stage.is_complete() for stage in stages}
