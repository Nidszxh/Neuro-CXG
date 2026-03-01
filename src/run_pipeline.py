#!/usr/bin/env python
"""
src/run_pipeline.py
Neuro-CXG Unified Pipeline Runner
==================================
Last Updated: March 1, 2026

Orchestrates the complete Neuro-CXG pipeline from data download to analysis:

Core Pipeline (15 stages):
  1. ABIDE Download - fMRI data + 7-slice ALFF export
  2. Train/Val/Test Split - 2D stratification (DX_GROUP + SITE_ID)
  3. Master Manifest - Subject-phenotype mapping
  4. Atlas Validation - Verify AAL3 atlas files
  5. Pipeline Validation - Pre-flight health check
  6. Post-Download Integrity - PNG/NPY validation
  7. Atlas-Based Annotation - Generate YOLO labels
  8. YOLO Training - 12-region ROI detection (YOLO26n)
  9. Spatial Features - 3D coordinate aggregation
 10. Temporal Features - 20 features/ROI (8 time + 12 frequency)
 11. Harmonization - Fold-safe neuroHarmonize (protects DX_GROUP)
 12. Pre-GNN Integrity - Validate dataset completeness
 13. Causal Graphs - Granger causality/lagged correlation (12×12)
 14. Diagnostics - Comprehensive health report
 15. Quality Validation - YOLO quality, graph sparsity checks

Main Training (Phase 3):
 16. GNN Training - 5-fold CV with GAT+GRL

Post-Training Analysis (Phases 8 & 9):
 17. Visualizations - Comprehensive plots and figures
 18. Evaluation - Bootstrap CI, permutation test, subgroups
 19. Explainability - Node/edge importance, feature attribution
 20. Result Analysis - Per-subject predictions, misclassification

Usage:
  # Interactive mode (default)
  python src/run_pipeline.py --interactive

  # Automatic mode
  python src/run_pipeline.py --auto

  # Skip data prep
  python src/run_pipeline.py --skip-download --skip-split

  # Run only analysis
  python src/run_pipeline.py --analysis-only

  # Full pipeline with all analysis
  python src/run_pipeline.py --auto

For detailed documentation, see:
  - .github/copilot-instructions.md (comprehensive guide)
  - docs/ROADMAP.md (project phases)
  - docs/DATAFLOW.md (data pipeline details)
"""
import argparse
import logging
import subprocess
import sys
import shutil
from pathlib import Path

# Setup Pathing
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import (
    validate_environment,
    PROJECT_ROOT,
    DATA_METADATA,
    FINAL_TRAIN,
    FINAL_VAL,
    FINAL_TEST,
    CAUSAL_GRAPHS_DIR,
    NODE_FEATURES_3D,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_ATTRIBUTES_HARMONIZED,
    CHECKPOINT_DIR,
    YOLO_WEIGHTS_PATH
)

# Standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PIPELINE] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")

# Path constants for new stages (some imported from config)
DOWNLOAD_LOG    = DATA_METADATA / "download_log.csv"
MASTER_MANIFEST = DATA_METADATA / "master_manifest.csv"
ATLAS_DIR       = PROJECT_ROOT  / "data" / "atlases"

def prompt_user(message, default=True):

    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"\n{message} {suffix}: ").strip().lower()
        
        if response == "":
            return default
        elif response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")

def clear_old_state():
    """
    Prevents 'Shape Mismatches' by clearing old 164/170 ROI data.
    This ensures the new 12-region architecture has a clean environment.
    """
    logger.info("Cleaning legacy pipeline state for 12-region alignment...")
    
    # Files to remove to force regeneration
    to_delete = [
        NODE_FEATURES_3D,
        NODE_ATTRIBUTES_TEMPORAL,
        NODE_ATTRIBUTES_HARMONIZED
    ]
    for f in to_delete:
        if f.exists():
            f.unlink()
            logger.debug(f"Removed stale metadata: {f.name}")

    # Remove old causal graphs (they are the wrong shape)
    if CAUSAL_GRAPHS_DIR.exists():
        shutil.rmtree(CAUSAL_GRAPHS_DIR)
        CAUSAL_GRAPHS_DIR.mkdir(parents=True)
        logger.info("Reset Causal Graph directory (cleared old 170x170 matrices)")

def run_module(module_path, args_list=None, description="", function_name=None):

    # Use the same Python executable that's running this script
    python_exe = sys.executable
    
    if function_name:
        # Call specific function within module; propagate False return as exit code 1
        cmd = [python_exe, "-c",
            f"import sys; from {module_path} import {function_name}; "
            f"result = {function_name}(); sys.exit(0 if result is not False else 1)"]
    else:
        # Run module as script
        cmd = [python_exe, "-m", module_path]
        if args_list:
            cmd.extend(args_list)
    
    log_msg = description if description else f"Module: {module_path}"
    logger.info(f"Running: {log_msg}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        logger.error(f"Module {module_path} failed with exit code {result.returncode}")
        sys.exit(1)
    
    logger.info(f"Completed: {log_msg}")

def check_download_status():
    """Check if ABIDE data has been downloaded."""
    if not DOWNLOAD_LOG.exists():
        logger.warning("Download log not found. Data may not be downloaded.")
        return False
    
    # Count downloaded subjects
    import pandas as pd
    try:
        log_df = pd.read_csv(DOWNLOAD_LOG)
        total = len(log_df)
        # Success includes both 'success' and 'skipped' (already downloaded)
        successful = len(log_df[log_df.get('status', '').str.lower().isin(['success', 'skipped'])])
        logger.info(f"Download status: {successful}/{total} subjects ready (downloaded/skipped)")
        return successful > 0
    except Exception as e:
        logger.warning(f"Could not parse download log: {e}")
        return False

def check_split_status():
    """Check if train/val/test splits exist."""
    splits_exist = all([
        FINAL_TRAIN.exists(),
        FINAL_VAL.exists(),
        FINAL_TEST.exists()
    ])
    
    if splits_exist:
        train_count = len(list((FINAL_TRAIN / "images").glob("*.png"))) if (FINAL_TRAIN / "images").exists() else 0
        val_count = len(list((FINAL_VAL / "images").glob("*.png"))) if (FINAL_VAL / "images").exists() else 0
        test_count = len(list((FINAL_TEST / "images").glob("*.png"))) if (FINAL_TEST / "images").exists() else 0
        logger.info(f"Split status: Train={train_count}, Val={val_count}, Test={test_count}")
        return True
    
    logger.warning("Train/val/test splits not found.")
    return False

def show_execution_plan(stages):
    """Display what will be executed."""
    print("\n" + "="*70)
    print("EXECUTION PLAN")
    print("="*70)
    for i, (stage_name, will_run, reason) in enumerate(stages, 1):
        status = "✓ WILL RUN" if will_run else "○ SKIP"
        print(f"{i}. {stage_name:40} {status:15} {reason}")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-CXG: 12-Region Causal GNN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for each stage)
  python run_pipeline.py --interactive

  # Automatic mode (run all missing stages)
  python run_pipeline.py --auto

  # Skip data preparation stages
  python run_pipeline.py --skip-download --skip-split

  # Force complete rebuild
  python run_pipeline.py --force-reset

  # Show execution plan without running
  python run_pipeline.py --dry-run

  # Run only post-training analysis
  python run_pipeline.py --analysis-only

  # Run only visualizations
  python run_pipeline.py --visualizations-only

  # Full pipeline with analysis
  python run_pipeline.py --auto

  # Skip slow analysis stages
  python run_pipeline.py --skip-evaluation --skip-explainability

  # Regenerate features without full reset
  python run_pipeline.py --regenerate-features
        """
    )
    
    # Execution modes
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Prompt user before each stage (default)")
    parser.add_argument("--auto", action="store_true",
                        help="Run all missing stages without prompts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show execution plan without running")
    
    # Stage control - Skip flags
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip ABIDE download (use existing data)")
    parser.add_argument("--skip-split", action="store_true",
                        help="Skip train/val/test split (use existing splits)")
    parser.add_argument("--skip-manifest", action="store_true",
                        help="Skip manifest generation")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="Skip atlas-based annotation (use existing labels)")
    parser.add_argument("--skip-yolo", action="store_true",
                        help="Skip YOLO training (use existing weights)")
    parser.add_argument("--skip-gnn", action="store_true",
                        help="Skip GNN training")
    parser.add_argument("--skip-integrity", action="store_true",
                        help="Skip all integrity checks")
    parser.add_argument("--skip-atlas-validation", action="store_true",
                        help="Skip atlas validation")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip validation checks")
    
    # Post-training analysis flags
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip generating visualizations after training")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip comprehensive evaluation (bootstrap CI, permutation test, subgroups)")
    parser.add_argument("--skip-explainability", action="store_true",
                        help="Skip explainability analysis (node/edge importance, feature attribution)")
    parser.add_argument("--skip-result-analysis", action="store_true",
                        help="Skip result analysis (per-subject predictions, misclassification analysis)")
    parser.add_argument("--visualizations-only", action="store_true",
                        help="Only run visualizations (skip all other stages)")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run post-training analysis stages (vis, eval, explain, results)")
    
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="Skip comprehensive health check")
    parser.add_argument("--skip-comprehensive-validation", action="store_true",
                        help="Skip comprehensive validation & tuning suite (YOLO quality, sparsity, stratification)")
    
    # Feature regeneration
    parser.add_argument("--regenerate-features", action="store_true",
                        help="Regenerate spatial/temporal features, harmonization, and graphs (keeps other data)")
    
    # Force/Diagnostics
    parser.add_argument("--force-reset", action="store_true",
                        help="Wipe all intermediate CSVs and Graphs")
    
    args = parser.parse_args()
    
    # Override interactive mode if --auto is set
    interactive = args.interactive and not args.auto
    
    print("\n" + "="*70)
    print("NEURO-CXG PIPELINE RUNNER")
    print("12-Region Causal GNN for fMRI Analysis")
    print("="*70)
    
    # STAGE 0: PRE-FLIGHT VALIDATION
    logger.info("\nStage 0: Pre-Flight Validation")
    
    if not validate_environment():
        logger.error("Environment validation failed. Check config.py and data paths.")
        sys.exit(1)
    
    logger.info("Environment validation passed")
    
    # DETERMINE STAGE EXECUTION BASED ON FLAGS
    
    yolo_weights = YOLO_WEIGHTS_PATH
    data_downloaded = check_download_status() if not args.skip_download else True
    data_split = check_split_status() if not args.skip_split else True
    
    # Build stages dictionary for better maintainability
    stages = {
        "download": {
            "name": "ABIDE Download",
            "should_run": not args.skip_download and not data_downloaded,
            "reason": "Download ABIDE fMRI data + 7-slice ALFF export (Stage 1)",
            "module": "src.data.abide_download",
            "function": None
        },
        "split": {
            "name": "Train/Val/Test Split (2D Stratified)",
            "should_run": not args.skip_split and not data_split,
            "reason": "2D stratification by DX_GROUP + SITE_ID (Stage 2)",
            "module": "src.data.split",
            "function": None
        },
        "manifest": {
            "name": "Generate Master Manifest",
            "should_run": not args.skip_manifest and (not MASTER_MANIFEST.exists() or args.force_reset),
            "reason": "Maps subjects to phenotypes (Stage 3)",
            "module": "src.utils.manifestor",
            "function": None
        },
        "atlas_validation": {
            "name": "Atlas Validation",
            "should_run": not args.skip_atlas_validation,
            "reason": "Verify AAL3 atlas files exist and are valid (Stage 4)",
            "module": "src.validation.atlas_validator",
            "function": None
        },
        "pipeline_validation": {
            "name": "Pipeline Validation (Comprehensive Health Check)",
            "should_run": not args.skip_validation,
            "reason": "Full pipeline health check (Stage 5)",
            "module": "src.validation.pipeline_checks",
            "function": None
        },
        "post_download_integrity": {
            "name": "Post-Download Integrity Check",
            "should_run": not args.skip_integrity and data_downloaded,
            "reason": "Validate PNG/NPY files after download (Stage 6)",
            "module": "src.validation.pipeline_checks",
            "function": "check_dataset_integrity"
        },
        "annotate": {
            "name": "Atlas-Based Label Annotation",
            # Run if labels are missing OR if split is about to happen (data_split
            # is evaluated before split runs, so we must also check if split will run).
            "should_run": not args.skip_annotate and (data_split or not args.skip_split),
            "reason": "Generate YOLO training labels from AAL3 atlas (Stage 7)",
            "module": "src.pipelines.generate_labels",
            "function": None
        },
        "yolo": {
            "name": "YOLO Training (ROI Detection)",
            "should_run": (not yolo_weights.exists() or args.force_reset) and not args.skip_yolo,
            "reason": "Train YOLO26n for 12-region detection (Stage 8)" if not yolo_weights.exists() else "Force retrain",
            "module": "src.pipelines.roi_detection",
            "function": None
        },
        "spatial_features": {
            "name": "Spatial Feature Extraction (12-region)",
            "should_run": not NODE_FEATURES_3D.exists() or args.force_reset or args.regenerate_features,
            "reason": "YOLO inference → 3D spatial coords aggregation (Stage 9)",
            "module": "src.features.extract_spatial",
            "function": None
        },
        "temporal_features": {
            "name": "Temporal Feature Extraction",
            "should_run": not NODE_ATTRIBUTES_TEMPORAL.exists() or args.force_reset or args.regenerate_features,
            "reason": "20 features per ROI: 8 time-domain + 12 frequency (Stage 10)",
            "module": "src.features.extract_temporal",
            "function": None,
            "args": ["--add-frequency"]  # Pass arguments to module
        },
        "harmonization": {
            "name": "Feature Harmonization",
            "should_run": not NODE_ATTRIBUTES_HARMONIZED.exists() or args.force_reset or args.regenerate_features,
            "reason": "Fold-safe neuroHarmonize, protects DX_GROUP (Stage 11)",
            "module": "src.features.fold_safe_harmonization",
            "function": None
        },
        "pre_gnn_integrity": {
            "name": "Pre-GNN Integrity Check",
            "should_run": not args.skip_integrity,
            "reason": "Validate dataset completeness per split (Stage 12)",
            "module": "src.validation.pipeline_checks",
            "function": "check_distribution"
        },
        "diagnostics": {
            "name": "Pipeline Diagnostics",
            "should_run": not args.skip_diagnostics,
            "reason": "Comprehensive health report after graphs built (Stage 13)",
            "module": "src.validation.pipeline_checks",
            "function": "generate_health_report"
        },
        "quality_validation": {
            "name": "Quality Validation (YOLO & Graph Sparsity)",
            "should_run": not args.skip_comprehensive_validation,
            "reason": "YOLO quality, graph sparsity, stratification (Stage 14)",
            "module": "src.validation.pipeline_checks",
            "function": "run_quality_validation"
        },
        "causal_graphs": {
            "name": "Causal Graph Construction (12×12)",
            "should_run": (not any(CAUSAL_GRAPHS_DIR.iterdir()) if CAUSAL_GRAPHS_DIR.exists() else True) or args.force_reset or args.regenerate_features,
            "reason": "Granger causality/lagged correlation graphs (Stage 15)",
            "module": "src.features.construct_causal",
            "function": None
        },
        "gnn_training": {
            "name": "GNN Training (5-Fold CV)",
            "should_run": not args.skip_gnn and not args.visualizations_only and not args.analysis_only,
            "reason": "Main training phase (Phase 3)",
            "module": "src.models.gnn_model",
            "function": None
        },
        "visualizations": {
            "name": "Generate Visualizations",
            "should_run": not args.skip_visualizations,  # Run by default unless skipped
            "reason": "Generate comprehensive visualizations (Phase 9 Reporting)",
            "module": "src.analysis.visualizations",
            "function": None
        },
        "evaluation": {
            "name": "Comprehensive Evaluation",
            "should_run": not args.skip_evaluation and (
                any(CHECKPOINT_DIR.glob("best_model_fold*.pt")) if CHECKPOINT_DIR.exists() else False
            ),
            "reason": "Ensemble evaluation, bootstrap CI, permutation test, subgroups (Phase 9.2)",
            "module": "src.run_evaluation",
            "function": None
        },
        "explainability": {
            "name": "Explainability Analysis",
            "should_run": not args.skip_explainability and (
                any(CHECKPOINT_DIR.glob("best_model_fold*.pt")) if CHECKPOINT_DIR.exists() else False
            ),
            "reason": "Node/edge importance, feature attribution, literature validation (Phase 8)",
            "module": "src.run_explainability",
            "function": None
        },
        "result_analysis": {
            "name": "Result Interpretation & Analysis",
            "should_run": not args.skip_result_analysis and (
                any(CHECKPOINT_DIR.glob("best_model_fold*.pt")) if CHECKPOINT_DIR.exists() else False
            ),
            "reason": "Per-subject predictions, misclassification analysis, site effects (Phase 9.3)",
            "module": "src.run_result_analysis",
            "function": None
        }
    }
    
    # Special handling for analysis-only mode
    if args.analysis_only:
        logger.info("📊 Analysis-only mode: Running post-training analysis stages only")
        analysis_stages = {"visualizations", "evaluation", "explainability", "result_analysis"}
        for key in stages.keys():
            if key not in analysis_stages:
                stages[key]["should_run"] = False
    
    # Special handling for visualizations-only mode
    if args.visualizations_only:
        logger.info("🎨 Visualizations-only mode: Skipping all training stages")
        for key in stages.keys():
            if key != "visualizations":
                stages[key]["should_run"] = False
    
    # Show execution plan
    stage_list = [(stage_info["name"], stage_info["should_run"], stage_info["reason"]) 
                  for stage_info in stages.values()]
    show_execution_plan(stage_list)
    
    if args.dry_run:
        logger.info("🔍 Dry-run mode: Exiting without execution")
        return
    
    # Reset state if requested
    if args.force_reset:
        if interactive and not prompt_user("⚠️  This will delete intermediate files. Continue?", default=False):
            logger.info("Aborted by user")
            sys.exit(0)
        clear_old_state()
    
    # STAGE EXECUTION
    # Execute stages in order
    # Core Pipeline (15 stages as per copilot-instructions.md):
    #   1-2:   download, split
    #   3-4:   manifest, atlas_validation
    #   5-6:   pipeline_validation, post_download_integrity
    #   7-9:   annotate, yolo, spatial_features
    #  10-12:  temporal_features, harmonization, pre_gnn_integrity
    #  13-15:  causal_graphs, diagnostics, quality_validation
    #     16:  gnn_training
    # Post-Training Analysis (Phases 8 & 9):
    #  17-20:  visualizations, evaluation, explainability, result_analysis
    
    for stage_key in ["download", "split", "manifest", "atlas_validation",
                      "pipeline_validation", "post_download_integrity", "annotate",
                      "yolo", "spatial_features", "temporal_features", "harmonization",
                      "pre_gnn_integrity",      # validate features BEFORE building graphs
                      "causal_graphs",           # build graphs
                      "diagnostics",             # health report (includes graph status)
                      "quality_validation",      # graph quality checks (needs graphs to exist)
                      "gnn_training",            # main training (Phase 3)
                      "visualizations",          # Phase 9 reporting
                      "evaluation",              # Phase 9.2 comprehensive evaluation
                      "explainability",          # Phase 8 explainability analysis
                      "result_analysis"]:        # Phase 9.3 result interpretation
        
        if stage_key not in stages:
            continue
        
        stage = stages[stage_key]
        
        if not stage["should_run"]:
            logger.info(f"⏭️  Skipping: {stage['name']}")
            continue
        
        # Special handling for long-running stages
        skip_prompt = False
        if stage_key == "yolo":
            msg = f"Run {stage['name']}? (This may take 1-2 hours)"
        elif stage_key == "download":
            msg = f"Run {stage['name']}? (This may take 2-4 hours)"
        elif stage_key == "gnn_training":
            msg = f"🚀 Start {stage['name']}? (Main training phase, ~20-30 min)"
        elif stage_key == "visualizations":
            msg = f"🎨 Generate {stage['name']}? (Creates plots and analysis)"
        elif stage_key == "evaluation":
            msg = f"📊 Run {stage['name']}? (Bootstrap CI, permutation test, subgroups)"
        elif stage_key == "explainability":
            msg = f"🔬 Run {stage['name']}? (Node/edge importance, feature attribution)"
        elif stage_key == "result_analysis":
            msg = f"📈 Run {stage['name']}? (Per-subject predictions, misclassification)"
        else:
            msg = f"Run {stage['name']}?"
        
        if interactive and not args.visualizations_only and not args.analysis_only:  # No prompt in special modes
            if not prompt_user(msg, default=True):
                logger.info(f"⏭️  User skipped: {stage['name']}")
                continue
        
        # Execute with function name if specified
        function_name = stage.get("function", None)
        args_list = stage.get("args", None)
        run_module(stage["module"], args_list=args_list, description=stage["name"], function_name=function_name)
    
    # COMPLETION
    
    logger.info("\n" + "="*70)
    logger.info("NEURO-CXG PIPELINE EXECUTION COMPLETE")
    logger.info("="*70)
    logger.info(f"📁 Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"📁 Causal graphs: {CAUSAL_GRAPHS_DIR}")
    logger.info(f"📁 Features: {DATA_METADATA}")
    
    # Check if analysis outputs exist
    from src.core.config import RESULTS_DIR
    if (RESULTS_DIR / "visualizations").exists():
        logger.info(f"📁 Visualizations: {RESULTS_DIR / 'visualizations'}")
    if (RESULTS_DIR / "evaluation").exists():
        logger.info(f"📁 Evaluation results: {RESULTS_DIR / 'evaluation'}")
    if (RESULTS_DIR / "explainability").exists():
        logger.info(f"📁 Explainability: {RESULTS_DIR / 'explainability'}")
    if (RESULTS_DIR / "analysis").exists():
        logger.info(f"📁 Result analysis: {RESULTS_DIR / 'analysis'}")
    
    logger.info("="*70)
    logger.info("\n✨ Post-Training Analysis Commands:")
    logger.info("   python src/run_pipeline.py --visualizations-only")
    logger.info("   python src/run_pipeline.py --analysis-only")
    logger.info("   python src/run_evaluation.py")
    logger.info("   python src/run_explainability.py")
    logger.info("   python src/run_result_analysis.py")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()