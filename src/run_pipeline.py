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
DOWNLOAD_LOG = DATA_METADATA / "download_log.csv"
MASTER_MANIFEST = DATA_METADATA / "master_manifest.csv"
ATLAS_DIR = PROJECT_ROOT / "data" / "atlases"

def prompt_user(message, default=True):
    """
    Interactive yes/no prompt.
    
    Args:
        message: Question to ask user
        default: Default value if user just hits Enter
    
    Returns:
        bool: True for yes, False for no
    """
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

def run_module(module_path, args_list=None, description=""):
    """
    Executes a submodule as a separate process to avoid ArgParse conflicts.
    
    Args:
        module_path: Python module path (e.g., 'src.data.split')
        args_list: Optional command-line arguments
        description: Human-readable description for logging
    """
    # Use the same Python executable that's running this script
    python_exe = sys.executable
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
        successful = len(log_df[log_df.get('status', '') == 'success'])
        logger.info(f"Download status: {successful}/{total} subjects successful")
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
  python run_pipeline.py --interactive                    # Prompt for each stage (default)
  python run_pipeline.py --auto                           # Run all missing stages automatically
  python run_pipeline.py --skip-download --skip-split     # Skip data prep stages
  python run_pipeline.py --force-reset                    # Clean state and rebuild everything
  python run_pipeline.py --dry-run                        # Show what would run without executing
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
    
    # NEW: Visualization flags
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip generating visualizations after training")
    parser.add_argument("--visualizations-only", action="store_true",
                        help="Only run visualizations (skip all other stages)")
    
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="Skip comprehensive health check")
    parser.add_argument("--skip-comprehensive-validation", action="store_true",
                        help="Skip comprehensive validation & tuning suite (YOLO quality, sparsity, stratification)")
    
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
            "reason": "Missing ABIDE data",
            "module": "src.data.abide_download"
        },
        "split": {
            "name": "Train/Val/Test Split (2D Stratified)",
            "should_run": not args.skip_split and not data_split,
            "reason": "Missing train/val/test splits",
            "module": "src.data.split"
        },
        "manifest": {
            "name": "Generate Master Manifest",
            "should_run": not args.skip_manifest and (not MASTER_MANIFEST.exists() or args.force_reset),
            "reason": "Missing manifest" if not MASTER_MANIFEST.exists() else "Force reset",
            "module": "src.utils.manifestor"
        },
        "atlas_validation": {
            "name": "Atlas Validation",
            "should_run": not args.skip_atlas_validation,
            "reason": "Verify atlas files exist and are valid",
            "module": "src.validation.atlas_validator"
        },
        "validation": {
            "name": "Comprehensive Validation & Tuning",
            "should_run": not args.skip_validation,
            "reason": "Detailed quality checks (YOLO, graphs, features, stratification)",
            "module": "src.validation.validator"
        },
        "post_download_integrity": {
            "name": "Post-Download Integrity Check",
            "should_run": not args.skip_integrity and data_downloaded,
            "reason": "Validate downloaded images",
            "module": "src.validation.integrity",
            "function": "check_dataset_integrity"
        },
        "annotate": {
            "name": "Atlas-Based Label Annotation",
            "should_run": not args.skip_annotate and data_split,
            "reason": "Generate YOLO training labels",
            "module": "src.utils.annotate"
        },
        "yolo": {
            "name": "YOLO Training (ROI Detection)",
            "should_run": (not yolo_weights.exists() or args.force_reset) and not args.skip_yolo,
            "reason": "Missing weights" if not yolo_weights.exists() else "Force reset",
            "module": "src.pipelines.roi_detection"
        },
        "spatial_features": {
            "name": "Spatial Feature Extraction (12-region)",
            "should_run": not NODE_FEATURES_3D.exists() or args.force_reset,
            "reason": "Missing features" if not NODE_FEATURES_3D.exists() else "Force reset",
            "module": "src.features.extract_features"
        },
        "temporal_features": {
            "name": "Temporal Feature Extraction",
            "should_run": not NODE_ATTRIBUTES_TEMPORAL.exists() or args.force_reset,
            "reason": "Missing features" if not NODE_ATTRIBUTES_TEMPORAL.exists() else "Force reset",
            "module": "src.utils.compute_roi"
        },
        "harmonization": {
            "name": "Feature Harmonization",
            "should_run": not NODE_ATTRIBUTES_HARMONIZED.exists() or args.force_reset,
            "reason": "Missing harmonized data" if not NODE_ATTRIBUTES_HARMONIZED.exists() else "Force reset",
            "module": "src.features.safe_harmonization"
        },
        "pre_gnn_integrity": {
            "name": "Pre-GNN Integrity Check",
            "should_run": not args.skip_integrity,
            "reason": "Validate intermediate outputs",
            "module": "src.validation.integrity",
            "function": "check_distribution"
        },
        "diagnostics": {
            "name": "Pipeline Diagnostics",
            "should_run": not args.skip_diagnostics,
            "reason": "Comprehensive health check",
            "module": "src.validation.integrity",
            "function": "generate_health_report"
        },
        "comprehensive_validation": {
            "name": "Comprehensive Validation",
            "should_run": not args.skip_comprehensive_validation,
            "reason": "YOLO quality, sparsity, stratification checks",
            "module": "src.validation.validator"
        },
        "causal_graphs": {
            "name": "Causal Graph Construction (12×12)",
            "should_run": (not any(CAUSAL_GRAPHS_DIR.iterdir()) if CAUSAL_GRAPHS_DIR.exists() else True) or args.force_reset,
            "reason": "Missing graphs" if (not any(CAUSAL_GRAPHS_DIR.iterdir()) if CAUSAL_GRAPHS_DIR.exists() else True) else "Force reset",
            "module": "src.features.construct_causal"
        },
        "gnn_training": {
            "name": "GNN Training (5-Fold CV)",
            "should_run": not args.skip_gnn and not args.visualizations_only,  # Skip if only visualizations requested
            "reason": "Main training phase",
            "module": "src.models.gnn_model"
        },
        "visualizations": {
            "name": "Generate Visualizations",
            "should_run": not args.skip_visualizations,  # Run by default unless skipped
            "reason": "Generate comprehensive visualizations",
            "module": "src.analysis.visualize_results"
        }
    }
    
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
    
    for stage_key in ["download", "split", "manifest", "atlas_validation",
                      "post_download_integrity", "annotate", "yolo", "spatial_features",
                      "temporal_features", "harmonization", "diagnostics",
                      "comprehensive_validation", "pre_gnn_integrity",
                      "causal_graphs", "gnn_training", "visualizations"]:  # Added visualizations
        
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
            msg = f"🚀 Start {stage['name']}? (Main training phase)"
        elif stage_key == "visualizations":
            msg = f"🎨 Generate {stage['name']}? (Creates plots and analysis)"
        else:
            msg = f"Run {stage['name']}?"
        
        if interactive and not args.visualizations_only:  # No prompt in visualizations-only mode
            if not prompt_user(msg, default=True):
                logger.info(f"⏭️  User skipped: {stage['name']}")
                continue
        
        run_module(stage["module"], description=stage["name"])
    
    # COMPLETION
    
    logger.info("\n" + "="*70)
    logger.info("NEURO-CXG PIPELINE EXECUTION COMPLETE")
    logger.info("="*70)
    logger.info(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}")
    logger.info(f"📁 Causal graphs in: {CAUSAL_GRAPHS_DIR}")
    logger.info(f"📁 Features in: {DATA_METADATA}")
    logger.info("="*70 + "\n")

if __name__ == "__main__":
    main()