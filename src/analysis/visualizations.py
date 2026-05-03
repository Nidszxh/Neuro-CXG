import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Setup logging early for import-time warnings
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    ALL_FEATURE_NAMES,
    GNN_IN_CHANNELS,
    GNN_HIDDEN_CHANNELS,
    GNN_NUM_HEADS,
    GNN_NUM_LAYERS,
    GNN_POOLING,
    GNN_USE_SITE_EMBEDDING,
    GNN_SITE_EMBEDDING_DIM,
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_GRL_ALPHA,
    GNN_EDGE_GATE,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    NUM_TEMPORAL_FEATURES,
    RESULTS_DIR,
    get_active_checkpoint_dir,
)
from src.core.plotting import ColorPalette, FigureSize, apply_publication_style

palette = ColorPalette()
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import build_model
from src.models.training_utils import make_loader, attach_feature_scaler_from_checkpoint

# Import analysis modules
try:
    from src.analysis.feature_attribution import FeatureAttributionAnalyzer
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logger.warning("Feature attribution module not available")

try:
    from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False
    logger.warning("Diagnostics module (TrainingMonitor/CausalGraphAnalyzer) not available")


def create_feature_names():
    """Create list of temporal + spatial feature names."""
    return ALL_FEATURE_NAMES.copy()


def run_visualization_pipeline(output_dir: Path):
    """Run complete visualization pipeline."""
    logger.info("=" * 60)
    logger.info("NEURO-CXG VISUALIZATION PIPELINE")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    if CAPTUM_AVAILABLE:
        try:
            logger.info("Running advanced feature importance analysis...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint_path = get_active_checkpoint_dir() / "best_model_fold0.pt"
            if not checkpoint_path.exists():
                logger.warning("Checkpoint not found: %s", checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

                site_dim = GNN_SITE_EMBEDDING_DIM if GNN_USE_SITE_EMBEDDING else 0
                saved_in_features = state_dict["lin_in.weight"].shape[1]
                node_emb_dim = saved_in_features - GNN_IN_CHANNELS - site_dim
                model = build_model(
                    device=device,
                    use_grl=GNN_USE_GRL,
                    grl_alpha=GNN_GRL_ALPHA,
                    node_emb_dim=node_emb_dim,
                )

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(f"Checkpoint load had missing keys: {missing}")
                    logger.warning(f"Checkpoint load had unexpected keys: {unexpected}")
                attach_feature_scaler_from_checkpoint(model, checkpoint, expected_dim=GNN_IN_CHANNELS)
                model.eval()

                test_dataset = ABIDECausalDataset(split="test")
                test_loader = make_loader([d for d in test_dataset if d is not None], batch_size=32)

                feature_names = create_feature_names()

                analyzer = FeatureAttributionAnalyzer(
                    model=model,
                    test_loader=test_loader,
                    feature_names=feature_names,
                    device=device,
                )

                try:
                    attributions = analyzer.compute_attributions()
                    analyzer.visualize_feature_importance(attributions, output_dir / "feature_importance_ig.png")
                    analyzer.visualize_per_class(output_dir / "feature_importance_per_class.png")
                    logger.info("Advanced feature importance completed")
                except (RuntimeError, IndexError) as shape_error:
                    logger.warning(f"Feature attribution skipped due to architecture mismatch: {str(shape_error)[:80]}…")
        except Exception as e:
            logger.error(f"Advanced feature importance failed: {e}")
            import traceback

            traceback.print_exc()

    if DIAGNOSTICS_AVAILABLE:
        try:
            logger.info("Generating training history visualizations...")

            training_results_dir = RESULTS_DIR / "experiments" / "training"
            history_files = list(training_results_dir.glob("training_history_fold*.json"))

            if history_files:
                monitor = TrainingMonitor(output_dir=training_results_dir, num_folds=5)

                import json

                for history_file in history_files:
                    fold_id = _parse_fold_id(history_file)
                    with open(history_file, "r") as f:
                        history_data = json.load(f)
                        for key in history_data:
                            if key in monitor.fold_histories[fold_id]:
                                monitor.fold_histories[fold_id][key] = history_data[key]

                for fold_id, history in monitor.fold_histories.items():
                    if history["train_loss"]:
                        monitor.plot_training_curves(fold_id)

                monitor.plot_fold_comparison()

                logger.info("Training history visualizations completed")
            else:
                logger.warning("No training history files found. Run training with monitoring enabled.")

        except Exception as e:
            logger.error(f"Training history visualization failed: {e}")
            import traceback

            traceback.print_exc()

    if DIAGNOSTICS_AVAILABLE:
        try:
            logger.info("Running graph topology analysis...")

            manifest = pd.read_csv(MASTER_MANIFEST)

            analyzer = CausalGraphAnalyzer(graphs_dir=CAUSAL_GRAPHS_DIR, manifest=manifest)

            properties_df = analyzer.compute_graph_properties(max_graphs=500)
            properties_df.to_csv(output_dir / "graph_properties.csv", index=False)

            analyzer.visualize_average_causal_graph(output_path=output_dir / "average_causal_graph.png")

            logger.info("Graph analysis completed")
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            import traceback

            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("VISUALIZATION PIPELINE COMPLETE")
    logger.info(f"All visualizations saved to: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualizations for Neuro-CXG results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "visualizations",
        help="Directory to save visualizations (default: results/visualizations/)",
    )

    args = parser.parse_args()

    run_visualization_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
