import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import LOBE_NAMES, NUM_LOBES, NUM_SPATIAL_FEATURES, NUM_TEMPORAL_FEATURES

# Captum for interpretability
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    logging.warning("Captum not installed. Install with: pip install captum")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAttributionAnalyzer:
    """
    Analyze which node features drive GNN predictions.

    Uses Integrated Gradients for cleaner attributions than raw gradients.
    Provides both global (dataset-level) and local (subject-level) explanations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        feature_names: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize analyzer.

        Args:
            model: Trained GNN model
            test_loader: DataLoader with test data
            feature_names: List of feature names (length must match model input features)
            device: Device to run analysis on
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum required. Install with: pip install captum")

        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.feature_names = feature_names
        self.device = device

        if len(feature_names) == 0:
            raise ValueError("feature_names must be non-empty")

        self.feature_dim = len(feature_names)

        logger.info("FeatureAttributionAnalyzer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Features: {len(feature_names)}")

    def _get_wrapper_for_batch(self, edge_index, edge_attr, batch, age=None, sex=None, fiq=None):
        """
        Create a forward function for a specific batch.

        Pass demographics to maintain correct classifier input dimensions (128 pooled + 3 demo = 131).
        """

        def wrapper(x):
            return self.model(x, edge_index, edge_attr, batch, None, age, sex, fiq)

        return wrapper

    def compute_attributions(
        self,
        n_steps: int = 50,
        target_class: Optional[int] = None,
        debug: bool = False,
        use_integrated_gradients: bool = False,
    ) -> np.ndarray:
        """
        Compute feature attributions across test set.

        Uses gradient-based saliency (fast) by default, or Integrated Gradients if requested.
        Note: Integrated Gradients has issues with graph data due to batch tensor expansion.

        Args:
            n_steps: Number of steps for Integrated Gradients (only used if use_integrated_gradients=True)
            target_class: Class to compute attributions for (None = predicted class)
            debug: If True, print full exceptions
            use_integrated_gradients: If True, use slower but more accurate IG method

        Returns:
            attributions: (num_samples, NUM_LOBES regions, feature_dim features) array
        """
        logger.info("Computing feature attributions...")
        logger.info(
            f"  Method: {'Integrated Gradients' if use_integrated_gradients else 'Gradient-based Saliency'}"
        )

        all_attributions = []
        all_labels = []
        all_predictions = []
        failed_count = 0

        for batch_idx, data in enumerate(tqdm(self.test_loader, desc="Computing attributions")):
            if data is None:
                continue

            data = data.to(self.device)

            edge_index = data.edge_index
            edge_attr = data.edge_attr
            batch_tensor = (
                data.batch
                if hasattr(data, "batch") and data.batch is not None
                else torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            )

            with torch.no_grad():
                out = self.model(
                    data.x,
                    edge_index,
                    edge_attr,
                    batch_tensor,
                    site_id=None,
                    age=data.age if hasattr(data, "age") else None,
                    sex=data.sex if hasattr(data, "sex") else None,
                    fiq=data.fiq if hasattr(data, "fiq") else None,
                )
                pred_class = out.argmax(dim=1)

            if target_class is not None:
                target = target_class
            else:
                target = pred_class

            try:
                if use_integrated_gradients and CAPTUM_AVAILABLE:
                    baseline = torch.zeros_like(data.x, requires_grad=False)
                    input_features = data.x.clone().detach().requires_grad_(True)
                    age = data.age if hasattr(data, "age") else None
                    sex = data.sex if hasattr(data, "sex") else None
                    fiq = data.fiq if hasattr(data, "fiq") else None
                    batch_wrapper = self._get_wrapper_for_batch(edge_index, edge_attr, batch_tensor, age, sex, fiq)
                    ig = IntegratedGradients(batch_wrapper)
                    attr = ig.attribute(input_features, baselines=baseline, target=target, n_steps=n_steps)
                else:
                    input_features = data.x.clone().detach().requires_grad_(True)
                    out = self.model(
                        input_features,
                        edge_index,
                        edge_attr,
                        batch_tensor,
                        site_id=None,
                        age=data.age if hasattr(data, "age") else None,
                        sex=data.sex if hasattr(data, "sex") else None,
                        fiq=data.fiq if hasattr(data, "fiq") else None,
                    )
                    if isinstance(target, torch.Tensor):
                        target = target.to(out.device)
                        target_idx = torch.arange(out.shape[0], device=out.device)
                        loss = out[target_idx, target].sum()
                    else:
                        loss = out[0, target]
                    loss.backward()
                    attr = input_features.grad.abs()

                num_graphs = batch_tensor.max().item() + 1 if batch_tensor.max() >= 0 else 1
                attr_reshaped = attr.reshape(num_graphs, NUM_LOBES, self.feature_dim)

                all_attributions.append(attr_reshaped.cpu().detach().numpy())
                all_labels.append(data.y.cpu().numpy())
                all_predictions.append(pred_class.cpu().numpy())

            except Exception as e:
                failed_count += 1
                logger.warning(
                    f"Attribution failed for batch {batch_idx}: {type(e).__name__}: {str(e)[:200]}"
                )
                if failed_count == 1 and debug:
                    logger.error("First attribution error details:")
                    import traceback

                    traceback.print_exc()
                continue

        logger.info(f"Successfully computed {len(all_attributions)} batch attributions, {failed_count} failed")

        if len(all_attributions) == 0:
            logger.error("No attributions computed. Check model forward pass and data format.")
            raise ValueError(
                "No attributions could be computed for any batch in the test set. "
                "This may indicate that the model forward pass is failing."
            )

        attributions = np.concatenate(all_attributions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)

        logger.info(f"Computed attributions for {len(attributions)} samples")

        self.attributions = attributions
        self.labels = labels
        self.predictions = predictions

        return attributions

    def visualize_feature_importance(
        self,
        attributions: np.ndarray,
        output_path: Path,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Create heatmap showing which features matter for which brain regions.

        Args:
            attributions: (num_samples, 12 regions, 14 features) array
            output_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating feature importance heatmap...")

        mean_attr = np.abs(attributions).mean(axis=0)
        region_names = [LOBE_NAMES[i] for i in range(NUM_LOBES)]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            mean_attr.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap="RdYlBu_r",
            center=0,
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Attribution Magnitude"},
            linewidths=0.5,
            ax=ax,
        )

        ax.set_title("Feature Importance by Brain Lobe (Integrated Gradients)", fontsize=16, pad=20, fontweight="bold")
        ax.set_xlabel("Brain Lobe", fontsize=14, fontweight="bold")
        ax.set_ylabel("Node Feature", fontsize=14, fontweight="bold")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Heatmap saved to {output_path}")
        return mean_attr

    def compare_temporal_vs_spatial(
        self,
        attributions: np.ndarray,
        temporal_indices: Optional[List[int]] = None,
        spatial_indices: Optional[List[int]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Compare how much temporal vs spatial features contribute.

        Returns:
            Dict with temporal and spatial contribution percentages
        """
        mean_attr = np.abs(attributions).mean(axis=0)

        if temporal_indices is None:
            temporal_indices = list(range(NUM_TEMPORAL_FEATURES))
        if spatial_indices is None:
            spatial_indices = list(range(NUM_TEMPORAL_FEATURES, NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES))

        temporal_contrib = mean_attr[:, temporal_indices].mean()
        spatial_contrib = mean_attr[:, spatial_indices].mean()

        total = temporal_contrib + spatial_contrib
        temporal_pct = temporal_contrib / total * 100 if total > 0 else 0
        spatial_pct = spatial_contrib / total * 100 if total > 0 else 0

        if output_path is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(["Temporal", "Spatial"], [temporal_pct, spatial_pct], color=["#3498db", "#e74c3c"])
            ax.set_ylabel("Contribution (%)")
            ax.set_title("Temporal vs Spatial Feature Contribution")
            for i, v in enumerate([temporal_pct, spatial_pct]):
                ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

        return {"temporal_pct": temporal_pct, "spatial_pct": spatial_pct}

    def visualize_per_class(
        self,
        output_path: Path,
        figsize: Tuple[int, int] = (14, 10),
    ):
        """
        Create per-class feature importance heatmaps (ASD vs Control).
        """
        if not hasattr(self, "attributions") or not hasattr(self, "labels"):
            raise ValueError("Run compute_attributions() before visualize_per_class")

        labels = self.labels
        attributions = self.attributions

        asd_attr = attributions[labels == 1]
        control_attr = attributions[labels == 0]

        if len(asd_attr) == 0 or len(control_attr) == 0:
            logger.warning("Not enough samples for per-class comparison")
            return

        asd_mean = np.abs(asd_attr).mean(axis=0)
        control_mean = np.abs(control_attr).mean(axis=0)

        region_names = [LOBE_NAMES[i] for i in range(NUM_LOBES)]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(
            asd_mean.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap="Reds",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "ASD Attribution"},
            linewidths=0.5,
            ax=axes[0],
        )
        axes[0].set_title("ASD Feature Importance", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Brain Lobe", fontsize=12)
        axes[0].set_ylabel("Node Feature", fontsize=12)

        sns.heatmap(
            control_mean.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap="Blues",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Control Attribution"},
            linewidths=0.5,
            ax=axes[1],
        )
        axes[1].set_title("Control Feature Importance", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Brain Lobe", fontsize=12)
        axes[1].set_ylabel("Node Feature", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Per-class heatmaps saved to {output_path}")


if __name__ == "__main__":
    logger.info("Feature attribution module is intended to be imported.")
