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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import LOBE_NAMES

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
    Analyze which of the 14 node features drive GNN predictions.
    
    Uses Integrated Gradients for cleaner attributions than raw gradients.
    Provides both global (dataset-level) and local (subject-level) explanations.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        feature_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize analyzer.
        
        Args:
            model: Trained GNN model
            test_loader: DataLoader with test data
            feature_names: List of 14 feature names (e.g., ['mean', 'std', ..., 'x', 'y', 'z'])
            device: Device to run analysis on
        """
        if not CAPTUM_AVAILABLE:
            raise ImportError("Captum required. Install with: pip install captum")
        
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.feature_names = feature_names
        self.device = device
        
        # Validate feature names
        if len(feature_names) != 14:
            raise ValueError(f"Expected 14 feature names, got {len(feature_names)}")
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(self._forward_wrapper)
        
        logger.info(f"FeatureAttributionAnalyzer initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Features: {len(feature_names)}")
    
    def _forward_wrapper(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for Captum compatibility.
        
        Captum needs a function that takes only input features and returns predictions.
        We need to reconstruct the full graph from stored edge_index/edge_attr/batch.
        """
        # Use stored graph structure from last forward pass
        return self.model(
            x, 
            self._current_edge_index,
            self._current_edge_attr,
            self._current_batch,
            self._current_site_id,
            self._current_age,
            self._current_sex,
            self._current_fiq
        )
    
    def compute_attributions(
        self,
        n_steps: int = 50,
        target_class: Optional[int] = None,
        debug: bool = False
    ) -> np.ndarray:
        """
        Compute feature attributions across test set.
        
        Args:
            n_steps: Number of steps for Integrated Gradients (more = more accurate, slower)
            target_class: Class to compute attributions for (None = predicted class)
            debug: If True, print full exceptions instead of warnings
        
        Returns:
            attributions: (num_samples, 12 regions, 14 features) array
        """
        logger.info("Computing feature attributions...")
        logger.info(f"  Integration steps: {n_steps}")
        
        all_attributions = []
        all_labels = []
        all_predictions = []
        failed_count = 0
        
        for batch_idx, data in enumerate(tqdm(self.test_loader, desc="Computing attributions")):
            if data is None:
                continue
            
            data = data.to(self.device)
            
            # Store graph structure for wrapper function
            self._current_edge_index = data.edge_index
            self._current_edge_attr = data.edge_attr
            self._current_batch = data.batch
            self._current_site_id = getattr(data, 'site_id', None)
            self._current_age = getattr(data, 'age', None)
            self._current_sex = getattr(data, 'sex', None)
            self._current_fiq = getattr(data, 'fiq', None)
            
            # Get predictions to determine target class
            with torch.no_grad():
                out = self.model(
                    data.x, data.edge_index, data.edge_attr, data.batch,
                    self._current_site_id, self._current_age,
                    self._current_sex, self._current_fiq
                )
                pred_class = out.argmax(dim=1)
            
            # Determine target for attribution
            if target_class is not None:
                target = target_class
            else:
                target = pred_class.item()
            
            # Create baseline (zero features)
            baseline = torch.zeros_like(data.x)
            
            # Compute attributions
            try:
                attr = self.ig.attribute(
                    data.x,
                    baselines=baseline,
                    target=target,
                    n_steps=n_steps
                )
                
                # Reshape to (batch_size, 12 regions, 14 features)
                # Assuming data.x is (num_nodes, 14) where num_nodes = batch_size * 12
                num_graphs = data.batch.max().item() + 1
                attr_reshaped = attr.reshape(num_graphs, 12, 14)
                
                all_attributions.append(attr_reshaped.cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
                all_predictions.append(pred_class.cpu().numpy())
                
            except Exception as e:
                failed_count += 1
                if debug and failed_count == 1:  # Print first error in detail
                    logger.error(f"Attribution failed for batch {batch_idx}: {type(e).__name__}: {e}")
                elif not debug:
                    logger.warning(f"Attribution failed for batch {batch_idx}: {type(e).__name__}")
                continue
        
        logger.info(f"Successfully computed {len(all_attributions)} batch attributions, {failed_count} failed")
        
        # Check if any attributions were computed
        if len(all_attributions) == 0:
            logger.error(f"✗ No attributions computed! All {failed_count} batches failed.")
            logger.error(f"  Check that model predictions are working and data has correct format.")
            raise ValueError("No attributions could be computed for any batch in the test set. "
                           "This may indicate that the model forward pass is failing. "
                           "Check the error messages above for details.")
        
        # Concatenate all results
        attributions = np.concatenate(all_attributions, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        
        logger.info(f"✓ Computed attributions for {len(attributions)} samples")
        
        # Store for later analysis
        self.attributions = attributions
        self.labels = labels
        self.predictions = predictions
        
        return attributions
    
    def visualize_feature_importance(
        self,
        attributions: np.ndarray,
        output_path: Path,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create heatmap showing which features matter for which brain regions.
        
        Args:
            attributions: (num_samples, 12 regions, 14 features) array
            output_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating feature importance heatmap...")
        
        # Average absolute attribution across all samples
        mean_attr = np.abs(attributions).mean(axis=0)  # (12 regions, 14 features)
        
        # Region names from config
        region_names = LOBE_NAMES
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            mean_attr.T,  # Transpose: features as rows, lobes as columns
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap='RdYlBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Attribution Magnitude'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title('Feature Importance by Brain Lobe (Integrated Gradients)', 
                     fontsize=16, pad=20, fontweight='bold')
        ax.set_xlabel('Brain Lobe', fontsize=14, fontweight='bold')
        ax.set_ylabel('Node Feature', fontsize=14, fontweight='bold')
        
        # Rotate labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Heatmap saved to {output_path}")
        
        return mean_attr
    
    def compare_temporal_vs_spatial(
        self,
        attributions: np.ndarray,
        temporal_indices: List[int] = list(range(8)),
        spatial_indices: List[int] = list(range(8, 14))
    ) -> Dict[str, float]:
        """
        Compare importance of temporal vs spatial features.
        
        Args:
            attributions: (num_samples, 12 regions, 14 features) array
            temporal_indices: Indices of temporal features (default: 0-7)
            spatial_indices: Indices of spatial features (default: 8-13)
        
        Returns:
            Dictionary with comparison statistics
        """
        logger.info("Comparing temporal vs spatial feature importance...")
        
        # Extract temporal and spatial attributions
        temporal_attr = attributions[:, :, temporal_indices]
        spatial_attr = attributions[:, :, spatial_indices]
        
        # Compute mean absolute importance
        temporal_importance = np.abs(temporal_attr).mean()
        spatial_importance = np.abs(spatial_attr).mean()
        
        # Per-sample importance for statistical testing
        temporal_per_sample = np.abs(temporal_attr).mean(axis=(1, 2))
        spatial_per_sample = np.abs(spatial_attr).mean(axis=(1, 2))
        
        # Statistical test: paired t-test
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(temporal_per_sample, spatial_per_sample)
        
        results = {
            'temporal_mean': temporal_importance,
            'spatial_mean': spatial_importance,
            'ratio': temporal_importance / spatial_importance if spatial_importance > 0 else float('inf'),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        # Print results
        print("\n" + "="*70)
        print("TEMPORAL vs SPATIAL FEATURE IMPORTANCE")
        print("="*70)
        print(f"Temporal features (mean attribution): {temporal_importance:.4f}")
        print(f"Spatial features (mean attribution):  {spatial_importance:.4f}")
        print(f"Ratio (Temporal/Spatial):             {results['ratio']:.2f}")
        print(f"\nStatistical Test (Paired t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value:     {p_value:.4f}")
        
        if p_value < 0.05:
            winner = "Temporal" if temporal_importance > spatial_importance else "Spatial"
            print(f"  ✓ {winner} features are significantly more important")
        else:
            print(f"  → No significant difference (both contribute)")
        print("="*70 + "\n")
        
        return results
    
    def analyze_per_lobe(
        self,
        attributions: np.ndarray,
        output_dir: Path
    ) -> pd.DataFrame:
        """
        Analyze feature importance separately for each region.
        
        Args:
            attributions: (num_samples, 12 regions, 14 features) array
            output_dir: Directory to save per-region analysis
        
        Returns:
            DataFrame with per-region feature rankings
        """
        logger.info("Analyzing feature importance per region...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        region_names = LOBE_NAMES
        
        results = []
        
        for lobe_idx, lobe_name in enumerate(region_names):
            # Extract attributions for this lobe
            lobe_attr = attributions[:, lobe_idx, :]  # (num_samples, 14 features)
            
            # Compute mean absolute attribution per feature
            feature_importance = np.abs(lobe_attr).mean(axis=0)
            
            # Create DataFrame
            lobe_df = pd.DataFrame({
                'lobe': lobe_name,
                'feature': self.feature_names,
                'importance': feature_importance
            })
            
            # Sort by importance
            lobe_df = lobe_df.sort_values('importance', ascending=False)
            
            results.append(lobe_df)
            
            # Save individual lobe analysis
            lobe_df.to_csv(output_dir / f'{lobe_name.lower()}_feature_importance.csv', index=False)
            
            # Print top 5 features for this lobe
            print(f"\n{lobe_name} Lobe - Top 5 Features:")
            print(lobe_df.head(5).to_string(index=False))
        
        # Combine all results
        all_results = pd.concat(results, ignore_index=True)
        all_results.to_csv(output_dir / 'all_lobes_feature_importance.csv', index=False)
        
        logger.info(f"✓ Per-lobe analysis saved to {output_dir}")
        
        return all_results
    
    def visualize_per_class(
        self,
        attributions: np.ndarray,
        labels: np.ndarray,
        output_path: Path,
        figsize: Tuple[int, int] = (16, 6)
    ):
        """
        Compare feature importance between ASD and Control groups.
        
        Args:
            attributions: (num_samples, 12 regions, 14 features) array
            labels: (num_samples,) array with true labels
            output_path: Path to save figure
            figsize: Figure size
        """
        logger.info("Creating per-class feature importance comparison...")
        
        # Separate by class
        asd_mask = labels == 1
        control_mask = labels == 0
        
        asd_attr = np.abs(attributions[asd_mask]).mean(axis=0)  # (12, 14)
        control_attr = np.abs(attributions[control_mask]).mean(axis=0)  # (12, 14)
        
        # Create side-by-side heatmaps
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        region_names = LOBE_NAMES
        
        # ASD heatmap
        sns.heatmap(
            asd_attr.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap='RdYlBu_r',
            annot=True,
            fmt='.3f',
            ax=axes[0],
            cbar_kws={'label': 'Attribution'}
        )
        axes[0].set_title('ASD Feature Importance', fontweight='bold')
        axes[0].set_xlabel('Brain Lobe')
        axes[0].set_ylabel('Feature')
        
        # Control heatmap
        sns.heatmap(
            control_attr.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap='RdYlBu_r',
            annot=True,
            fmt='.3f',
            ax=axes[1],
            cbar_kws={'label': 'Attribution'}
        )
        axes[1].set_title('Control Feature Importance', fontweight='bold')
        axes[1].set_xlabel('Brain Lobe')
        axes[1].set_ylabel('')
        
        # Difference (ASD - Control)
        diff = asd_attr - control_attr
        sns.heatmap(
            diff.T,
            xticklabels=region_names,
            yticklabels=self.feature_names,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.3f',
            ax=axes[2],
            cbar_kws={'label': 'Difference'}
        )
        axes[2].set_title('Difference (ASD - Control)', fontweight='bold')
        axes[2].set_xlabel('Brain Lobe')
        axes[2].set_ylabel('')
        
        plt.suptitle('Feature Importance: ASD vs Control', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Per-class comparison saved to {output_path}")


# Standalone execution
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from src.core.config import CHECKPOINT_DIR, GNN_IN_CHANNELS
    from src.models.causal_gnn import CausalBrainGNN
    from src.features.graph_factory import ABIDECausalDataset
    from torch_geometric.loader import DataLoader
    
    logger.info("="*70)
    logger.info("FEATURE ATTRIBUTION ANALYSIS")
    logger.info("="*70)
    
    # Define feature names (14 total)
    feature_names = [
        # Temporal features (8)
        'mean', 'std', 'skew', 'kurt', 'psd', 'mssd', 'range', 'autocorr',
        # Spatial features (6)
        'x', 'y', 'z_depth', 'size', 'conf_std', 'detection_count'
    ]
    
    # Load test dataset
    test_dataset = ABIDECausalDataset(split='test')
    test_data = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i] is not None]
    test_loader = DataLoader(test_data, batch_size=1)  # Batch=1 for attributions
    
    logger.info(f"Test set: {len(test_data)} subjects")
    
    # Load best model (fold 0 for simplicity)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalBrainGNN(
        num_node_features=GNN_IN_CHANNELS,
        hidden_channels=64,
        num_classes=2
    ).to(device)
    
    checkpoint_path = CHECKPOINT_DIR / "best_model_fold0.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        logger.info(f"✓ Loaded model from {checkpoint_path}")
    else:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = FeatureAttributionAnalyzer(
        model=model,
        test_loader=test_loader,
        feature_names=feature_names,
        device=device
    )
    
    # Compute attributions
    attributions = analyzer.compute_attributions(n_steps=50)
    
    # Create output directory
    output_dir = Path("results/analysis/feature_attribution")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    analyzer.visualize_feature_importance(
        attributions,
        output_dir / "feature_importance_heatmap.png"
    )
    
    analyzer.compare_temporal_vs_spatial(attributions)
    
    analyzer.analyze_per_lobe(attributions, output_dir / "per_lobe")
    
    analyzer.visualize_per_class(
        attributions,
        analyzer.labels,
        output_dir / "per_class_comparison.png"
    )
    
    logger.info("="*70)
    logger.info("✓ Feature attribution analysis complete")
    logger.info(f"  Results saved to: {output_dir}")
    logger.info("="*70)