import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Real-time and post-hoc training visualization.
    
    Tracks metrics across epochs and generates diagnostic plots:
    - Loss curves (detect overfitting/underfitting)
    - AUC progression (track learning quality)
    - Learning rate schedule (verify warmup/annealing)
    - Gradient norms (detect training instabilities)
    - Confusion matrix evolution (detect class collapse)
    """
    
    def __init__(self, output_dir: Path, num_folds: int = 5):
        """
        Initialize training monitor.
        
        Args:
            output_dir: Directory to save plots
            num_folds: Number of cross-validation folds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_folds = num_folds
        
        # Per-fold history
        self.fold_histories = {
            fold_id: {
                'train_loss': [],
                'val_loss': [],
                'val_auc': [],
                'val_f1': [],
                'val_acc': [],
                'learning_rate': [],
                'grad_norm': [],
                'confusion_matrices': []
            }
            for fold_id in range(num_folds)
        }
        
        logger.info(f"TrainingMonitor initialized")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Tracking {num_folds} folds")
    
    def log_epoch(
        self,
        fold_id: int,
        epoch: int,
        metrics: Dict[str, float],
        grad_norm: Optional[float] = None,
        confusion_matrix: Optional[np.ndarray] = None
    ):
        """
        Log metrics for one epoch.
        
        Args:
            fold_id: Current fold (0-4)
            epoch: Current epoch number
            metrics: Dictionary with keys: 'train_loss', 'val_loss', 'val_auc', 'val_f1', 'lr'
            grad_norm: Optional gradient norm (for stability tracking)
            confusion_matrix: Optional 2x2 confusion matrix
        """
        if fold_id not in self.fold_histories:
            raise ValueError(f"Invalid fold_id: {fold_id}")
        
        history = self.fold_histories[fold_id]
        
        # Log core metrics
        history['train_loss'].append(metrics.get('train_loss', 0.0))
        history['val_loss'].append(metrics.get('val_loss', 0.0))
        history['val_auc'].append(metrics.get('val_auc', 0.0))
        history['val_f1'].append(metrics.get('val_f1', 0.0))
        history['val_acc'].append(metrics.get('val_acc', 0.0))
        history['learning_rate'].append(metrics.get('lr', 0.0))
        
        # Log gradient norm if provided
        if grad_norm is not None:
            history['grad_norm'].append(grad_norm)
        
        # Log confusion matrix if provided
        if confusion_matrix is not None:
            history['confusion_matrices'].append(confusion_matrix.copy())
    
    def plot_training_curves(
        self,
        fold_id: int,
        figsize: Tuple[int, int] = (18, 12)
    ) -> Path:
        """
        Generate comprehensive training diagnostic plot.
        
        Creates a 4-panel figure:
        1. Loss curves (train/val)
        2. AUC progression
        3. Learning rate schedule
        4. Gradient norm (if tracked)
        
        Args:
            fold_id: Fold to visualize
            figsize: Figure size
        
        Returns:
            Path to saved figure
        """
        logger.info(f"Generating training curves for fold {fold_id}...")
        
        history = self.fold_histories[fold_id]
        
        if not history['train_loss']:
            logger.warning(f"No training history for fold {fold_id}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Panel 1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], label='Train Loss', 
                linewidth=2.5, color='#3498db', alpha=0.8)
        ax.plot(epochs, history['val_loss'], label='Val Loss', 
                linewidth=2.5, color='#e74c3c', alpha=0.8)
        
        # Mark best validation loss
        best_val_idx = np.argmin(history['val_loss'])
        best_val_epoch = best_val_idx + 1
        best_val_loss = history['val_loss'][best_val_idx]
        ax.scatter([best_val_epoch], [best_val_loss], 
                  color='#e74c3c', s=200, zorder=5, marker='*',
                  label=f'Best Val (Epoch {best_val_epoch})')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Loss Curves', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # Panel 2: AUC progression
        ax = axes[0, 1]
        ax.plot(epochs, history['val_auc'], 
                color='#2ecc71', linewidth=2.5, alpha=0.8, label='Validation AUC')
        
        # Reference lines
        ax.axhline(0.5, color='#95a5a6', linestyle='--', 
                  label='Random (0.5)', alpha=0.7, linewidth=2)
        
        best_auc = max(history['val_auc'])
        best_auc_idx = history['val_auc'].index(best_auc)
        best_auc_epoch = best_auc_idx + 1
        
        ax.axhline(best_auc, color='#27ae60', linestyle='--', 
                  label=f'Best: {best_auc:.4f} (Epoch {best_auc_epoch})', 
                  alpha=0.7, linewidth=2)
        ax.scatter([best_auc_epoch], [best_auc], 
                  color='#27ae60', s=200, zorder=5, marker='*')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation AUC', fontsize=12, fontweight='bold')
        ax.set_title('AUC Progression', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([0.4, 1.0])
        
        # Panel 3: Learning rate schedule
        ax = axes[1, 0]
        ax.plot(epochs, history['learning_rate'], 
                color='#f39c12', linewidth=2.5, alpha=0.8)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('LR Schedule (Warmup + Cosine)', fontsize=14, fontweight='bold', pad=15)
        ax.set_yscale('log')
        ax.grid(alpha=0.3, linestyle='--')
        
        # Panel 4: Gradient norm
        ax = axes[1, 1]
        if history['grad_norm']:
            ax.plot(epochs, history['grad_norm'], 
                   color='#9b59b6', linewidth=2.5, alpha=0.8, label='Gradient Norm')
            ax.axhline(1.0, color='#e74c3c', linestyle='--', 
                      label='Clip Threshold (1.0)', linewidth=2, alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
            ax.set_title('Gradient Stability', fontsize=14, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, 'Gradient norm not tracked', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, color='#7f8c8d')
            ax.axis('off')
        
        plt.suptitle(f'Training Diagnostics - Fold {fold_id}', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        fold_plots_dir = self.output_dir / 'fold_plots'
        fold_plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = fold_plots_dir / f'training_curves_fold_{fold_id}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Training curves saved to {output_path}")
        
        return output_path
    
    def plot_confusion_evolution(
        self,
        fold_id: int,
        key_epochs: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (18, 12)
    ) -> Optional[Path]:
        """
        Visualize confusion matrix changes across epochs.
        
        Shows if model is learning both classes or collapsing to majority class.
        
        Args:
            fold_id: Fold to visualize
            key_epochs: Specific epochs to plot (default: [1, 10, 25, 50, 75, -1])
            figsize: Figure size
        
        Returns:
            Path to saved figure (or None if no confusion matrices tracked)
        """
        logger.info(f"Generating confusion matrix evolution for fold {fold_id}...")
        
        history = self.fold_histories[fold_id]
        cm_history = history['confusion_matrices']
        
        if not cm_history:
            logger.warning(f"No confusion matrices tracked for fold {fold_id}")
            return None
        
        # Default key epochs
        if key_epochs is None:
            total_epochs = len(cm_history)
            key_epochs = [0, 9, 24, 49, 74, -1]  # Indices for epochs 1, 10, 25, 50, 75, last
            key_epochs = [e if e < total_epochs else -1 for e in key_epochs]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        class_names = ['Control', 'ASD']
        
        for idx, epoch_idx in enumerate(key_epochs):
            if idx >= len(axes):
                break
            
            # Get confusion matrix
            if epoch_idx == -1 or epoch_idx >= len(cm_history):
                epoch_idx = len(cm_history) - 1
            
            cm = cm_history[epoch_idx]
            
            # Plot heatmap
            ax = axes[idx]
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                cbar=False,
                linewidths=1,
                linecolor='white'
            )
            
            ax.set_title(f'Epoch {epoch_idx + 1}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('True', fontsize=12, fontweight='bold')
            
            # Add accuracy annotation
            accuracy = np.trace(cm) / np.sum(cm)
            ax.text(1, -0.3, f'Acc: {accuracy:.3f}', 
                   transform=ax.transData, fontsize=11,
                   ha='center', color='#2c3e50', fontweight='bold')
        
        plt.suptitle(f'Confusion Matrix Evolution - Fold {fold_id}', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / f'confusion_evolution_fold_{fold_id}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Confusion evolution saved to {output_path}")
        
        return output_path
    
    def plot_fold_comparison(self, figsize: Tuple[int, int] = (16, 10)) -> Path:
        """
        Compare final performance across all folds.
        
        Creates visualizations showing:
        - Final AUC per fold
        - Final F1 per fold
        - Best epoch per fold
        - Training stability (std of metrics)
        
        Args:
            figsize: Figure size
        
        Returns:
            Path to saved figure
        """
        logger.info("Generating fold comparison plot...")
        
        # Extract final metrics from each fold
        fold_data = []
        for fold_id in range(self.num_folds):
            history = self.fold_histories[fold_id]
            
            if not history['val_auc']:
                continue
            
            fold_data.append({
                'fold': fold_id,
                'best_auc': max(history['val_auc']),
                'best_f1': max(history['val_f1']),
                'final_auc': history['val_auc'][-1],
                'final_f1': history['val_f1'][-1],
                'best_epoch': history['val_auc'].index(max(history['val_auc'])) + 1,
                'total_epochs': len(history['val_auc'])
            })
        
        if not fold_data:
            logger.warning("No fold data available for comparison")
            return None
        
        df = pd.DataFrame(fold_data)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Panel 1: Best AUC per fold
        ax = axes[0, 0]
        bars = ax.bar(df['fold'], df['best_auc'], color='#3498db', alpha=0.7, edgecolor='black')
        ax.axhline(df['best_auc'].mean(), color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'Mean: {df["best_auc"].mean():.4f}')
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Validation AUC', fontsize=12, fontweight='bold')
        ax.set_title('Best AUC per Fold', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # Annotate bars
        for bar, value in zip(bars, df['best_auc']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Panel 2: Best F1 per fold
        ax = axes[0, 1]
        bars = ax.bar(df['fold'], df['best_f1'], color='#2ecc71', alpha=0.7, edgecolor='black')
        ax.axhline(df['best_f1'].mean(), color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'Mean: {df["best_f1"].mean():.4f}')
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Validation F1', fontsize=12, fontweight='bold')
        ax.set_title('Best F1 per Fold', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        for bar, value in zip(bars, df['best_f1']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Panel 3: Best epoch per fold
        ax = axes[1, 0]
        bars = ax.bar(df['fold'], df['best_epoch'], color='#f39c12', alpha=0.7, edgecolor='black')
        ax.axhline(df['best_epoch'].mean(), color='#e74c3c', linestyle='--', 
                  linewidth=2, label=f'Mean: {df["best_epoch"].mean():.1f}')
        ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_title('Best Epoch per Fold', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        for bar, value in zip(bars, df['best_epoch']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(value)}', ha='center', va='bottom', fontsize=10)
        
        # Panel 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        Cross-Validation Summary
        {'='*40}
        
        Best AUC:
          Mean:  {df['best_auc'].mean():.4f}
          Std:   {df['best_auc'].std():.4f}
          Range: [{df['best_auc'].min():.4f}, {df['best_auc'].max():.4f}]
        
        Best F1:
          Mean:  {df['best_f1'].mean():.4f}
          Std:   {df['best_f1'].std():.4f}
          Range: [{df['best_f1'].min():.4f}, {df['best_f1'].max():.4f}]
        
        Training:
          Avg Best Epoch: {df['best_epoch'].mean():.1f}
          Avg Total Epochs: {df['total_epochs'].mean():.1f}
        """
        
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))
        
        plt.suptitle('5-Fold Cross-Validation Comparison', 
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'fold_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"✓ Fold comparison saved to {output_path}")
        
        return output_path
    
    def save_history(self, fold_id: int) -> Path:
        """
        Save training history to JSON for later analysis.
        
        Args:
            fold_id: Fold to save
        
        Returns:
            Path to saved JSON file
        """
        history = self.fold_histories[fold_id]
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, values in history.items():
            if key == 'confusion_matrices':
                serializable_history[key] = [cm.tolist() for cm in values]
            else:
                serializable_history[key] = values
        
        output_path = self.output_dir / f'training_history_fold_{fold_id}.json'
        with open(output_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"✓ Training history saved to {output_path}")
        
        return output_path
    
    def load_history(self, fold_id: int, filepath: Path):
        """
        Load training history from JSON.
        
        Args:
            fold_id: Fold to load
            filepath: Path to JSON file
        """
        with open(filepath, 'r') as f:
            loaded_history = json.load(f)
        
        # Convert confusion matrices back to numpy arrays
        if 'confusion_matrices' in loaded_history:
            loaded_history['confusion_matrices'] = [
                np.array(cm) for cm in loaded_history['confusion_matrices']
            ]
        
        self.fold_histories[fold_id] = loaded_history
        
        logger.info(f"✓ Training history loaded from {filepath}")


# Standalone execution
if __name__ == "__main__":
    logger.info("="*70)
    logger.info("TRAINING DYNAMICS MONITOR - DEMO")
    logger.info("="*70)
    
    # Create demo monitor
    monitor = TrainingMonitor(output_dir='results/experiments/training/training_demo', num_folds=1)
    
    # Simulate training for 1 fold
    logger.info("Simulating training for demo...")
    
    np.random.seed(42)
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Simulate metrics
        train_loss = 0.8 * np.exp(-epoch / 30) + np.random.normal(0, 0.05)
        val_loss = 0.9 * np.exp(-epoch / 35) + np.random.normal(0, 0.07)
        val_auc = 0.5 + 0.3 * (1 - np.exp(-epoch / 25)) + np.random.normal(0, 0.03)
        val_f1 = 0.4 + 0.25 * (1 - np.exp(-epoch / 30)) + np.random.normal(0, 0.02)
        val_acc = 0.5 + 0.2 * (1 - np.exp(-epoch / 28)) + np.random.normal(0, 0.03)
        
        # Learning rate with warmup and cosine annealing
        if epoch < 5:
            lr = 0.001 * (epoch + 1) / 5
        else:
            lr = 0.001 * 0.5 * (1 + np.cos((epoch - 5) / (num_epochs - 5) * np.pi))
        
        # Gradient norm
        grad_norm = 0.8 + 0.3 * np.random.random()
        
        # Confusion matrix
        cm = np.array([
            [50 + int(epoch * 0.3), 50 - int(epoch * 0.3)],
            [50 - int(epoch * 0.2), 50 + int(epoch * 0.2)]
        ])
        
        metrics = {
            'train_loss': max(0, train_loss),
            'val_loss': max(0, val_loss),
            'val_auc': np.clip(val_auc, 0, 1),
            'val_f1': np.clip(val_f1, 0, 1),
            'val_acc': np.clip(val_acc, 0, 1),
            'lr': lr
        }
        
        monitor.log_epoch(fold_id=0, epoch=epoch, metrics=metrics, 
                         grad_norm=grad_norm, confusion_matrix=cm)
    
    # Generate all plots
    monitor.plot_training_curves(fold_id=0)
    monitor.plot_confusion_evolution(fold_id=0)
    monitor.save_history(fold_id=0)
    
    logger.info("="*70)
    logger.info("✓ Demo complete")
    logger.info("  Check results/experiments/training/training_demo/ for outputs")
    logger.info("="*70)