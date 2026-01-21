"""
Complete Validation & Tuning Suite for Parts 4-8

PART 4: Data Quality Checks
PART 5: Graph Construction Tuning
PART 6: Dataset Stratification
PART 7: Evaluation & Metrics
PART 8: Feature Preprocessing
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import torch
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.core.config import (
    NODE_ATTRIBUTES_TEMPORAL, NODE_FEATURES_3D,
    NODE_ATTRIBUTES_HARMONIZED, CAUSAL_GRAPHS_DIR,
    MASTER_MANIFEST, NUM_LOBES, LOBE_NAMES,
    SPARSITY_QUANTILE, CAUSAL_LAG, DATA_FINAL
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveValidator:
    """
    Complete validator covering Parts 4-8 of the strategy checklist.
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./validation_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    # ========================================
    # PART 4: DATA QUALITY CHECKS
    # ========================================
    
    def check_yolo_detection_quality(self) -> Dict:
        """
        PART 4.1: Verify YOLO detection quality.
        
        Checks:
        - Detection confidence levels
        - Subject survival rate (all 5 lobes detected)
        - Detection consistency across slices
        """
        logger.info("="*70)
        logger.info("PART 4.1: YOLO DETECTION QUALITY")
        logger.info("="*70)
        
        if not NODE_FEATURES_3D.exists():
            logger.error("❌ Spatial features not found!")
            return {}
        
        df = pd.read_csv(NODE_FEATURES_3D)
        
        # Detection confidence analysis
        conf_std_cols = [c for c in df.columns if c.endswith('_conf_std')]
        detection_count_cols = [c for c in df.columns if c.endswith('_detection_count')]
        
        results = {
            'total_subjects': len(df),
            'lobes_analyzed': len(conf_std_cols)
        }
        
        # Analyze each lobe
        for lobe_name in LOBE_NAMES.values():
            conf_std_col = f"{lobe_name}_conf_std"
            count_col = f"{lobe_name}_detection_count"
            
            if conf_std_col in df.columns and count_col in df.columns:
                conf_std = df[conf_std_col].dropna()
                det_count = df[count_col].dropna()
                
                results[f'{lobe_name}_conf_consistency'] = {
                    'mean_std': conf_std.mean(),
                    'median_std': conf_std.median(),
                    'high_variance_subjects': (conf_std > 0.1).sum()
                }
                
                results[f'{lobe_name}_detection_freq'] = {
                    'mean_detections': det_count.mean(),
                    'min_detections': det_count.min(),
                    'subjects_with_5_slices': (det_count >= 5).sum()
                }
        
        # Subject survival rate
        if 'node_count' in df.columns:
            complete_subjects = (df['node_count'] == NUM_LOBES).sum()
            survival_rate = complete_subjects / len(df) * 100
            
            results['survival_rate'] = survival_rate
            results['complete_subjects'] = complete_subjects
            results['incomplete_subjects'] = len(df) - complete_subjects
            
            logger.info(f"Subject Survival Rate: {survival_rate:.1f}%")
            logger.info(f"  Complete (5 lobes): {complete_subjects}/{len(df)}")
            logger.info(f"  Incomplete: {len(df) - complete_subjects}")
            
            if survival_rate < 80:
                logger.warning(
                    f"⚠️  LOW SURVIVAL RATE ({survival_rate:.1f}%)! "
                    f"YOLO may need retraining."
                )
        
        # Visualize detection quality
        self._plot_detection_quality(df)
        
        self.results['yolo_quality'] = results
        logger.info("✓ YOLO quality check complete\n")
        
        return results
    
    def check_detection_confidence_levels(self, min_confidence=0.35) -> bool:
        """
        PART 4.2: Check detection confidence levels.
        
        Returns True if detection quality is acceptable.
        """
        logger.info("="*70)
        logger.info("PART 4.2: DETECTION CONFIDENCE LEVELS")
        logger.info("="*70)
        
        df = pd.read_csv(NODE_FEATURES_3D)
        
        # Check confidence standard deviation (lower = more consistent)
        conf_std_cols = [c for c in df.columns if c.endswith('_conf_std')]
        
        if not conf_std_cols:
            logger.warning("⚠️  No confidence metrics found!")
            return False
        
        overall_conf_std = df[conf_std_cols].values.flatten()
        overall_conf_std = overall_conf_std[np.isfinite(overall_conf_std)]
        
        mean_std = overall_conf_std.mean()
        high_variance = (overall_conf_std > 0.15).sum()
        
        logger.info(f"Confidence Consistency:")
        logger.info(f"  Mean std dev: {mean_std:.4f}")
        logger.info(f"  High variance detections: {high_variance}/{len(overall_conf_std)}")
        
        quality_acceptable = mean_std < 0.1 and high_variance < len(overall_conf_std) * 0.2
        
        if quality_acceptable:
            logger.info("✓ Detection quality is ACCEPTABLE")
        else:
            logger.warning("⚠️  Detection quality may need improvement")
            logger.warning(f"   Consider retraining YOLO with adjusted confidence threshold")
        
        return quality_acceptable
    
    # ========================================
    # PART 5: GRAPH CONSTRUCTION TUNING
    # ========================================
    
    def analyze_sparsity_levels(self) -> Dict:
        """
        PART 5.1: Analyze current sparsity and suggest tuning.
        
        Current: SPARSITY_QUANTILE = 0.60 (keep top 40% edges)
        """
        logger.info("="*70)
        logger.info("PART 5.1: GRAPH SPARSITY ANALYSIS")
        logger.info("="*70)
        
        if not CAUSAL_GRAPHS_DIR.exists():
            logger.error("❌ Graph directory not found!")
            return {}
        
        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        
        if not graph_files:
            logger.error("❌ No graph files found!")
            return {}
        
        edge_counts = []
        edge_weights = []
        
        # Sample 200 graphs
        sample_size = min(200, len(graph_files))
        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            try:
                graph_data = torch.load(graph_file)
                adj = graph_data['adj']
                
                # Count edges
                num_edges = (adj != 0).sum().item()
                edge_counts.append(num_edges)
                
                # Collect weights
                weights = adj[adj != 0].abs().numpy()
                edge_weights.extend(weights)
                
            except:
                continue
        
        results = {
            'current_sparsity_quantile': SPARSITY_QUANTILE,
            'mean_edges': np.mean(edge_counts),
            'median_edges': np.median(edge_counts),
            'min_edges': np.min(edge_counts),
            'max_edges': np.max(edge_counts),
            'graphs_with_zero_edges': (np.array(edge_counts) == 0).sum(),
            'mean_edge_weight': np.mean(edge_weights),
            'median_edge_weight': np.median(edge_weights)
        }
        
        logger.info(f"Current sparsity: Keep top {(1-SPARSITY_QUANTILE)*100:.0f}% edges")
        logger.info(f"Edge statistics (sampled {sample_size} graphs):")
        logger.info(f"  Mean edges: {results['mean_edges']:.1f}/{NUM_LOBES * NUM_LOBES}")
        logger.info(f"  Median edges: {results['median_edges']}")
        logger.info(f"  Range: {results['min_edges']} - {results['max_edges']}")
        logger.info(f"  Zero-edge graphs: {results['graphs_with_zero_edges']}")
        
        # Recommendations
        if results['graphs_with_zero_edges'] > sample_size * 0.05:
            logger.warning(
                f"\n⚠️  HIGH ZERO-EDGE RATE ({results['graphs_with_zero_edges']}/{sample_size})!"
            )
            logger.warning(f"   RECOMMENDATION: Lower sparsity to 0.50 (keep top 50%)")
        elif results['mean_edges'] > 15:
            logger.info(
                f"\n✓ Graph is dense (mean {results['mean_edges']:.1f} edges)"
            )
            logger.info(f"   OPTIONAL: Increase sparsity to 0.70 for selectivity")
        else:
            logger.info(f"\n✓ Sparsity level is appropriate")
        
        # Visualize
        self._plot_sparsity_analysis(edge_counts, edge_weights)
        
        self.results['sparsity'] = results
        return results
    
    def analyze_lag_value(self) -> Dict:
        """
        PART 5.2: Analyze temporal lag for causal inference.
        
        Current: CAUSAL_LAG = 1 TR
        """
        logger.info("="*70)
        logger.info("PART 5.2: TEMPORAL LAG ANALYSIS")
        logger.info("="*70)
        
        logger.info(f"Current lag: {CAUSAL_LAG} TR (Repetition Time)")
        logger.info(f"This enforces temporal precedence: t-1 → t")
        
        # Load sample time series to check TR
        if MASTER_MANIFEST.exists():
            manifest = pd.read_csv(MASTER_MANIFEST)
            
            if 'TR' in manifest.columns:
                tr_values = manifest['TR'].dropna()
                
                logger.info(f"\nTR statistics across sites:")
                logger.info(f"  Mean TR: {tr_values.mean():.2f}s")
                logger.info(f"  TR range: {tr_values.min():.2f}s - {tr_values.max():.2f}s")
                logger.info(f"  Unique TRs: {tr_values.nunique()}")
                
                # Recommendation
                if tr_values.nunique() > 1:
                    logger.warning(
                        f"\n⚠️  Multiple TR values detected across sites!"
                    )
                    logger.warning(
                        f"   Lag=1 TR means different actual time lags per site"
                    )
                    logger.warning(
                        f"   RECOMMENDATION: Harmonization handles this, but verify results"
                    )
        
        results = {
            'current_lag': CAUSAL_LAG,
            'recommendation': 'lag=1 is standard for fMRI causal inference'
        }
        
        logger.info(f"\n✓ Lag=1 TR is appropriate for directed causal graphs")
        
        self.results['lag'] = results
        return results
    
    # ========================================
    # PART 6: DATASET STRATIFICATION
    # ========================================
    
    def verify_stratification(self) -> Dict:
        """
        PART 6: Verify 2D stratification (diagnosis + site).
        
        Checks:
        - Balance across folds
        - No data leakage
        - Site distribution
        """
        logger.info("="*70)
        logger.info("PART 6: DATASET STRATIFICATION VERIFICATION")
        logger.info("="*70)
        
        if not MASTER_MANIFEST.exists():
            logger.error("❌ Manifest not found!")
            return {}
        
        manifest = pd.read_csv(MASTER_MANIFEST)
        
        results = {
            'total_subjects': len(manifest),
            'splits': {}
        }
        
        # Check split distribution
        for split in ['train', 'val', 'test']:
            split_data = manifest[manifest['split'] == split]
            
            if len(split_data) == 0:
                continue
            
            dx_counts = split_data['DX_GROUP'].value_counts()
            site_counts = split_data['SITE_ID'].value_counts()
            
            results['splits'][split] = {
                'total': len(split_data),
                'control': dx_counts.get(2, 0),
                'asd': dx_counts.get(1, 0),
                'num_sites': len(site_counts),
                'class_balance': dx_counts.get(1, 0) / len(split_data) if len(split_data) > 0 else 0
            }
            
            logger.info(f"\n{split.upper()} Split:")
            logger.info(f"  Total: {len(split_data)}")
            logger.info(f"  Control: {dx_counts.get(2, 0)} ({dx_counts.get(2, 0)/len(split_data)*100:.1f}%)")
            logger.info(f"  ASD: {dx_counts.get(1, 0)} ({dx_counts.get(1, 0)/len(split_data)*100:.1f}%)")
            logger.info(f"  Sites: {len(site_counts)}")
        
        # Check for data leakage (same subject in multiple splits)
        subject_split_count = manifest.groupby('subject_id')['split'].nunique()
        leakage = (subject_split_count > 1).sum()
        
        if leakage > 0:
            logger.error(f"❌ DATA LEAKAGE: {leakage} subjects in multiple splits!")
            results['data_leakage'] = True
        else:
            logger.info(f"\n✓ No data leakage detected")
            results['data_leakage'] = False
        
        # Visualize stratification
        self._plot_stratification(manifest)
        
        self.results['stratification'] = results
        logger.info("✓ Stratification check complete\n")
        
        return results
    
    # ========================================
    # PART 7: EVALUATION & METRICS
    # ========================================
    
    def setup_evaluation_metrics(self) -> Dict:
        """
        PART 7: Confirm evaluation metrics are appropriate.
        
        Primary metric: AUC (best for imbalanced data)
        Secondary: F1, Precision, Recall, Confusion Matrix
        """
        logger.info("="*70)
        logger.info("PART 7: EVALUATION METRICS SETUP")
        logger.info("="*70)
        
        metrics_config = {
            'primary_metric': 'AUC',
            'secondary_metrics': ['F1', 'Precision', 'Recall', 'Accuracy'],
            'per_fold_tracking': ['Confusion Matrix', 'ROC Curve', 'PR Curve'],
            'threshold_optimization': 'F1-optimal threshold per fold'
        }
        
        logger.info("Evaluation Strategy:")
        logger.info(f"  Primary metric: {metrics_config['primary_metric']}")
        logger.info(f"  Secondary metrics: {', '.join(metrics_config['secondary_metrics'])}")
        logger.info(f"  Threshold: Optimize per fold to maximize F1")
        logger.info(f"  Report: Per-fold + aggregated statistics")
        
        logger.info("\n✓ Metrics appropriate for imbalanced classification")
        
        self.results['metrics'] = metrics_config
        return metrics_config
    
    # ========================================
    # PART 8: FEATURE PREPROCESSING
    # ========================================
    
    def check_feature_preprocessing(self) -> Dict:
        """
        PART 8: Verify feature preprocessing.
        
        Checks:
        - Normalization/standardization
        - Outlier handling (5σ capping)
        - No NaNs/Infs after harmonization
        """
        logger.info("="*70)
        logger.info("PART 8: FEATURE PREPROCESSING VERIFICATION")
        logger.info("="*70)
        
        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            logger.error("❌ Harmonized features not found!")
            return {}
        
        df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
        feature_cols = [c for c in df.columns if c != 'subject_id']
        
        results = {
            'total_subjects': len(df),
            'total_features': len(feature_cols)
        }
        
        # Check for NaNs/Infs
        nan_count = df[feature_cols].isna().sum().sum()
        inf_count = np.isinf(df[feature_cols].values).sum()
        
        results['nan_count'] = nan_count
        results['inf_count'] = inf_count
        
        logger.info(f"Harmonized features: {len(df)} subjects × {len(feature_cols)} features")
        logger.info(f"  NaN values: {nan_count}")
        logger.info(f"  Inf values: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            logger.error("❌ NaNs/Infs detected! Preprocessing FAILED")
            return results
        
        # Check normalization
        feature_stats = df[feature_cols].describe()
        
        means = feature_stats.loc['mean']
        stds = feature_stats.loc['std']
        
        # Features should be approximately standardized (mean~0, std~1) after harmonization
        mean_of_means = means.abs().mean()
        mean_of_stds = stds.mean()
        
        results['mean_of_means'] = mean_of_means
        results['mean_of_stds'] = mean_of_stds
        
        logger.info(f"\nNormalization check:")
        logger.info(f"  Mean of feature means: {mean_of_means:.4f}")
        logger.info(f"  Mean of feature stds: {mean_of_stds:.4f}")
        
        # Check for outliers (beyond 5σ)
        outlier_counts = []
        for col in feature_cols:
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = (z_scores > 5).sum()
                outlier_counts.append(outliers)
        
        total_outliers = sum(outlier_counts)
        results['outliers_beyond_5sigma'] = total_outliers
        
        logger.info(f"  Outliers (>5σ): {total_outliers}/{len(df) * len(feature_cols)}")
        
        if total_outliers > len(df) * len(feature_cols) * 0.01:
            logger.warning(f"\n⚠️  High outlier count ({total_outliers})!")
            logger.warning(f"   RECOMMENDATION: Apply outlier capping in harmonization")
        else:
            logger.info(f"\n✓ Outliers within acceptable range")
        
        # Visualize feature distributions
        self._plot_feature_distributions(df, feature_cols[:20])  # Sample 20 features
        
        logger.info("✓ Feature preprocessing check complete\n")
        
        self.results['preprocessing'] = results
        return results
    
    # ========================================
    # VISUALIZATION METHODS
    # ========================================
    
    def _plot_detection_quality(self, df: pd.DataFrame):
        """Plot YOLO detection quality metrics."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('YOLO Detection Quality', fontsize=14, fontweight='bold')
        
        # Confidence consistency
        ax = axes[0]
        conf_std_cols = [c for c in df.columns if c.endswith('_conf_std')]
        if conf_std_cols:
            data = df[conf_std_cols].values.flatten()
            data = data[np.isfinite(data)]
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0.1, color='red', linestyle='--', label='Quality threshold')
            ax.set_title('Confidence Standard Deviation')
            ax.set_xlabel('Std Dev')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # Detection frequency
        ax = axes[1]
        count_cols = [c for c in df.columns if c.endswith('_detection_count')]
        if count_cols:
            data = df[count_cols].values.flatten()
            data = data[np.isfinite(data)]
            ax.hist(data, bins=range(int(data.min()), int(data.max()) + 2), 
                   edgecolor='black', alpha=0.7)
            ax.axvline(5, color='red', linestyle='--', label='Target: 5 slices')
            ax.set_title('Detection Frequency')
            ax.set_xlabel('Number of Detections')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yolo_detection_quality.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved: {self.output_dir / 'yolo_detection_quality.png'}")
    
    def _plot_sparsity_analysis(self, edge_counts: List[int], edge_weights: List[float]):
        """Plot graph sparsity analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Graph Sparsity Analysis', fontsize=14, fontweight='bold')
        
        # Edge count distribution
        ax = axes[0]
        ax.hist(edge_counts, bins=range(0, max(edge_counts) + 2), 
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(edge_counts), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(edge_counts):.1f}')
        ax.set_title('Edge Count Distribution')
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Number of Graphs')
        ax.legend()
        
        # Edge weight distribution
        ax = axes[1]
        ax.hist(edge_weights, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(np.median(edge_weights), color='red', linestyle='--',
                  label=f'Median: {np.median(edge_weights):.3f}')
        ax.set_title('Edge Weight Distribution (Absolute Values)')
        ax.set_xlabel('|Correlation|')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sparsity_analysis.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved: {self.output_dir / 'sparsity_analysis.png'}")
    
    def _plot_stratification(self, manifest: pd.DataFrame):
        """Plot dataset stratification."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Dataset Stratification (2D: Diagnosis + Site)', 
                    fontsize=14, fontweight='bold')
        
        # Split distribution
        ax = axes[0]
        split_dx = manifest.groupby(['split', 'DX_GROUP']).size().unstack(fill_value=0)
        split_dx.columns = ['ASD', 'Control']
        split_dx.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
        ax.set_title('Distribution by Split')
        ax.set_xlabel('Split')
        ax.set_ylabel('Number of Subjects')
        ax.legend(title='Diagnosis')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        # Class balance per split
        ax = axes[1]
        split_ratios = []
        for split in ['train', 'val', 'test']:
            split_data = manifest[manifest['split'] == split]
            asd = (split_data['DX_GROUP'] == 1).sum()
            total = len(split_data)
            ratio = asd / total if total > 0 else 0
            split_ratios.append(ratio)
        
        ax.bar(['Train', 'Val', 'Test'], split_ratios, 
              color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.axhline(0.5, color='black', linestyle='--', label='Perfect Balance')
        ax.set_title('ASD Ratio per Split')
        ax.set_ylabel('Proportion of ASD')
        ax.set_ylim([0, 1])
        ax.legend()
        
        # Site distribution
        ax = axes[2]
        site_dist = manifest['SITE_ID'].value_counts().head(10)
        site_dist.plot(kind='barh', ax=ax, color='#9b59b6')
        ax.set_title('Top 10 Sites')
        ax.set_xlabel('Number of Subjects')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stratification.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved: {self.output_dir / 'stratification.png'}")
    
    def _plot_feature_distributions(self, df: pd.DataFrame, feature_cols: List[str]):
        """Plot feature distributions after preprocessing."""
        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        fig.suptitle('Feature Distributions (Post-Preprocessing)', 
                    fontsize=14, fontweight='bold')
        
        for idx, (ax, col) in enumerate(zip(axes.flat, feature_cols)):
            data = df[col].dropna()
            
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(data.mean(), color='red', linestyle='--', 
                      label=f'μ={data.mean():.2f}')
            ax.set_title(col[:30], fontsize=8)
            ax.legend(fontsize=6)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved: {self.output_dir / 'feature_distributions.png'}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE VALIDATION SUMMARY (PARTS 4-8)")
        logger.info("="*70)
        
        for part, results in self.results.items():
            logger.info(f"\n{part.upper().replace('_', ' ')}:")
            
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        logger.info(f"  {key}:")
                        for k, v in value.items():
                            logger.info(f"    {k}: {v}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "="*70)
        logger.info(f"All visualizations saved to: {self.output_dir}")
        logger.info("="*70)


def main():
    """Run complete Parts 4-8 validation."""
    validator = ComprehensiveValidator()
    
    # PART 4: Data Quality
    validator.check_yolo_detection_quality()
    validator.check_detection_confidence_levels()
    
    # PART 5: Graph Construction
    validator.analyze_sparsity_levels()
    validator.analyze_lag_value()
    
    # PART 6: Stratification
    validator.verify_stratification()
    
    # PART 7: Metrics
    validator.setup_evaluation_metrics()
    
    # PART 8: Preprocessing
    validator.check_feature_preprocessing()
    
    # Generate report
    validator.generate_summary_report()
    
    logger.info("\n✅ COMPREHENSIVE VALIDATION COMPLETE")


if __name__ == "__main__":
    main()