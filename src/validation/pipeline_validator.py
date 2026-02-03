#!/usr/bin/env python3
"""
Comprehensive Pipeline Validator

Consolidates all validation logic into a single, maintainable module.
Provides pre-flight checks, stage-specific validation, and post-training analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ATLAS_PATH, MASTER_MANIFEST, DATA_ROOT, DATA_PROCESSED, DATA_FINAL,
    NODE_ATTRIBUTES_TEMPORAL, NODE_FEATURES_3D, NODE_ATTRIBUTES_HARMONIZED,
    CAUSAL_GRAPHS_DIR, CHECKPOINT_DIR, YOLO_WEIGHTS_PATH,
    LOBE_MAPPING, NUM_LOBES, NUM_TEMPORAL_FEATURES, NUM_SPATIAL_FEATURES,
    GNN_IN_CHANNELS, SPARSITY_QUANTILE, K_FOLDS,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Structured validation result."""
    stage: str
    passed: bool
    message: str
    severity: str  # 'critical', 'warning', 'info'
    fix_suggestion: Optional[str] = None
    metrics: Optional[Dict] = None


class PipelineValidator:
    """
    Unified pipeline validation system.
    
    Validates data integrity, feature quality, and model outputs
    at each stage of the pipeline.
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.metrics: Dict = {}
    
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        
        # Log immediately
        if result.severity == 'critical':
            logger.error(f"[{result.stage}] {result.message}")
            if result.fix_suggestion:
                logger.error(f"  Fix: {result.fix_suggestion}")
        elif result.severity == 'warning':
            logger.warning(f"[{result.stage}] {result.message}")
        else:
            logger.info(f"[{result.stage}] {result.message}")
    
    # STAGE 1: ENVIRONMENT VALIDATION
    
    def validate_environment(self) -> bool:
        """Pre-flight environment checks."""
        logger.info("="*70)
        logger.info("STAGE 1: ENVIRONMENT VALIDATION")
        logger.info("="*70)
        
        all_passed = True
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            self.add_result(ValidationResult(
                stage="Environment",
                passed=False,
                message=f"Python {sys.version_info.major}.{sys.version_info.minor} detected",
                severity='critical',
                fix_suggestion="Upgrade to Python 3.8+"
            ))
            all_passed = False
        else:
            self.add_result(ValidationResult(
                stage="Environment",
                passed=True,
                message=f"✓ Python {sys.version_info.major}.{sys.version_info.minor}",
                severity='info'
            ))
        
        # Check critical paths
        critical_dirs = {
            'DATA_ROOT': DATA_ROOT,
            'DATA_PROCESSED': DATA_PROCESSED,
            'DATA_FINAL': DATA_FINAL,
        }
        
        for name, path in critical_dirs.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.add_result(ValidationResult(
                    stage="Environment",
                    passed=True,
                    message=f"Created {name}: {path}",
                    severity='info'
                ))
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            self.add_result(ValidationResult(
                stage="Environment",
                passed=True,
                message=f"✓ CUDA available: {device_name}",
                severity='info'
            ))
        else:
            self.add_result(ValidationResult(
                stage="Environment",
                passed=True,
                message="⚠ CPU-only mode (training will be slow)",
                severity='warning'
            ))
        
        # Validate configuration consistency
        if len(LOBE_MAPPING) != NUM_LOBES:
            self.add_result(ValidationResult(
                stage="Environment",
                passed=False,
                message=f"Config mismatch: LOBE_MAPPING has {len(LOBE_MAPPING)} lobes but NUM_LOBES={NUM_LOBES}",
                severity='critical',
                fix_suggestion="Fix LOBE_MAPPING in src/core/config.py"
            ))
            all_passed = False
        
        expected_features = NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES
        if GNN_IN_CHANNELS != expected_features:
            self.add_result(ValidationResult(
                stage="Environment",
                passed=False,
                message=f"Config mismatch: GNN_IN_CHANNELS={GNN_IN_CHANNELS} but expected {expected_features}",
                severity='critical',
                fix_suggestion="Set GNN_IN_CHANNELS = NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES"
            ))
            all_passed = False
        
        return all_passed
    
    # STAGE 2: DATA VALIDATION
    
    def validate_downloaded_data(self) -> bool:
        """Validate downloaded fMRI data."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: DATA VALIDATION")
        logger.info("="*70)
        
        all_passed = True
        
        # Check images
        image_dir = DATA_ROOT / "images"
        if not image_dir.exists() or not list(image_dir.glob("*.png")):
            self.add_result(ValidationResult(
                stage="Data",
                passed=False,
                message="No PNG images found",
                severity='critical',
                fix_suggestion="Run: python -m src.data.abide_download"
            ))
            return False
        
        png_files = list(image_dir.glob("*.png"))
        subjects = set(f.stem.rsplit('_z', 1)[0] for f in png_files)
        
        # Check time series
        ts_files = list(DATA_PROCESSED.glob("*_ts.npy"))
        ts_subjects = set(f.stem.replace('_ts', '') for f in ts_files)
        
        # Find mismatches
        missing_ts = subjects - ts_subjects
        missing_img = ts_subjects - subjects
        
        if missing_ts:
            self.add_result(ValidationResult(
                stage="Data",
                passed=False,
                message=f"{len(missing_ts)} subjects have images but no time series",
                severity='warning',
                metrics={'missing_subjects': list(missing_ts)[:5]}
            ))
        
        if missing_img:
            self.add_result(ValidationResult(
                stage="Data",
                passed=False,
                message=f"{len(missing_img)} subjects have time series but no images",
                severity='warning'
            ))
        
        complete_subjects = subjects & ts_subjects
        self.add_result(ValidationResult(
            stage="Data",
            passed=True,
            message=f"✓ {len(complete_subjects)} subjects with complete data",
            severity='info',
            metrics={
                'total_images': len(png_files),
                'total_subjects': len(complete_subjects)
            }
        ))
        
        # Sample validation
        sample_size = min(10, len(ts_files))
        corrupted = 0
        wrong_shape = 0
        
        for ts_file in np.random.choice(ts_files, sample_size, replace=False):
            try:
                data = np.load(ts_file)
                if data.ndim != 2:
                    wrong_shape += 1
                elif np.isnan(data).any() or np.isinf(data).any():
                    corrupted += 1
            except Exception:
                corrupted += 1
        
        if corrupted > 0 or wrong_shape > 0:
            self.add_result(ValidationResult(
                stage="Data",
                passed=False,
                message=f"Sample validation: {corrupted} corrupted, {wrong_shape} wrong shape",
                severity='critical',
                fix_suggestion="Re-run data download"
            ))
            all_passed = False
        
        return all_passed
    
    # STAGE 3: FEATURE VALIDATION
    
    def validate_features(self) -> bool:
        """Validate extracted features."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: FEATURE VALIDATION")
        logger.info("="*70)
        
        all_passed = True
        
        # Check temporal features
        if not NODE_ATTRIBUTES_TEMPORAL.exists():
            self.add_result(ValidationResult(
                stage="Features",
                passed=False,
                message="Temporal features not found",
                severity='critical',
                fix_suggestion="Run: python -m src.utils.compute_roi"
            ))
            return False
        
        try:
            temporal_df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            feature_cols = [c for c in temporal_df.columns if c != 'subject_id']
            
            # Check NaN levels
            nan_count = temporal_df[feature_cols].isna().sum().sum()
            total_values = len(temporal_df) * len(feature_cols)
            nan_pct = (nan_count / total_values) * 100
            
            if nan_pct > 5:
                self.add_result(ValidationResult(
                    stage="Features",
                    passed=False,
                    message=f"High NaN rate: {nan_pct:.1f}%",
                    severity='critical' if nan_pct > 20 else 'warning',
                    fix_suggestion="Check atlas alignment and time series quality"
                ))
                if nan_pct > 20:
                    all_passed = False
            else:
                self.add_result(ValidationResult(
                    stage="Features",
                    passed=True,
                    message=f"✓ Temporal features: {len(temporal_df)} subjects, {nan_pct:.2f}% NaN",
                    severity='info'
                ))
            
            # Check value distributions
            numeric_data = temporal_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(numeric_data) > 0:
                extreme_values = (np.abs(numeric_data.values) > 1e6).sum()
                if extreme_values > 0:
                    self.add_result(ValidationResult(
                        stage="Features",
                        passed=False,
                        message=f"{extreme_values} extreme values detected (|x| > 1e6)",
                        severity='warning',
                        fix_suggestion="Check feature extraction for numerical issues"
                    ))
        
        except Exception as e:
            self.add_result(ValidationResult(
                stage="Features",
                passed=False,
                message=f"Error loading temporal features: {e}",
                severity='critical',
                fix_suggestion="Regenerate temporal features"
            ))
            return False
        
        # Check spatial features
        if not NODE_FEATURES_3D.exists():
            self.add_result(ValidationResult(
                stage="Features",
                passed=False,
                message="Spatial features not found",
                severity='critical',
                fix_suggestion="Run: python -m src.features.extract_features"
            ))
            return False
        
        try:
            spatial_df = pd.read_csv(NODE_FEATURES_3D)
            
            # Check YOLO detection completeness
            lobe_cols = [c for c in spatial_df.columns if any(c.startswith(f"{lobe}_") for lobe in ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Limbic'])]
            
            # Count subjects with all lobes detected
            complete_detections = 0
            for _, row in spatial_df.iterrows():
                has_all = all(pd.notna(row[col]) for col in lobe_cols[:5])  # Check first feature of each lobe
                if has_all:
                    complete_detections += 1
            
            survival_rate = (complete_detections / len(spatial_df)) * 100 if len(spatial_df) > 0 else 0
            
            if survival_rate < 80:
                self.add_result(ValidationResult(
                    stage="Features",
                    passed=False,
                    message=f"Low YOLO survival rate: {survival_rate:.1f}%",
                    severity='warning',
                    fix_suggestion="Check YOLO model quality or detection threshold"
                ))
            else:
                self.add_result(ValidationResult(
                    stage="Features",
                    passed=True,
                    message=f"✓ Spatial features: {complete_detections}/{len(spatial_df)} complete ({survival_rate:.1f}%)",
                    severity='info'
                ))
        
        except Exception as e:
            self.add_result(ValidationResult(
                stage="Features",
                passed=False,
                message=f"Error loading spatial features: {e}",
                severity='critical'
            ))
            all_passed = False
        
        # Check harmonized features
        if NODE_ATTRIBUTES_HARMONIZED.exists():
            try:
                harm_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
                harm_cols = [c for c in harm_df.columns if c != 'subject_id']
                
                harm_nan = harm_df[harm_cols].isna().sum().sum()
                harm_inf = np.isinf(harm_df[harm_cols].values).sum()
                
                if harm_nan > 0 or harm_inf > 0:
                    self.add_result(ValidationResult(
                        stage="Features",
                        passed=False,
                        message=f"Harmonized features have {harm_nan} NaN, {harm_inf} Inf",
                        severity='critical',
                        fix_suggestion="Re-run harmonization with safe_harmonization.py"
                    ))
                    all_passed = False
                else:
                    self.add_result(ValidationResult(
                        stage="Features",
                        passed=True,
                        message=f"✓ Harmonized features: {len(harm_df)} subjects, clean",
                        severity='info'
                    ))
            except Exception as e:
                self.add_result(ValidationResult(
                    stage="Features",
                    passed=False,
                    message=f"Error loading harmonized features: {e}",
                    severity='warning'
                ))
        
        return all_passed
    
    # STAGE 4: GRAPH VALIDATION
    
    def validate_graphs(self) -> bool:
        """Validate causal graph construction."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: GRAPH VALIDATION")
        logger.info("="*70)
        
        if not CAUSAL_GRAPHS_DIR.exists() or not list(CAUSAL_GRAPHS_DIR.glob("*.pt")):
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message="No graph files found",
                severity='critical',
                fix_suggestion="Run: python -m src.features.construct_causal"
            ))
            return False
        
        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        
        # Sample validation
        sample_size = min(50, len(graph_files))
        stats = {
            'valid': 0,
            'corrupted': 0,
            'zero_edges': 0,
            'wrong_shape': 0,
            'edge_counts': []
        }
        
        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            try:
                graph_data = torch.load(graph_file, weights_only=False)
                
                if 'adj' not in graph_data:
                    stats['corrupted'] += 1
                    continue
                
                adj = graph_data['adj']
                
                # Check shape
                if adj.shape != (NUM_LOBES, NUM_LOBES):
                    stats['wrong_shape'] += 1
                    continue
                
                # Check for NaN/Inf
                if torch.isnan(adj).any() or torch.isinf(adj).any():
                    stats['corrupted'] += 1
                    continue
                
                # Count edges
                num_edges = (adj != 0).sum().item()
                stats['edge_counts'].append(num_edges)
                
                if num_edges == 0:
                    stats['zero_edges'] += 1
                else:
                    stats['valid'] += 1
            
            except Exception:
                stats['corrupted'] += 1
        
        # Analyze results
        all_passed = True
        
        if stats['corrupted'] > sample_size * 0.05:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['corrupted']}/{sample_size} graphs corrupted",
                severity='critical',
                fix_suggestion="Re-run graph construction"
            ))
            all_passed = False
        
        if stats['wrong_shape'] > 0:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['wrong_shape']} graphs have wrong shape (expected {NUM_LOBES}×{NUM_LOBES})",
                severity='critical',
                fix_suggestion="Clear graph directory and rebuild"
            ))
            all_passed = False
        
        if stats['zero_edges'] > sample_size * 0.1:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['zero_edges']}/{sample_size} graphs have zero edges",
                severity='warning',
                fix_suggestion=f"Lower SPARSITY_QUANTILE from {SPARSITY_QUANTILE} to 0.70 or 0.60"
            ))
        
        if stats['edge_counts']:
            mean_edges = np.mean(stats['edge_counts'])
            median_edges = np.median(stats['edge_counts'])
            
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=True,
                message=f"✓ {len(graph_files)} graphs, mean edges: {mean_edges:.1f}, median: {median_edges:.0f}",
                severity='info',
                metrics={
                    'total_graphs': len(graph_files),
                    'mean_edges': mean_edges,
                    'median_edges': median_edges,
                    'max_edges': max(stats['edge_counts']),
                    'min_edges': min(stats['edge_counts'])
                }
            ))
        
        return all_passed
    
    # STAGE 5: MODEL VALIDATION
    
    def validate_trained_models(self) -> bool:
        """Validate trained model checkpoints."""
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: MODEL VALIDATION")
        logger.info("="*70)
        
        if not CHECKPOINT_DIR.exists():
            self.add_result(ValidationResult(
                stage="Models",
                passed=False,
                message="Checkpoint directory not found",
                severity='warning',
                fix_suggestion="Train models first"
            ))
            return False
        
        # Check for fold checkpoints
        fold_checkpoints = list(CHECKPOINT_DIR.glob("best_model_fold*.pt"))
        
        if len(fold_checkpoints) == 0:
            self.add_result(ValidationResult(
                stage="Models",
                passed=False,
                message="No trained models found",
                severity='warning',
                fix_suggestion="Run: python -m src.models.gnn_model"
            ))
            return False
        
        if len(fold_checkpoints) < K_FOLDS:
            self.add_result(ValidationResult(
                stage="Models",
                passed=False,
                message=f"Incomplete training: {len(fold_checkpoints)}/{K_FOLDS} folds",
                severity='warning'
            ))
        
        # Validate checkpoint contents
        fold_metrics = []
        for ckpt_path in fold_checkpoints:
            try:
                ckpt = torch.load(ckpt_path, weights_only=False)
                
                required_keys = ['model_state', 'epoch']
                missing = [k for k in required_keys if k not in ckpt]
                
                if missing:
                    self.add_result(ValidationResult(
                        stage="Models",
                        passed=False,
                        message=f"{ckpt_path.name} missing keys: {missing}",
                        severity='warning'
                    ))
                else:
                    metrics = {
                        'fold': ckpt_path.stem.replace('best_model_fold', ''),
                        'epoch': ckpt.get('epoch', -1),
                        'auc': ckpt.get('auc', 0.0),
                        'f1': ckpt.get('f1', 0.0)
                    }
                    fold_metrics.append(metrics)
            
            except Exception as e:
                self.add_result(ValidationResult(
                    stage="Models",
                    passed=False,
                    message=f"Error loading {ckpt_path.name}: {e}",
                    severity='warning'
                ))
        
        if fold_metrics:
            mean_auc = np.mean([m['auc'] for m in fold_metrics])
            mean_f1 = np.mean([m['f1'] for m in fold_metrics])
            
            self.add_result(ValidationResult(
                stage="Models",
                passed=True,
                message=f"✓ {len(fold_checkpoints)} trained models, mean AUC: {mean_auc:.4f}, F1: {mean_f1:.4f}",
                severity='info',
                metrics={'fold_metrics': fold_metrics}
            ))
        
        return True
    
    # REPORTING
    
    def generate_report(self) -> Tuple[bool, Dict]:
        """Generate comprehensive validation report."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATION REPORT")
        logger.info("="*70)
        
        # Group by severity
        critical = [r for r in self.results if r.severity == 'critical' and not r.passed]
        warnings = [r for r in self.results if r.severity == 'warning']
        passed = [r for r in self.results if r.passed and r.severity == 'info']
        
        # Critical issues
        if critical:
            logger.error(f"\n❌ CRITICAL ISSUES ({len(critical)}):")
            logger.error("-"*70)
            for result in critical:
                logger.error(f"  [{result.stage}] {result.message}")
                if result.fix_suggestion:
                    logger.error(f"    → Fix: {result.fix_suggestion}")
        
        # Warnings
        if warnings:
            logger.warning(f"\n⚠ WARNINGS ({len(warnings)}):")
            logger.warning("-"*70)
            for result in warnings:
                logger.warning(f"  [{result.stage}] {result.message}")
                if result.fix_suggestion:
                    logger.warning(f"    → Suggestion: {result.fix_suggestion}")
        
        # Passed checks
        if passed:
            logger.info(f"\n✓ PASSED CHECKS ({len(passed)}):")
            logger.info("-"*70)
            for result in passed:
                logger.info(f"  {result.message}")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY:")
        logger.info(f"  Passed: {len(passed)}")
        logger.info(f"  Warnings: {len(warnings)}")
        logger.info(f"  Critical Issues: {len(critical)}")
        logger.info("="*70)
        
        is_healthy = len(critical) == 0
        
        if is_healthy:
            if warnings:
                logger.info("\n✓ Pipeline functional with warnings")
            else:
                logger.info("\n✅ PIPELINE FULLY HEALTHY")
        else:
            logger.error("\n❌ PIPELINE HAS CRITICAL ISSUES")
        
        report = {
            'healthy': is_healthy,
            'passed': len(passed),
            'warnings': len(warnings),
            'critical': len(critical),
            'results': self.results,
            'metrics': self.metrics
        }
        
        return is_healthy, report
    
    def run_full_validation(self) -> bool:
        """Execute complete pipeline validation."""
        logger.info("Starting comprehensive pipeline validation...")
        
        # Run all validation stages
        self.validate_environment()
        self.validate_downloaded_data()
        self.validate_features()
        self.validate_graphs()
        self.validate_trained_models()
        
        # Generate final report
        is_healthy, report = self.generate_report()
        
        return is_healthy


def main():
    """CLI entry point."""
    validator = PipelineValidator()
    is_healthy = validator.run_full_validation()
    sys.exit(0 if is_healthy else 1)


if __name__ == "__main__":
    main()
