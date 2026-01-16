"""
Comprehensive Pipeline Diagnostics Tool

Analyzes the entire Neuro-CXG pipeline to identify issues before they cascade.
Provides actionable recommendations for each detected problem.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ATLAS_PATH, PHENO_PATH, MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL, NODE_FEATURES_3D,
    NODE_ATTRIBUTES_HARMONIZED, CAUSAL_GRAPHS_DIR,
    DATA_FINAL, LOBE_MAPPING, NUM_LOBES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineHealthCheck:
    """Comprehensive pipeline health checker."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed_checks = []
    
    def add_issue(self, stage: str, message: str, fix: str):
        """Add a critical issue."""
        self.issues.append({
            'stage': stage,
            'message': message,
            'fix': fix
        })
    
    def add_warning(self, stage: str, message: str):
        """Add a non-critical warning."""
        self.warnings.append({
            'stage': stage,
            'message': message
        })
    
    def add_pass(self, stage: str, message: str):
        """Add a passed check."""
        self.passed_checks.append({
            'stage': stage,
            'message': message
        })
    
    def check_atlas(self) -> bool:
        """Check atlas availability and validity."""
        logger.info("Checking Atlas...")
        
        if not ATLAS_PATH.exists():
            self.add_issue(
                "Atlas",
                f"AAL3 Atlas missing: {ATLAS_PATH}",
                "Run: python -m src.utils.atlas_validator"
            )
            return False
        
        try:
            import nibabel as nib
            atlas_img = nib.load(str(ATLAS_PATH))
            data = atlas_img.get_fdata()
            num_rois = len(np.unique(data)) - 1
            
            if num_rois not in [116, 117, 166, 170]:
                self.add_issue(
                    "Atlas",
                    f"Unexpected ROI count: {num_rois} (expected 116/117/166/170)",
                    "Re-download atlas from official source"
                )
                return False
            
            self.add_pass("Atlas", f"✓ Valid atlas with {num_rois} ROIs")
            return True
            
        except Exception as e:
            self.add_issue(
                "Atlas",
                f"Atlas corrupted: {e}",
                "Re-download atlas"
            )
            return False
    
    def check_lobe_mapping(self) -> bool:
        """Check LOBE_MAPPING consistency."""
        logger.info("Checking LOBE_MAPPING...")
        
        try:
            # Validate LOBE_MAPPING structure
            if len(LOBE_MAPPING) != NUM_LOBES:
                raise ValueError(f"Expected {NUM_LOBES} lobes, got {len(LOBE_MAPPING)}")
            
            # Check for completeness (no duplicates, covers 1-170 AAL ROIs)
            all_rois = set()
            for lobe_id, roi_list in LOBE_MAPPING.items():
                for roi in roi_list:
                    if roi in all_rois:
                        raise ValueError(f"Duplicate ROI {roi} in lobe {lobe_id}")
                    all_rois.add(roi)
            
            # Verify range
            expected_rois = set(range(1, 171))  # AAL3 has 170 ROIs (1-indexed)
            if all_rois != expected_rois:
                missing = expected_rois - all_rois
                extra = all_rois - expected_rois
                if missing:
                    logger.warning(f"Missing ROIs: {missing}")
                if extra:
                    logger.warning(f"Extra ROIs: {extra}")
            
            self.add_pass("Config", "✓ LOBE_MAPPING valid")
            return True
        except ValueError as e:
            self.add_issue(
                "Config",
                f"LOBE_MAPPING invalid: {e}",
                "Fix LOBE_MAPPING in src/config.py"
            )
            return False
    
    def check_manifest(self) -> Tuple[bool, pd.DataFrame]:
        """Check master manifest."""
        logger.info("Checking manifest...")
        
        if not MASTER_MANIFEST.exists():
            self.add_issue(
                "Manifest",
                f"Master manifest missing: {MASTER_MANIFEST}",
                "Run: python -m src.utils.manifest"
            )
            return False, None
        
        try:
            df = pd.read_csv(MASTER_MANIFEST)
            
            # Check required columns
            required = ['subject_id', 'split', 'DX_GROUP', 'SITE_ID']
            missing = [c for c in required if c not in df.columns]
            
            if missing:
                self.add_issue(
                    "Manifest",
                    f"Missing columns: {missing}",
                    "Regenerate manifest with required columns"
                )
                return False, None
            
            # Check splits
            splits = df['split'].unique()
            if not all(s in splits for s in ['train', 'val', 'test']):
                self.add_warning(
                    "Manifest",
                    f"Unexpected splits: {splits}"
                )
            
            # Check diagnosis distribution
            dx_counts = df['DX_GROUP'].value_counts()
            
            self.add_pass(
                "Manifest",
                f"✓ {len(df)} subjects across {len(splits)} splits"
            )
            return True, df
            
        except Exception as e:
            self.add_issue(
                "Manifest",
                f"Error reading manifest: {e}",
                "Check manifest file integrity"
            )
            return False, None
    
    def check_time_series(self, manifest: pd.DataFrame) -> Dict:
        """Check time series availability."""
        logger.info("Checking time series files...")
        
        if manifest is None:
            return {'available': 0, 'missing': 0}
        
        available = 0
        missing = 0
        
        for _, row in manifest.iterrows():
            ts_path = DATA_FINAL / row['split'] / 'time_series' / f"{row['subject_id']}_ts.npy"
            
            if ts_path.exists():
                try:
                    data = np.load(ts_path)
                    if data.shape[0] < 50:
                        self.add_warning(
                            "Time Series",
                            f"Subject {row['subject_id']}: Only {data.shape[0]} timepoints"
                        )
                    available += 1
                except:
                    missing += 1
            else:
                missing += 1
        
        if missing > 0:
            self.add_warning(
                "Time Series",
                f"{missing}/{len(manifest)} subjects missing time series"
            )
        else:
            self.add_pass(
                "Time Series",
                f"✓ All {available} subjects have time series"
            )
        
        return {'available': available, 'missing': missing}
    
    def check_temporal_features(self, manifest: pd.DataFrame) -> bool:
        """Check temporal feature extraction."""
        logger.info("Checking temporal features...")
        
        if not NODE_ATTRIBUTES_TEMPORAL.exists():
            self.add_issue(
                "Features",
                f"Temporal features missing: {NODE_ATTRIBUTES_TEMPORAL}",
                "Run: python -m src.utils.compute_roi"
            )
            return False
        
        try:
            df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            feature_cols = [c for c in df.columns if c != 'subject_id']
            
            # Check for NaNs
            nan_count = df[feature_cols].isna().sum().sum()
            total_values = len(df) * len(feature_cols)
            nan_pct = (nan_count / total_values) * 100
            
            if nan_pct > 20:
                self.add_issue(
                    "Features",
                    f"CRITICAL: {nan_pct:.1f}% of temporal features are NaN!",
                    "This indicates feature extraction failure. Check:\n"
                    "  1. Atlas file is valid\n"
                    "  2. Time series are correctly formatted\n"
                    "  3. ROI extraction completed successfully"
                )
                return False
            elif nan_pct > 5:
                self.add_warning(
                    "Features",
                    f"{nan_pct:.1f}% of temporal features are NaN"
                )
            
            # Check ROI count
            expected_rois = len(feature_cols) // 6  # 6 features per ROI
            
            if expected_rois not in [116, 117, 164, 166, 170]:
                self.add_warning(
                    "Features",
                    f"Detected {expected_rois} ROIs (expected 116/117/164-166/170)"
                )
            
            self.add_pass(
                "Features",
                f"✓ Temporal features for {len(df)} subjects ({expected_rois} ROIs)"
            )
            return True
            
        except Exception as e:
            self.add_issue(
                "Features",
                f"Error reading temporal features: {e}",
                "Regenerate temporal features"
            )
            return False
    
    def check_harmonization(self) -> bool:
        """Check harmonized features."""
        logger.info("Checking harmonization...")
        
        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            self.add_issue(
                "Harmonization",
                f"Harmonized features missing: {NODE_ATTRIBUTES_HARMONIZED}",
                "Run: python -m src.data.harmonize"
            )
            return False
        
        try:
            df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            feature_cols = [c for c in df.columns if c != 'subject_id']
            
            # Check for NaNs (should be ZERO after harmonization)
            nan_count = df[feature_cols].isna().sum().sum()
            
            if nan_count > 0:
                self.add_issue(
                    "Harmonization",
                    f"CRITICAL: {nan_count} NaNs in harmonized features!",
                    "Re-run harmonization with safe_harmonization.py"
                )
                return False
            
            # Check for infinites
            inf_count = np.isinf(df[feature_cols].values).sum()
            
            if inf_count > 0:
                self.add_warning(
                    "Harmonization",
                    f"{inf_count} infinite values in harmonized features"
                )
            
            self.add_pass(
                "Harmonization",
                f"✓ Clean harmonized features for {len(df)} subjects"
            )
            return True
            
        except Exception as e:
            self.add_issue(
                "Harmonization",
                f"Error reading harmonized features: {e}",
                "Re-run harmonization"
            )
            return False
    
    def check_spatial_features(self) -> bool:
        """Check spatial feature extraction."""
        logger.info("Checking spatial features...")
        
        if not NODE_FEATURES_3D.exists():
            self.add_issue(
                "Spatial Features",
                f"3D features missing: {NODE_FEATURES_3D}",
                "Run: python -m src.data.extract_features"
            )
            return False
        
        try:
            df = pd.read_csv(NODE_FEATURES_3D)
            
            # Check node_count column
            if 'node_count' in df.columns:
                counts = df['node_count'].value_counts()
                
                if 5 not in counts.index or counts[5] < len(df) * 0.5:
                    self.add_warning(
                        "Spatial Features",
                        f"Many subjects don't have all 5 lobes detected: {counts.to_dict()}"
                    )
                else:
                    complete = counts[5]
                    self.add_pass(
                        "Spatial Features",
                        f"✓ {complete}/{len(df)} subjects have all 5 lobes"
                    )
            
            return True
            
        except Exception as e:
            self.add_issue(
                "Spatial Features",
                f"Error reading spatial features: {e}",
                "Re-run feature extraction"
            )
            return False
    
    def check_causal_graphs(self, manifest: pd.DataFrame) -> Dict:
        """Check causal graph construction."""
        logger.info("Checking causal graphs...")
        
        if not CAUSAL_GRAPHS_DIR.exists():
            self.add_issue(
                "Graphs",
                f"Graph directory missing: {CAUSAL_GRAPHS_DIR}",
                "Run: python -m src.data.construct_causal"
            )
            return {'available': 0, 'missing': 0}
        
        if manifest is None:
            return {'available': 0, 'missing': 0}
        
        available = 0
        missing = 0
        corrupted = 0
        
        for _, row in manifest.iterrows():
            graph_path = CAUSAL_GRAPHS_DIR / f"{row['subject_id']}_graph.pt"
            
            if graph_path.exists():
                try:
                    graph_data = torch.load(graph_path)
                    
                    # Validate structure
                    if 'adj' not in graph_data:
                        corrupted += 1
                        continue
                    
                    adj = graph_data['adj']
                    
                    # Check for NaN/Inf in adjacency
                    if torch.isnan(adj).any() or torch.isinf(adj).any():
                        corrupted += 1
                        continue
                    
                    available += 1
                except:
                    corrupted += 1
            else:
                missing += 1
        
        if corrupted > 0:
            self.add_warning(
                "Graphs",
                f"{corrupted} graphs are corrupted"
            )
        
        if missing > 0:
            self.add_warning(
                "Graphs",
                f"{missing}/{len(manifest)} graphs missing"
            )
        
        if available == len(manifest):
            self.add_pass(
                "Graphs",
                f"✓ All {available} causal graphs available"
            )
        
        return {
            'available': available,
            'missing': missing,
            'corrupted': corrupted
        }
    
    def generate_report(self):
        """Generate final diagnostic report."""
        print("\n" + "="*70)
        print("NEURO-CXG PIPELINE DIAGNOSTICS REPORT")
        print("="*70)
        
        # Passed checks
        if self.passed_checks:
            print("\n✓ PASSED CHECKS:")
            print("-"*70)
            for check in self.passed_checks:
                print(f"  [{check['stage']}] {check['message']}")
        
        # Warnings
        if self.warnings:
            print("\n⚠ WARNINGS:")
            print("-"*70)
            for warn in self.warnings:
                print(f"  [{warn['stage']}] {warn['message']}")
        
        # Critical issues
        if self.issues:
            print("\n❌ CRITICAL ISSUES:")
            print("-"*70)
            for issue in self.issues:
                print(f"\n  [{issue['stage']}]")
                print(f"  Problem: {issue['message']}")
                print(f"  Fix: {issue['fix']}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY:")
        print(f"  Passed: {len(self.passed_checks)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Critical Issues: {len(self.issues)}")
        print("="*70)
        
        if self.issues:
            print("\n⚠️  PIPELINE HAS CRITICAL ISSUES - FIX BEFORE CONTINUING")
            return False
        elif self.warnings:
            print("\n✓ Pipeline functional but has warnings to address")
            return True
        else:
            print("\n✅ PIPELINE FULLY HEALTHY")
            return True


def run_full_diagnostics():
    """Execute full pipeline diagnostics."""
    checker = PipelineHealthCheck()
    
    # Run all checks
    checker.check_atlas()
    checker.check_lobe_mapping()
    manifest_ok, manifest = checker.check_manifest()
    
    if manifest_ok:
        checker.check_time_series(manifest)
        checker.check_temporal_features(manifest)
        checker.check_harmonization()
        checker.check_spatial_features()
        checker.check_causal_graphs(manifest)
    
    # Generate report
    is_healthy = checker.generate_report()
    
    return is_healthy


if __name__ == "__main__":
    is_healthy = run_full_diagnostics()
    sys.exit(0 if is_healthy else 1)