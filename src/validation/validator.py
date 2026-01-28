import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ATLAS_PATH, MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL, NODE_FEATURES_3D,
    NODE_ATTRIBUTES_HARMONIZED, CAUSAL_GRAPHS_DIR,
    LOBE_MAPPING, NUM_LOBES, 
    NUM_TEMPORAL_FEATURES, SPARSITY_QUANTILE
)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineHealthCheck:
    """Unified pipeline validation and diagnostics."""
    
    def __init__(self, visualize: bool = False):
        self.visualize = visualize
        self.output_dir = Path("./results/validation_outputs")
        if visualize:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.issues = []
        self.warnings = []
        self.passed_checks = []
        self.results = {}
    
    # RESULT TRACKING
    
    def add_issue(self, stage: str, message: str, fix: str):
        """Record a critical issue."""
        self.issues.append({
            'stage': stage,
            'message': message,
            'fix': fix
        })
    
    def add_warning(self, stage: str, message: str):
        """Record a non-critical warning."""
        self.warnings.append({
            'stage': stage,
            'message': message
        })
    
    def add_pass(self, stage: str, message: str):
        """Record a passed check."""
        self.passed_checks.append({
            'stage': stage,
            'message': message
        })
    
    # VALIDATION CHECKS
    
    def check_atlas(self) -> bool:
        """Validate atlas file."""
        logger.info("Checking atlas...")
        
        if not ATLAS_PATH.exists():
            self.add_issue(
                "Atlas",
                f"Atlas missing: {ATLAS_PATH}",
                "Run: python -m src.validation.atlas_validator"
            )
            return False
        
        try:
            import nibabel as nib
            atlas_img = nib.load(str(ATLAS_PATH))
            data = atlas_img.get_fdata()
            num_rois = len(np.unique(data)) - 1
            
            valid_counts = {116, 117, 164, 166, 170}
            if num_rois not in valid_counts:
                self.add_warning(
                    "Atlas",
                    f"Unexpected ROI count: {num_rois} (expected {valid_counts})"
                )
            
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
        """Validate LOBE_MAPPING configuration."""
        logger.info("Checking LOBE_MAPPING...")
        
        try:
            if len(LOBE_MAPPING) != NUM_LOBES:
                raise ValueError(f"Expected {NUM_LOBES} lobes, got {len(LOBE_MAPPING)}")
            
            # Check completeness
            all_rois = set()
            for lobe_id, roi_list in LOBE_MAPPING.items():
                for roi in roi_list:
                    if roi in all_rois:
                        raise ValueError(f"Duplicate ROI {roi} in lobe {lobe_id}")
                    all_rois.add(roi)
            
            # Verify range (AAL3: 1-170)
            expected_rois = set(range(1, 171))
            if all_rois != expected_rois:
                missing = expected_rois - all_rois
                extra = all_rois - expected_rois
                if missing or extra:
                    self.add_warning(
                        "Config",
                        f"ROI coverage: missing={len(missing)}, extra={len(extra)}"
                    )
            
            self.add_pass("Config", "✓ LOBE_MAPPING valid")
            return True
            
        except ValueError as e:
            self.add_issue(
                "Config",
                f"LOBE_MAPPING invalid: {e}",
                "Fix LOBE_MAPPING in src/core/config.py"
            )
            return False
    
    def check_manifest(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Validate master manifest."""
        logger.info("Checking manifest...")
        
        if not MASTER_MANIFEST.exists():
            self.add_issue(
                "Manifest",
                f"Manifest missing: {MASTER_MANIFEST}",
                "Run: python -m src.utils.manifestor"
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
                    "Regenerate manifest"
                )
                return False, None
            
            # Verify splits
            splits = set(df['split'].unique())
            expected_splits = {'train', 'val', 'test'}
            if not expected_splits.issubset(splits):
                self.add_warning(
                    "Manifest",
                    f"Missing splits: {expected_splits - splits}"
                )
            
            self.add_pass(
                "Manifest",
                f"✓ {len(df)} subjects across {len(splits)} splits"
            )
            return True, df
            
        except Exception as e:
            self.add_issue(
                "Manifest",
                f"Error reading manifest: {e}",
                "Check manifest file"
            )
            return False, None
    
    def check_temporal_features(self) -> bool:
        """Validate temporal features."""
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
                    f"CRITICAL: {nan_pct:.1f}% NaN values!",
                    "Check feature extraction pipeline"
                )
                return False
            elif nan_pct > 5:
                self.add_warning(
                    "Features",
                    f"{nan_pct:.1f}% NaN values detected"
                )
            
            # Estimate ROI count
            expected_rois = len(feature_cols) // NUM_TEMPORAL_FEATURES if NUM_TEMPORAL_FEATURES > 0 else 170
            
            self.add_pass(
                "Features",
                f"✓ Temporal features: {len(df)} subjects, ~{expected_rois} ROIs"
            )
            return True
            
        except Exception as e:
            self.add_issue(
                "Features",
                f"Error reading features: {e}",
                "Regenerate temporal features"
            )
            return False
    
    def check_harmonization(self) -> bool:
        """Validate harmonized features."""
        logger.info("Checking harmonization...")
        
        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            self.add_issue(
                "Harmonization",
                f"Harmonized features missing: {NODE_ATTRIBUTES_HARMONIZED}",
                "Run: python -m src.features.safe_harmonization"
            )
            return False
        
        try:
            df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            feature_cols = [c for c in df.columns if c != 'subject_id']
            
            # Check for NaNs (should be ZERO)
            nan_count = df[feature_cols].isna().sum().sum()
            if nan_count > 0:
                self.add_issue(
                    "Harmonization",
                    f"CRITICAL: {nan_count} NaNs after harmonization!",
                    "Re-run safe_harmonization.py"
                )
                return False
            
            # Check for infinites
            inf_count = np.isinf(df[feature_cols].values).sum()
            if inf_count > 0:
                self.add_warning(
                    "Harmonization",
                    f"{inf_count} infinite values detected"
                )
            
            self.add_pass(
                "Harmonization",
                f"✓ Clean harmonized features: {len(df)} subjects"
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
        """Validate YOLO spatial features."""
        logger.info("Checking spatial features...")
        
        if not NODE_FEATURES_3D.exists():
            self.add_issue(
                "Spatial Features",
                f"3D features missing: {NODE_FEATURES_3D}",
                "Run: python -m src.features.extract_features"
            )
            return False
        
        try:
            df = pd.read_csv(NODE_FEATURES_3D)
            
            # Check detection completeness
            if 'node_count' in df.columns:
                complete = (df['node_count'] == NUM_LOBES).sum()
                survival_rate = complete / len(df) * 100
                
                self.results['yolo_survival_rate'] = survival_rate
                
                if survival_rate < 80:
                    self.add_warning(
                        "Spatial Features",
                        f"LOW survival rate: {survival_rate:.1f}% ({complete}/{len(df)})"
                    )
                else:
                    self.add_pass(
                        "Spatial Features",
                        f"✓ {complete}/{len(df)} subjects with all {NUM_LOBES} lobes ({survival_rate:.1f}%)"
                    )
            
            return True
            
        except Exception as e:
            self.add_issue(
                "Spatial Features",
                f"Error reading spatial features: {e}",
                "Re-run feature extraction"
            )
            return False
    
    def check_causal_graphs(self, manifest: Optional[pd.DataFrame] = None) -> Dict:
        """Validate causal graph construction."""
        logger.info("Checking causal graphs...")
        
        if not CAUSAL_GRAPHS_DIR.exists():
            self.add_issue(
                "Graphs",
                f"Graph directory missing: {CAUSAL_GRAPHS_DIR}",
                "Run: python -m src.features.construct_causal"
            )
            return {'available': 0, 'missing': 0, 'corrupted': 0}
        
        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        
        if not graph_files:
            self.add_issue(
                "Graphs",
                "No graph files found",
                "Run: python -m src.features.construct_causal"
            )
            return {'available': 0, 'missing': 0, 'corrupted': 0}
        
        stats = {
            'total_files': len(graph_files),
            'valid': 0,
            'corrupted': 0,
            'zero_edges': 0,
            'edge_counts': []
        }
        
        # Sample graphs for analysis
        sample_size = min(200, len(graph_files))
        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            try:
                graph_data = torch.load(graph_file)
                
                if 'adj' not in graph_data:
                    stats['corrupted'] += 1
                    continue
                
                adj = graph_data['adj']
                
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
        
        # Calculate statistics
        if stats['edge_counts']:
            stats['mean_edges'] = np.mean(stats['edge_counts'])
            stats['median_edges'] = np.median(stats['edge_counts'])
        
        # Report findings
        if stats['corrupted'] > sample_size * 0.05:
            self.add_warning(
                "Graphs",
                f"{stats['corrupted']}/{sample_size} graphs corrupted"
            )
        
        if stats['zero_edges'] > sample_size * 0.05:
            self.add_warning(
                "Graphs",
                f"{stats['zero_edges']}/{sample_size} graphs have zero edges"
            )
            self.add_warning(
                "Graphs",
                f"Consider lowering SPARSITY_QUANTILE from {SPARSITY_QUANTILE}"
            )
        
        if stats['valid'] > 0:
            self.add_pass(
                "Graphs",
                f"✓ {len(graph_files)} graphs available, "
                f"mean edges: {stats.get('mean_edges', 0):.1f}/{NUM_LOBES * NUM_LOBES}"
            )
        
        self.results['graph_stats'] = stats
        return stats
    
    def check_stratification(self, manifest: Optional[pd.DataFrame] = None) -> Dict:
        """Validate dataset stratification."""
        logger.info("Checking stratification...")
        
        if manifest is None:
            _, manifest = self.check_manifest()
            if manifest is None:
                return {}
        
        stats = {
            'total': len(manifest),
            'splits': {}
        }
        
        # Per-split analysis
        for split in ['train', 'val', 'test']:
            split_data = manifest[manifest['split'] == split]
            
            if len(split_data) == 0:
                continue
            
            dx_counts = split_data['DX_GROUP'].value_counts()
            
            stats['splits'][split] = {
                'total': len(split_data),
                'control': dx_counts.get(2, 0),
                'asd': dx_counts.get(1, 0),
                'num_sites': split_data['SITE_ID'].nunique()
            }
        
        # Check data leakage
        subject_counts = manifest.groupby('subject_id')['split'].nunique()
        leakage = (subject_counts > 1).sum()
        
        if leakage > 0:
            self.add_issue(
                "Stratification",
                f"DATA LEAKAGE: {leakage} subjects in multiple splits!",
                "Re-run split.py with proper stratification"
            )
        else:
            self.add_pass(
                "Stratification",
                f"✓ No data leakage, {len(stats['splits'])} splits"
            )
        
        self.results['stratification'] = stats
        return stats
    
    # VISUALIZATION (OPTIONAL)
    
    def visualize_results(self):
        """Generate validation visualizations."""
        if not self.visualize:
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Graph sparsity visualization
            if 'graph_stats' in self.results:
                stats = self.results['graph_stats']
                if stats.get('edge_counts'):
                    self._plot_graph_sparsity(stats['edge_counts'])
            
            # Stratification visualization
            if 'stratification' in self.results:
                manifest = pd.read_csv(MASTER_MANIFEST)
                self._plot_stratification(manifest)
            
            logger.info(f"✓ Visualizations saved to: {self.output_dir}")
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping visualizations")
    
    def _plot_graph_sparsity(self, edge_counts: List[int]):
        """Plot edge count distribution."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(edge_counts, bins=range(0, max(edge_counts) + 2), 
               edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(edge_counts), color='red', linestyle='--',
                  label=f'Mean: {np.mean(edge_counts):.1f}')
        ax.set_title('Graph Edge Count Distribution')
        ax.set_xlabel('Number of Edges')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'graph_sparsity.png', dpi=150)
        plt.close()
    
    def _plot_stratification(self, manifest: pd.DataFrame):
        """Plot stratification analysis."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Split distribution
        ax = axes[0]
        split_dx = manifest.groupby(['split', 'DX_GROUP']).size().unstack(fill_value=0)
        split_dx.columns = ['ASD', 'Control']
        split_dx.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
        ax.set_title('Distribution by Split')
        ax.set_ylabel('Number of Subjects')
        ax.legend(title='Diagnosis')
        
        # Class balance
        ax = axes[1]
        split_ratios = []
        for split in ['train', 'val', 'test']:
            split_data = manifest[manifest['split'] == split]
            asd = (split_data['DX_GROUP'] == 1).sum()
            ratio = asd / len(split_data) if len(split_data) > 0 else 0
            split_ratios.append(ratio)
        
        ax.bar(['Train', 'Val', 'Test'], split_ratios)
        ax.axhline(0.5, color='black', linestyle='--', label='Perfect Balance')
        ax.set_title('ASD Ratio per Split')
        ax.set_ylabel('Proportion')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'stratification.png', dpi=150)
        plt.close()
    
    # REPORTING
    
    def generate_report(self) -> bool:
        """Generate and display final report."""
        print("\n" + "="*70)
        print("NEURO-CXG PIPELINE HEALTH CHECK")
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
            print("\n⚠️  PIPELINE HAS CRITICAL ISSUES")
            return False
        elif self.warnings:
            print("\n✓ Pipeline functional with warnings")
            return True
        else:
            print("\n✅ PIPELINE FULLY HEALTHY")
            return True
    
    # MAIN EXECUTION
    
    def run_full_check(self) -> bool:
        """Execute complete health check."""
        # Core checks
        self.check_atlas()
        self.check_lobe_mapping()
        manifest_ok, manifest = self.check_manifest()
        
        # Feature checks
        if manifest_ok:
            self.check_temporal_features()
            self.check_harmonization()
            self.check_spatial_features()
            self.check_causal_graphs(manifest)
            self.check_stratification(manifest)
        
        # Optional visualizations
        self.visualize_results()
        
        # Generate report
        return self.generate_report()


# CLI

def main():
    """Command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline health check"
    )
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--atlas',
        action='store_true',
        help='Check only atlas'
    )
    parser.add_argument(
        '--features',
        action='store_true',
        help='Check only features'
    )
    parser.add_argument(
        '--graphs',
        action='store_true',
        help='Check only graphs'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with error code if issues found'
    )
    
    args = parser.parse_args()
    
    checker = PipelineHealthCheck(visualize=args.visualize)
    
    # Selective checks
    if args.atlas or args.features or args.graphs:
        if args.atlas:
            checker.check_atlas()
            checker.check_lobe_mapping()
        if args.features:
            checker.check_temporal_features()
            checker.check_harmonization()
            checker.check_spatial_features()
        if args.graphs:
            checker.check_causal_graphs()
    else:
        # Full check
        is_healthy = checker.run_full_check()
        
        if args.strict:
            sys.exit(0 if is_healthy else 1)
    
    # Always exit 0 in non-strict mode (informational)
    sys.exit(0)


if __name__ == "__main__":
    main()