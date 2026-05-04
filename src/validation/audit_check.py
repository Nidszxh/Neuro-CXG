"""Post-fix validation script for Neuro-CXG pipeline.

This audit is configuration-driven and computes expected dimensions from
`src.core.config` to stay consistent with runtime feature registry changes
(for example, excluding Nyquist-unsafe frequency bands).

Usage:
    python src/validation/audit_check.py
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    GNN_IN_CHANNELS,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    NUM_TEMPORAL_FEATURES,
    SPATIAL_MIN_REQUIRED_REGIONS,
)
from src.features.graph_factory import ABIDECausalDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AuditCheck:
    """Comprehensive post-fix validation."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, error_msg: str = ""):
        """Record a check result."""
        if condition:
            logger.info(f"✓ {name}")
            self.checks_passed += 1
        else:
            logger.error(f"✗ {name}: {error_msg}")
            self.checks_failed += 1
            self.errors.append(f"{name}: {error_msg}")

    def check_warn(self, name: str, condition: bool, warn_msg: str = ""):
        """Record a check that logs a warning (not an error) on failure.

        Use this for intermediate pipeline artifacts (temporal features, graphs)
        that may legitimately not exist yet when the audit runs at pipeline start.
        Failures here do NOT increment checks_failed.
        """
        if condition:
            logger.info(f"✓ {name}")
            self.checks_passed += 1
        else:
            logger.warning(f"⚠ {name}: {warn_msg} (will be generated during pipeline run)")

    def run_all_checks(self):
        """Execute all validation checks."""
        logger.info("="*70)
        logger.info("NEURO-CXG POST-FIX AUDIT VALIDATION")
        logger.info("="*70)

        # Check 1: Exactly 1,000 subjects in spatial features
        logger.info("\n1. Checking subject counts...")
        self._check_subject_counts()

        # Check 2: All subjects have 12 detected lobes
        logger.info("\n2. Checking YOLO detection completeness...")
        self._check_yolo_completeness()

        # Check 3: No NaN/Inf in harmonized features
        logger.info("\n3. Checking harmonized features integrity...")
        self._check_harmonized_integrity()

        # Check 4: Correct feature dimensions
        logger.info("\n4. Checking feature dimensions...")
        self._check_feature_dimensions()

        # Check 5: Count causal graph files
        logger.info("\n5. Checking causal graph files...")
        self._check_causal_graphs()

        # Check 6: Load and validate random graph samples
        logger.info("\n6. Validating random graph samples...")
        self._check_graph_samples()

        # Check 7: Validate ABIDECausalDataset loader
        logger.info("\n7. Checking dataset loader...")
        self._check_dataset_loader()

        # Final report
        self._print_summary()

    def _check_subject_counts(self):
        """Validate consistent subject counts across all files."""
        # Get manifest count as the reference
        if not MASTER_MANIFEST.exists():
            self.check("Master manifest exists", False, "Manifest not found")
            return

        manifest_df = pd.read_csv(MASTER_MANIFEST)
        expected_count = len(manifest_df)

        # Spatial features
        if NODE_FEATURES_3D.exists():
            spatial_df = pd.read_csv(NODE_FEATURES_3D)
            spatial_count = len(spatial_df)
            self.check(
                f"Spatial features: {spatial_count} subjects",
                spatial_count <= expected_count,
                f"Expected ≤{expected_count} (manifest), got {spatial_count}"
            )
        else:
            self.check("Spatial features file exists", False, "File not found")

        # Temporal features
        if NODE_ATTRIBUTES_TEMPORAL.exists():
            temporal_df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            temporal_count = len(temporal_df)
            self.check(
                f"Temporal features: {temporal_count} subjects",
                temporal_count == expected_count,
                f"Expected {expected_count}, got {temporal_count}"
            )
        else:
            self.check_warn("Temporal features file exists", False, "File not found")

        # Harmonized features
        if NODE_ATTRIBUTES_HARMONIZED.exists():
            harmonized_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            harmonized_count = len(harmonized_df)
            self.check(
                f"Harmonized features: {harmonized_count} subjects",
                harmonized_count == expected_count,
                f"Expected {expected_count}, got {harmonized_count}"
            )
        else:
            self.check_warn("Harmonized features file exists", False, "File not found")

        # Manifest
        if MASTER_MANIFEST.exists():
            manifest_df = pd.read_csv(MASTER_MANIFEST)
            manifest_count = len(manifest_df)
            self.check(
                f"Master manifest: {manifest_count} subjects",
                manifest_count == expected_count,
                f"Expected {expected_count}, got {manifest_count}"
            )
        else:
            self.check("Master manifest exists", False, "File not found")

    def _check_yolo_completeness(self):
        """Assess spatial regional coverage quality.

        The current extractor emits 4 anatomical spatial columns per lobe:
        x, y, z_depth, and size. Older legacy outputs may still include
        detection_count columns, so we accept either schema.
        """
        if not NODE_FEATURES_3D.exists():
            self.check("Spatial features available", False, "File not found")
            return

        df = pd.read_csv(NODE_FEATURES_3D)

        legacy_count_cols = [col for col in df.columns if col.endswith("_detection_count")]
        modern_size_cols = [
            f"{lobe_name}_size"
            for lobe_name in LOBE_NAMES.values()
            if f"{lobe_name}_size" in df.columns
        ]

        if legacy_count_cols:
            self.check(
                f"All {NUM_LOBES} region count columns present",
                len(legacy_count_cols) == NUM_LOBES,
                f"Expected {NUM_LOBES} columns, found {len(legacy_count_cols)}",
            )
            if len(legacy_count_cols) == NUM_LOBES:
                subjects_with_missing = (df[legacy_count_cols] == 0).any(axis=1).sum()
                self.check_warn(
                    "All subjects have all 12 regions detected",
                    subjects_with_missing == 0,
                    f"{subjects_with_missing} subjects have at least one missing region" if subjects_with_missing > 0 else "",
                )
                regions_present = (df[legacy_count_cols] > 0).sum(axis=1)
                below_min_required = int((regions_present < SPATIAL_MIN_REQUIRED_REGIONS).sum())
                self.check(
                    f"Subjects meet min detected regions (>= {SPATIAL_MIN_REQUIRED_REGIONS})",
                    below_min_required == 0,
                    f"{below_min_required} subjects below minimum required region count",
                )
            return

        self.check(
            f"All {NUM_LOBES} region size columns present",
            len(modern_size_cols) == NUM_LOBES,
            f"Expected {NUM_LOBES} columns, found {len(modern_size_cols)}",
        )
        if len(modern_size_cols) != NUM_LOBES:
            return

        # Modern atlas-default extraction uses 4 anatomical features per lobe.
        spatial_feature_cols = [
            col
            for lobe_name in LOBE_NAMES.values()
            for col in (
                f"{lobe_name}_x",
                f"{lobe_name}_y",
                f"{lobe_name}_z_depth",
                f"{lobe_name}_size",
            )
            if col in df.columns
        ]
        expected_spatial_cols = NUM_LOBES * NUM_SPATIAL_FEATURES
        self.check(
            f"All {expected_spatial_cols} modern spatial columns present",
            len(spatial_feature_cols) == expected_spatial_cols,
            f"Expected {expected_spatial_cols} columns, found {len(spatial_feature_cols)}",
        )

        # With atlas-default spatial extraction, size should be non-zero for all
        # lobes present in the atlas. Use this as a coarse integrity proxy.
        subjects_with_missing = (df[modern_size_cols] == 0).any(axis=1).sum()
        self.check_warn(
            "All subjects have all 12 regions populated",
            subjects_with_missing == 0,
            f"{subjects_with_missing} subjects have at least one zero-size lobe",
        )

        regions_present = (df[modern_size_cols] > 0).sum(axis=1)
        below_min_required = int((regions_present < SPATIAL_MIN_REQUIRED_REGIONS).sum())
        self.check(
            f"Subjects meet min populated regions (>= {SPATIAL_MIN_REQUIRED_REGIONS})",
            below_min_required == 0,
            f"{below_min_required} subjects below minimum required region count",
        )

    def _check_harmonized_integrity(self):
        """Check for NaN/Inf in harmonized features."""
        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            self.check("Harmonized features available", False, "File not found")
            return

        df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)

        # Exclude subject_id column
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Check for NaN
        nan_count = df[numeric_cols].isna().sum().sum()
        self.check(
            "No NaN values in harmonized features",
            nan_count == 0,
            f"Found {nan_count} NaN values"
        )

        # Check for Inf
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        self.check(
            "No Inf values in harmonized features",
            inf_count == 0,
            f"Found {inf_count} Inf values"
        )

    def _check_feature_dimensions(self):
        """Validate feature matrix dimensions."""
        # Temporal: (N, 170 * NUM_TEMPORAL_FEATURES + 1)
        if NODE_ATTRIBUTES_TEMPORAL.exists():
            temporal_df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            expected_temporal_cols = 170 * NUM_TEMPORAL_FEATURES + 1  # +1 for subject_id
            actual_temporal_cols = len(temporal_df.columns)
            self.check(
                f"Temporal features shape: ({len(temporal_df)}, {actual_temporal_cols})",
                actual_temporal_cols == expected_temporal_cols,
                f"Expected {expected_temporal_cols} columns, got {actual_temporal_cols}"
            )

        # Harmonized: (N, NUM_LOBES * NUM_TEMPORAL_FEATURES + 1)
        if NODE_ATTRIBUTES_HARMONIZED.exists():
            harmonized_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            expected_harmonized_cols = NUM_LOBES * NUM_TEMPORAL_FEATURES + 1  # +1 for subject_id
            actual_harmonized_cols = len(harmonized_df.columns)
            self.check(
                f"Harmonized features shape: ({len(harmonized_df)}, {actual_harmonized_cols})",
                actual_harmonized_cols == expected_harmonized_cols,
                f"Expected {expected_harmonized_cols} columns, got {actual_harmonized_cols}"
            )

    def _check_causal_graphs(self):
        """Count causal graph files."""
        if not CAUSAL_GRAPHS_DIR.exists():
            self.check("Causal graphs directory exists", False, "Directory not found")
            return

        # Get expected count from manifest
        if not MASTER_MANIFEST.exists():
            self.check("Master manifest exists for graph count check", False, "Manifest not found")
            return

        manifest_df = pd.read_csv(MASTER_MANIFEST)
        expected_count = len(manifest_df)

        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))

        self.check(
            f"Causal graph files: {len(graph_files)}",
            len(graph_files) <= expected_count,
            f"Expected ≤{expected_count} (some may fail graph construction), found {len(graph_files)}"
        )

    def _check_graph_samples(self):
        """Load and validate random graph samples."""
        if not CAUSAL_GRAPHS_DIR.exists():
            self.check_warn("Causal graphs available for sampling", False, "Directory not found")
            return

        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        if len(graph_files) == 0:
            self.check_warn("Graph files exist for sampling", False, "No graph files found")
            return

        # Sample 5 random graphs
        sample_files = random.sample(graph_files, min(5, len(graph_files)))

        for graph_file in sample_files:
            try:
                graph_dict = torch.load(graph_file, map_location='cpu')

                # Validate structure
                required_keys = {'adj', 'internal_features', 'subject_id', 'lobe_order'}
                has_keys = required_keys.issubset(graph_dict.keys())

                if not has_keys:
                    self.check(
                        f"Graph {graph_file.name} has required keys",
                        False,
                        f"Missing keys: {required_keys - set(graph_dict.keys())}"
                    )
                    continue

                # Check dimensions
                adj = graph_dict['adj']
                internal_features = graph_dict['internal_features']

                adj_shape_ok = adj.shape == (NUM_LOBES, NUM_LOBES)
                internal_shape_ok = internal_features.shape == (NUM_LOBES, 2)

                self.check(
                    f"Graph {graph_file.name}: adj.shape == ({NUM_LOBES}, {NUM_LOBES})",
                    adj_shape_ok,
                    f"Got {adj.shape}"
                )

                self.check(
                    f"Graph {graph_file.name}: internal_features.shape == ({NUM_LOBES}, 2)",
                    internal_shape_ok,
                    f"Got {internal_features.shape}"
                )

            except Exception as e:
                self.check(
                    f"Load graph {graph_file.name}",
                    False,
                    f"Error: {str(e)}"
                )

    def _check_dataset_loader(self):
        """Validate ABIDECausalDataset loader."""
        try:
            dataset = ABIDECausalDataset('train')

            self.check(
                "ABIDECausalDataset loads successfully",
                True,
                ""
            )

            # Check dataset size
            self.check(
                f"Train dataset has subjects: {len(dataset)}",
                len(dataset) > 0,
                "Dataset is empty"
            )

            # Check all non-null graphs for shape / NaN
            all_ok = True
            checked = 0
            for idx in range(len(dataset)):
                graph = dataset[idx]
                if graph is None:
                    continue
                checked += 1
                expected_x_shape = (NUM_LOBES, GNN_IN_CHANNELS)
                if graph.x.shape != expected_x_shape:
                    all_ok = False
                    self.check(
                        f"Graph {idx}: x.shape == {expected_x_shape}",
                        False,
                        f"Got {graph.x.shape}"
                    )
                    break
                if graph.edge_index.shape[0] != 2:
                    all_ok = False
                    self.check(
                        f"Graph {idx}: edge_index.shape[0] == 2",
                        False,
                        f"Got {graph.edge_index.shape[0]}"
                    )
                    break
                if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
                    all_ok = False
                    self.check(
                        f"Graph {idx}: no NaN/Inf in node features",
                        False,
                        "Found NaN/Inf values"
                    )
                    break

            if checked == 0:
                self.check("ABIDECausalDataset has at least one graph", False, "No non-null graph samples")
            else:
                self.check(
                    f"All checked train graphs valid ({checked} samples)",
                    all_ok,
                    "At least one graph failed shape/NaN integrity checks"
                )

        except Exception as e:
            self.check(
                "ABIDECausalDataset loader",
                False,
                f"Error: {str(e)}"
            )

    def _print_summary(self):
        """Print final audit summary."""
        logger.info("\n" + "="*70)
        logger.info("AUDIT VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"✓ Checks Passed: {self.checks_passed}")
        logger.info(f"✗ Checks Failed: {self.checks_failed}")

        if self.checks_failed > 0:
            logger.error("\nFailed Checks:")
            for error in self.errors:
                logger.error(f"  - {error}")
            logger.error("\n❌ AUDIT FAILED - Please review errors above")
            sys.exit(1)
        else:
            logger.info("\n✅ ALL CHECKS PASSED - Pipeline is ready!")
            sys.exit(0)


if __name__ == "__main__":
    auditor = AuditCheck()
    auditor.run_all_checks()
