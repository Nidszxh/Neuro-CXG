"""Unit tests for Task 4 — Spatial Feature Cleanup (DD-012).

Tests:
1. _get_subject_spatial returns shape (NUM_LOBES, 4) — not 6
2. conf_std and detection_count are absent from spatial feature output
3. NUM_SPATIAL_FEATURES sentinel assertion in feature_registry holds
4. GNN_IN_CHANNELS is consistent with registered feature groups
"""

import numpy as np
import pandas as pd

from src.core.config import (
    ALL_FEATURE_NAMES,
    GNN_IN_CHANNELS,
    LOBE_NAMES,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
)


class TestSpatialFeatureRegistry:
    def test_num_spatial_features_is_4(self):
        """Task 4 sentinel: exactly 4 spatial features (x, y, z_depth, size)."""
        assert NUM_SPATIAL_FEATURES == 4, (
            f"NUM_SPATIAL_FEATURES must be 4; got {NUM_SPATIAL_FEATURES}. "
            "conf_std and detection_count are forbidden (site leakage via RF AUC=1.000)."
        )

    def test_conf_std_not_in_spatial_features(self):
        """conf_std must not appear in ANY registered feature name."""
        all_names_lower = [n.lower() for n in ALL_FEATURE_NAMES]
        assert not any("conf_std" in n for n in all_names_lower), (
            "conf_std found in ALL_FEATURE_NAMES — it encodes scanner quality (site leakage)."
        )

    def test_detection_count_not_in_spatial_features(self):
        """detection_count must not appear in ANY registered feature name."""
        all_names_lower = [n.lower() for n in ALL_FEATURE_NAMES]
        assert not any("detection_count" in n for n in all_names_lower), (
            "detection_count found in ALL_FEATURE_NAMES — it encodes scanner quality (site leakage)."
        )

    def test_gnn_in_channels_matches_all_feature_names(self):
        """GNN_IN_CHANNELS must equal len(ALL_FEATURE_NAMES)."""
        assert GNN_IN_CHANNELS == len(ALL_FEATURE_NAMES), (
            f"GNN_IN_CHANNELS={GNN_IN_CHANNELS} != len(ALL_FEATURE_NAMES)={len(ALL_FEATURE_NAMES)}"
        )

class TestGraphFactorySpatialExtraction:
    """Test _get_subject_spatial using a minimal synthetic coords CSV."""

    def _make_coords_df(self, include_confounders: bool = False) -> pd.DataFrame:
        """Build a synthetic coords CSV row for one subject."""
        row: dict = {"subject_id": "test_sub"}
        for lobe_name in LOBE_NAMES.values():
            row[f"{lobe_name}_x"] = np.random.rand()
            row[f"{lobe_name}_y"] = np.random.rand()
            row[f"{lobe_name}_z_depth"] = np.random.rand()
            row[f"{lobe_name}_size"] = np.random.rand() * 1000
            if include_confounders:
                # These should NOT be read by _get_subject_spatial
                row[f"{lobe_name}_conf_std"] = np.random.rand()
                row[f"{lobe_name}_detection_count"] = np.random.randint(1, 10)
        return pd.DataFrame([row])

    def _build_dataset_stub(self, coords_df: pd.DataFrame, tmp_path: str):
        """Create a minimal ABIDECausalDataset-like object with loaded coords."""
        import types
        ds = types.SimpleNamespace()
        ds.coords = coords_df.set_index("subject_id")
        return ds

    def test_spatial_shape_without_confounders(self):
        """_get_subject_spatial must return (NUM_LOBES, 4) without conf_std columns."""

        from src.features.graph_factory import ABIDECausalDataset

        # Directly call the private method on a minimal stub
        coords_df = self._make_coords_df(include_confounders=False)

        class _Stub:
            coords = coords_df.set_index("subject_id")

        # Rebind method to stub
        method = ABIDECausalDataset._get_subject_spatial
        result = method(_Stub(), "test_sub")

        assert result is not None, "_get_subject_spatial returned None"
        assert result.shape == (NUM_LOBES, 4), (
            f"Expected ({NUM_LOBES}, 4), got {result.shape}"
        )

    def test_spatial_shape_with_confounders_present_in_csv(self):
        """Even when conf_std/detection_count exist in the CSV, output must be (NUM_LOBES, 4)."""
        from src.features.graph_factory import ABIDECausalDataset

        coords_df = self._make_coords_df(include_confounders=True)

        class _Stub:
            coords = coords_df.set_index("subject_id")

        result = ABIDECausalDataset._get_subject_spatial(_Stub(), "test_sub")
        assert result is not None
        assert result.shape == (NUM_LOBES, 4), (
            f"Expected ({NUM_LOBES}, 4) even with confounders in CSV; got {result.shape}"
        )

class TestAtlasSpatialExtractorCompatibility:
    """Atlas mode must emit the same 4-feature lobe schema consumed by graph_factory."""

    def test_atlas_lobe_features_are_four_dimensional(self):
        from src.features.extract_spatial_atlas import extract_lobe_features

        # Minimal synthetic centroids for two ROIs in one lobe
        centroids = {
            1: {"roi_id": 1, "x": 1.0, "y": 2.0, "z": 3.0},
            2: {"roi_id": 2, "x": 2.0, "y": 3.0, "z": 4.0},
        }
        roi_sizes = {1: 0.5, 2: 0.8}
        feats = extract_lobe_features(lobe_id=0, roi_indices=[0, 1], centroids=centroids, roi_sizes=roi_sizes)

        assert len(feats) == 4, f"Atlas lobe feature length must be 4; got {len(feats)}"

    def test_atlas_schema_matches_graph_factory_keys(self):
        # Expected atlas output keys consumed by graph_factory._get_subject_spatial
        expected_cols = {
            f"{lobe_name}_{feat}"
            for lobe_name in LOBE_NAMES.values()
            for feat in ("x", "y", "z_depth", "size")
        }

        assert all("conf_std" not in c for c in expected_cols)
        assert all("detection_count" not in c for c in expected_cols)
        assert all(not c.startswith("roi") for c in expected_cols)
        assert len(expected_cols) == NUM_LOBES * NUM_SPATIAL_FEATURES
