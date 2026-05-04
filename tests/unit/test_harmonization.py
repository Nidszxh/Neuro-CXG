"""
Unit tests for src/features/fold_safe_harmonization.py.

Guards against two high-impact regressions:
  1. DX_GROUP accidentally removed from ComBat covariates (would strip diagnostic signal).
  2. Spatial feature harmonization running on wrong (full-dataset) statistics.

Run: pytest tests/unit/test_harmonization.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_manifest(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Create a minimal manifest with required columns."""
    rng = np.random.default_rng(seed)
    sites = ["CMU", "CALTECH", "KKI", "LEUVEN_1"]
    rows = []
    for i in range(n):
        split = "train" if i < 14 else ("val" if i < 17 else "test")
        rows.append(
            {
                "subject_id": f"SUB_{i:04d}",
                "split": split,
                "SITE_ID": sites[i % len(sites)],
                "AGE_AT_SCAN": float(20 + rng.integers(0, 30)),
                "SEX": int(rng.integers(1, 3)),
                "DX_GROUP": int(1 + (i % 2)),  # alternating Control/ASD
                "FIQ": float(100 + rng.standard_normal()),
                "TR": 2.0,
                "cv_fold": i % 5,
            }
        )
    return pd.DataFrame(rows)


def _make_temporal_features(manifest: pd.DataFrame, n_feat: int = None) -> pd.DataFrame:
    """Create a mock temporal features DataFrame aligned to manifest subjects."""
    rng = np.random.default_rng(42)
    from src.core.config import FEATURE_GROUPS, LOBE_NAMES
    feat_names = FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["frequency"]
    if n_feat is None:
        n_feat = len(feat_names)
    # Pad or trim to n_feat columns
    feat_names = (feat_names * ((n_feat // len(feat_names)) + 1))[:n_feat]
    cols = {}
    for lobe in LOBE_NAMES.values():
        for feat in feat_names:
            cols[f"{lobe}_{feat}"] = rng.standard_normal(len(manifest))
    df = pd.DataFrame(cols)
    df.insert(0, "subject_id", manifest["subject_id"].values)
    return df


def _make_spatial_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """Create a mock node_features_3d-style DataFrame with per-site conf_std bias."""
    rng = np.random.default_rng(0)
    from src.core.config import LOBE_NAMES
    rows = []
    site_bias = {"CMU": 2.0, "CALTECH": 0.5, "KKI": 1.0, "LEUVEN_1": 3.0}
    for _, row in manifest.iterrows():
        rec = {"subject_id": row["subject_id"], "spatial_complete": 1, "SITE_ID": row["SITE_ID"]}
        bias = site_bias.get(row["SITE_ID"], 1.0)
        for lobe in LOBE_NAMES.values():
            rec[f"{lobe}_x"]               = float(rng.uniform(0, 100))
            rec[f"{lobe}_y"]               = float(rng.uniform(0, 120))
            rec[f"{lobe}_z_depth"]         = float(rng.uniform(0, 80))
            rec[f"{lobe}_size"]            = float(rng.uniform(50, 500))
            # conf_std has an injected site-level offset so the test can verify removal
            rec[f"{lobe}_conf_std"]        = float(rng.uniform(0, 0.1)) + bias
            rec[f"{lobe}_detection_count"] = float(int(5 + rng.integers(0, 3)) * (bias > 1))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrepareCovariateDXGroup:
    """_prepare_covariates must include DX_GROUP as a protected covariate."""

    def test_dx_group_present(self):
        from src.features.fold_safe_harmonization import _prepare_covariates

        manifest = _make_manifest(n=10)
        feats = pd.DataFrame({"subject_id": manifest["subject_id"]})
        cov = _prepare_covariates(manifest, feats)

        assert "DX_GROUP" in cov.columns, (
            "DX_GROUP must be in the covariate matrix passed to ComBat so that "
            "diagnostic variance is protected, not removed as a batch effect."
        )

    def test_site_column_renamed_to_SITE(self):
        """neuroHarmonize requires the batch column to be named 'SITE'."""
        from src.features.fold_safe_harmonization import _prepare_covariates

        manifest = _make_manifest(n=10)
        feats = pd.DataFrame({"subject_id": manifest["subject_id"]})
        cov = _prepare_covariates(manifest, feats)

        assert "SITE" in cov.columns, "SITE_ID must be renamed to 'SITE' for neuroHarmonize"
        assert "SITE_ID" not in cov.columns

    def test_no_nans_in_covariates(self):
        """NaN covariates cause ComBat to crash — verify imputation is applied."""
        from src.features.fold_safe_harmonization import _prepare_covariates

        manifest = _make_manifest(n=12)
        manifest.loc[0, "AGE_AT_SCAN"] = float("nan")
        manifest.loc[1, "SEX"] = float("nan")
        feats = pd.DataFrame({"subject_id": manifest["subject_id"]})
        cov = _prepare_covariates(manifest, feats)

        assert not cov.isna().any().any(), "Covariate matrix must not contain any NaN values"


class TestOutlierClipFoldSafe:
    """Outlier bounds must be derived from train data only."""

    def test_clip_bounds_from_train_only(self):
        """Injecting a large outlier only in the test set must not shift the clip boundary."""
        from src.features.fold_safe_harmonization import (
            _outlier_clip_apply,
            _outlier_clip_fit,
        )

        rng = np.random.default_rng(7)
        n_train = 50
        train_df = pd.DataFrame({"col_a": rng.standard_normal(n_train)})
        # Test set has one extreme outlier — this must not affect the train-derived bounds.
        test_df = pd.DataFrame({"col_a": np.append(rng.standard_normal(20), [1_000.0])})

        _, bounds = _outlier_clip_fit(train_df)
        train_clipped, _ = _outlier_clip_fit(train_df)
        test_clipped = _outlier_clip_apply(test_df, bounds)

        # The outlier (1000) must be clipped down to the train upper bound.
        assert test_clipped["col_a"].max() <= bounds["upper"]["col_a"] + 1e-6, (
            "Test outlier should be clipped to train-derived boundary, not its own statistics"
        )
        # The extreme outlier must not survive clipping.
        assert test_clipped["col_a"].max() < 100.0, "Extreme test outlier (1000) survived clipping"

    def test_harmonize_cv_safe_fold_uses_train_only_bounds(self, tmp_path):
        """harmonize_cv_safe_fold() must not use val/test statistics for clipping."""
        from src.features.fold_safe_harmonization import harmonize_cv_safe_fold

        manifest = _make_manifest(n=30)
        features = _make_temporal_features(manifest)

        # Inject a large outlier only in the held-out test split.
        test_subjects = set(manifest.loc[manifest["split"] == "test", "subject_id"])
        feature_cols = [c for c in features.columns if c != "subject_id"]
        target_col = feature_cols[0]
        features.loc[features["subject_id"].isin(test_subjects), target_col] = 999.0

        out_path = tmp_path / "harmonized.csv"
        folds = harmonize_cv_safe_fold(
            features,
            manifest,
            full_output_path=out_path,
        )

        assert folds, "harmonize_cv_safe_fold() returned no folds"
        assert out_path.exists(), "Expected harmonized output CSV to be written"

        result = pd.read_csv(out_path)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0, "No numeric columns found in harmonized output"

        # If clipping is train-only and active, the injected 999.0 outlier should never survive.
        assert result[numeric_cols].to_numpy().max() < 100.0, (
            "Test-set outlier survived harmonization — train-only clipping may not be applied"
        )


class TestSpatialHarmonization:
    """harmonize_spatial_features must reduce site-level variance in conf_std."""

    def test_site_variance_reduced(self, tmp_path):
        """After harmonization, inter-site variance in conf_std should decrease."""
        from scipy.stats import kruskal

        from src.features.fold_safe_harmonization import harmonize_spatial_features

        manifest = _make_manifest(n=40, seed=1)
        spatial  = _make_spatial_features(manifest)
        output   = tmp_path / "node_features_3d_harmonized.csv"

        harmonized = harmonize_spatial_features(spatial, manifest, output_path=output)

        assert output.exists(), "Harmonized spatial CSV must be written to output_path"

        from src.core.config import LOBE_NAMES
        col = f"{LOBE_NAMES[0]}_conf_std"  # first lobe conf_std
        if col not in harmonized.columns:
            pytest.skip(f"Column {col} not in harmonized output — check LOBE_NAMES")

        # harmonize_spatial_features preserves SITE_ID from spatial_df; use it directly.
        site_col = "SITE_ID" if "SITE_ID" in harmonized.columns else None
        if site_col is None:
            harmonized_with_site = harmonized.merge(manifest[["subject_id", "SITE_ID"]], on="subject_id")
            site_col = "SITE_ID"
        else:
            harmonized_with_site = harmonized

        before_merged = spatial.merge(manifest[["subject_id", "SITE_ID"]], on="subject_id", suffixes=("_sp", "_m"))
        site_col_before = "SITE_ID_m" if "SITE_ID_m" in before_merged.columns else "SITE_ID"

        groups_before = [g[col].values for _, g in before_merged.groupby(site_col_before) if len(g) > 1]
        groups_after  = [g[col].values for _, g in harmonized_with_site.groupby(site_col)  if len(g) > 1]

        if len(groups_before) < 2 or len(groups_after) < 2:
            pytest.skip("Not enough multi-subject sites for Kruskal-Wallis")

        _, p_before = kruskal(*groups_before)
        _, p_after  = kruskal(*groups_after)

        # After harmonization, p-value should be larger (less site differentiation).
        assert p_after > p_before or p_after > 0.05, (
            f"Harmonization did not reduce site effect: p_before={p_before:.4f}, p_after={p_after:.4f}"
        )

    def test_anatomical_columns_unchanged(self, tmp_path):
        """x, y, z_depth, size must be identical before and after harmonization."""
        from src.core.config import LOBE_NAMES
        from src.features.fold_safe_harmonization import harmonize_spatial_features

        manifest  = _make_manifest(n=24, seed=2)
        spatial   = _make_spatial_features(manifest)
        output    = tmp_path / "node_features_3d_harmonized.csv"
        harmonized = harmonize_spatial_features(spatial, manifest, output_path=output)

        for lobe in LOBE_NAMES.values():
            for suffix in ("_x", "_y", "_z_depth", "_size"):
                col = f"{lobe}{suffix}"
                if col in spatial.columns and col in harmonized.columns:
                    pd.testing.assert_series_equal(
                        spatial[col].reset_index(drop=True),
                        harmonized[col].reset_index(drop=True),
                        check_names=False,
                        rtol=1e-5,
                        obj=f"Column {col} should be unchanged after spatial harmonization",
                    )

    def test_same_subjects_in_output(self, tmp_path):
        """Output must contain the same set of subjects as the input."""
        from src.features.fold_safe_harmonization import harmonize_spatial_features

        manifest  = _make_manifest(n=20, seed=3)
        spatial   = _make_spatial_features(manifest)
        output    = tmp_path / "node_features_3d_harmonized.csv"
        harmonized = harmonize_spatial_features(spatial, manifest, output_path=output)

        assert set(harmonized["subject_id"]) == set(spatial["subject_id"])
