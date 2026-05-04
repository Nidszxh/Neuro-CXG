"""
Unit tests for critical feature-ordering correctness properties.

Tests guard against the P0/P1 bugs identified in the March 2026 audit:

- FEATURE_TYPES in fold_safe_harmonization.py must use interleaved frequency
  ordering (delta_power, delta_peak, theta_power, theta_peak, ...) to match
  config.FEATURE_GROUPS['frequency'] — the ordering determines the column
  position in the harmonized CSV, which graph_factory.py reads positionally.

- config.FEATURE_GROUPS['temporal'][3] must be 'kurt' (matching the actual
  CSV column names produced by extract_temporal.py).

- aggregate_to_lobes() must produce columns in the correct canonical order
  so that positional loading in graph_factory._get_subject_temporal() gets
  the right feature at each index.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config import FEATURE_GROUPS, LOBE_NAMES, NUM_LOBES
from src.features.fold_safe_harmonization import FEATURE_TYPES, aggregate_to_lobes

# ──────────────────────────────────────────────────────────────────────────────
# 1. FEATURE_TYPES ordering matches config source of truth
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureTypesOrdering:
    """FEATURE_TYPES must be interleaved (delta_power, delta_peak, …) not grouped."""

    def test_temporal_names_match_config(self):
        """First 8 entries of FEATURE_TYPES must equal config.FEATURE_GROUPS['temporal']."""
        expected = FEATURE_GROUPS['temporal']  # ["mean","std","skew","kurt", ...]
        actual = FEATURE_TYPES[:len(expected)]
        assert actual == expected, (
            f"FEATURE_TYPES temporal prefix does not match config.\n"
            f"Expected: {expected}\n"
            f"   Got:   {actual}"
        )

    def test_frequency_names_interleaved(self):
        """
        Frequency features after the temporal prefix must be in interleaved order:
        delta_power, delta_peak, theta_power, theta_peak, …
        NOT grouped (all powers then all peaks).
        """
        expected_freq = FEATURE_GROUPS['frequency']
        n_temporal = len(FEATURE_GROUPS['temporal'])
        actual_freq = FEATURE_TYPES[n_temporal: n_temporal + len(expected_freq)]
        assert actual_freq == expected_freq, (
            f"FEATURE_TYPES frequency section is NOT interleaved.\n"
            f"Expected: {expected_freq}\n"
            f"   Got:   {actual_freq}"
        )

    def test_no_grouped_powers_block(self):
        """Reject old bug: 5 consecutive *_power entries."""
        power_entries = [f for f in FEATURE_TYPES if f.endswith("_power")]
        indices = [FEATURE_TYPES.index(p) for p in power_entries]
        # If they were grouped, their indices would be consecutive
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i]
            assert gap > 1, (
                f"Consecutive *_power features detected at FEATURE_TYPES positions "
                f"{indices[i]} and {indices[i+1]} — frequency features are still grouped!"
            )

    def test_no_grouped_peaks_block(self):
        """Reject old bug: 5 consecutive *_peak entries."""
        peak_entries = [f for f in FEATURE_TYPES if f.endswith("_peak")]
        indices = [FEATURE_TYPES.index(p) for p in peak_entries]
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i]
            assert gap > 1, (
                f"Consecutive *_peak features detected at FEATURE_TYPES positions "
                f"{indices[i]} and {indices[i+1]} — frequency features are still grouped!"
            )

    def test_kurt_not_kurtosis(self):
        """CSV columns use 'kurt'; FEATURE_GROUPS['temporal'][3] must be 'kurt'."""
        assert "kurt" in FEATURE_GROUPS['temporal'], (
            "config.FEATURE_GROUPS['temporal'] should contain 'kurt' (matching "
            "extract_temporal.py CSV column names), not 'kurtosis'."
        )
        assert "kurtosis" not in FEATURE_GROUPS['temporal'], (
            "config.FEATURE_GROUPS['temporal'] contains 'kurtosis' but the CSV "
            "uses 'kurt'. This would cause silent feature misalignment."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. aggregate_to_lobes() column ordering
# ──────────────────────────────────────────────────────────────────────────────

class TestAggregateToLobesOrdering:
    """Columns produced by aggregate_to_lobes() must follow the canonical order."""

    @pytest.fixture
    def minimal_roi_df(self):
        """Build a minimal (2-subject) ROI-level DataFrame with all required columns."""
        n_subjects = 2
        n_rois = 170
        n_features = len(FEATURE_TYPES)

        data = {"subject_id": [f"sub_{i:04d}" for i in range(n_subjects)]}
        rng = np.random.default_rng(42)
        for r in range(1, n_rois + 1):
            for feat in FEATURE_TYPES:
                data[f"roi{r}_{feat}"] = rng.standard_normal(n_subjects)

        return pd.DataFrame(data)

    def test_output_column_count(self, minimal_roi_df):
        """Output should have subject_id + NUM_LOBES × len(FEATURE_TYPES) columns."""
        result = aggregate_to_lobes(minimal_roi_df)
        expected_cols = 1 + NUM_LOBES * len(FEATURE_TYPES)
        assert len(result.columns) == expected_cols, (
            f"Expected {expected_cols} columns, got {len(result.columns)}"
        )

    def test_column_order_matches_feature_types(self, minimal_roi_df):
        """
        For each lobe, the feature columns must appear in the order defined by
        FEATURE_TYPES — this is what graph_factory.py's positional load relies on.
        """
        result = aggregate_to_lobes(minimal_roi_df)
        feat_cols = [c for c in result.columns if c != "subject_id"]

        # Reconstruct expected column order (LOBE_NAMES is a dict {int: str})
        expected = [
            f"{LOBE_NAMES[lobe_id]}_{feat}"
            for lobe_id in range(NUM_LOBES)
            for feat in FEATURE_TYPES
        ]
        assert feat_cols == expected, (
            "aggregate_to_lobes() output column order does not match "
            "LOBE_NAMES × FEATURE_TYPES.\n"
            f"First mismatch at index: "
            f"{next(i for i, (a, b) in enumerate(zip(feat_cols, expected)) if a != b)}"
        )

    def test_frequency_interleaved_in_output(self, minimal_roi_df):
        """
        Specifically verify the interleaved ordering for the first lobe:
        Frontal_Superior_kurt should appear before Frontal_Superior_delta_power,
        which should appear before lobe0_theta_power (with delta_peak between them).
        """
        result = aggregate_to_lobes(minimal_roi_df)
        lobe0 = LOBE_NAMES[0]
        cols = [c for c in result.columns if c.startswith(lobe0)]
        feat_suffix = [c.replace(f"{lobe0}_", "") for c in cols]
        assert feat_suffix == FEATURE_TYPES, (
            f"Column order for lobe '{lobe0}' does not match FEATURE_TYPES.\n"
            f"Expected: {FEATURE_TYPES}\n"
            f"   Got:   {feat_suffix}"
        )

    def test_subject_id_column_present(self, minimal_roi_df):
        """Output must include a subject_id column."""
        result = aggregate_to_lobes(minimal_roi_df)
        assert "subject_id" in result.columns


# ──────────────────────────────────────────────────────────────────────────────
# 3. Z-score normalisation: construct_causal applies exactly one z-score
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleZScore:
    """
    construct_causal.construct_graph() must apply exactly one z-score.
    We verify this by inspecting the source code for the z-score pattern.
    """

    def test_zscore_present_in_construct_graph(self):
        """construct_graph() source must contain the z-score block."""
        import inspect

        from src.features.construct_causal import construct_graph
        source = inspect.getsource(construct_graph)
        assert "ts_mean" in source and "ts_std" in source, (
            "construct_causal.construct_graph() is missing the z-score block "
            "(ts_mean / ts_std). Re-apply Phase 1 fix."
        )
        # The import of NiftiLabelsMasker must NOT appear (only in abide_download.py).
        # Check outside docstring by verifying no assignment like "masker = NiftiLabelsMasker("
        code_lines = [
            line for line in source.splitlines()
            if not line.strip().startswith('"""') and not line.strip().startswith('#')
            and not line.strip().startswith("'")
        ]
        code_body = "\n".join(code_lines)
        assert "NiftiLabelsMasker(" not in code_body, (
            "construct_causal.construct_graph() constructs a NiftiLabelsMasker — "
            "z-score should be applied manually after loading the .npy file, "
            "not via nilearn standardize."
        )

    def test_zscore_not_in_abide_download(self):
        """abide_download.py NiftiLabelsMasker must use standardize=False."""
        abide_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "data" / "abide_download.py"
        )
        source = abide_path.read_text()
        # Check that standardize=True does NOT appear
        assert "standardize=True" not in source, (
            "abide_download.py still has standardize=True in NiftiLabelsMasker. "
            "This causes double z-scoring. Fix: set standardize=False."
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Evaluation threshold: loaded from checkpoints, not optimised on test set
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluationThreshold:
    """
    run_evaluation.run_ensemble_evaluation() must derive the decision threshold
    from the mean of val-fold thresholds stored in the checkpoint files,
    NOT by maximising F1 on the test set labels.
    """

    def test_optimal_threshold_not_called_on_test_labels(self):
        """
        Inspect run_ensemble_evaluation source: _optimal_threshold must not be called
        with (ens_probs, labels) — that would look at test labels to pick threshold.
        """
        import inspect
        import re

        from src.run_evaluation import run_ensemble_evaluation

        source = inspect.getsource(run_ensemble_evaluation)
        # The bad pattern: _optimal_threshold(...labels...) after building ens_probs
        bad_pattern = re.compile(r"_optimal_threshold\s*\(\s*ens_probs\s*,\s*labels\s*\)")
        assert not bad_pattern.search(source), (
            "run_ensemble_evaluation() still calls _optimal_threshold(ens_probs, labels). "
            "This optimises the threshold on the test set — data leakage. "
            "Fix: use mean of val-fold thresholds from checkpoint files."
        )

    def test_threshold_loaded_from_checkpoint(self):
        """
        Inspect source: fold_thresholds list must be built from ckpt.get('threshold').
        """
        import inspect

        from src.run_evaluation import run_ensemble_evaluation

        source = inspect.getsource(run_ensemble_evaluation)
        assert "fold_thresholds" in source, (
            "run_ensemble_evaluation() must build a fold_thresholds list from "
            "checkpoint 'threshold' keys. This key is absent from the source."
        )
        assert "ckpt.get(\"threshold\"" in source or "ckpt.get('threshold'" in source, (
            "run_ensemble_evaluation() must read 'threshold' from each checkpoint."
        )
        assert "np.mean(fold_thresholds)" in source, (
            "run_ensemble_evaluation() must set threshold = np.mean(fold_thresholds)."
        )

    def test_fixed_threshold_policy_supported(self):
        """Evaluation must support a fixed deployment threshold policy."""
        import inspect

        from src.run_evaluation import run_ensemble_evaluation

        source = inspect.getsource(run_ensemble_evaluation)
        assert '"fixed"' in source, (
            "run_ensemble_evaluation() must handle EVAL_THRESHOLD_POLICY='fixed'."
        )
