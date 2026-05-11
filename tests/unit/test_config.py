"""
Unit tests for src/core/config.py — validate_lobe_mapping().

Run:
    pytest tests/unit/test_config.py -v
"""
import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    LOBE_MAPPING,
    NUM_LOBES,
)
from src.core.validators import validate_lobe_mapping

# ── Happy-path tests ───────────────────────────────────────────────────────────

class TestValidateLobeMappingHappyPath:
    """validate_lobe_mapping() must pass on the production LOBE_MAPPING."""

    def test_returns_true(self):
        """Function should return True when the mapping is valid."""
        assert validate_lobe_mapping() is True

    def test_exactly_num_lobes_regions(self):
        """LOBE_MAPPING must contain exactly NUM_LOBES entries."""
        assert len(LOBE_MAPPING) == NUM_LOBES

    def test_all_indices_in_range(self):
        """Every 0-indexed ROI must be in [0, 169] (1-indexed [1, 170])."""
        for lobe_id, indices in LOBE_MAPPING.items():
            for idx in indices:
                assert 0 <= idx <= 169, (
                    f"LOBE_MAPPING[{lobe_id}] contains index {idx} (1-indexed {idx + 1}) "
                    f"which is outside the valid range [0, 169]."
                )

    def test_no_duplicate_roi_across_lobes(self):
        """Each ROI index must appear in exactly one lobe."""
        all_indices = []
        for indices in LOBE_MAPPING.values():
            all_indices.extend(indices)
        seen, duplicates = set(), set()
        for idx in all_indices:
            if idx in seen:
                duplicates.add(idx + 1)  # 1-indexed for readability
            seen.add(idx)
        assert not duplicates, (
            f"Duplicate ROI indices detected (1-indexed): {sorted(duplicates)}"
        )

    def test_full_170_roi_coverage(self):
        """All 170 AAL ROIs (0-indexed: 0–169) must be covered by some lobe."""
        covered = set()
        for indices in LOBE_MAPPING.values():
            covered.update(indices)
        expected = set(range(170))
        missing_0idx = expected - covered
        missing_1idx = sorted(i + 1 for i in missing_0idx)
        assert not missing_1idx, (
            f"{len(missing_1idx)} AAL ROI(s) not assigned to any lobe "
            f"(1-indexed): {missing_1idx}"
        )

    def test_lobe_ids_are_consecutive_from_zero(self):
        """Lobe IDs should be 0, 1, ..., NUM_LOBES-1 (no gaps)."""
        assert set(LOBE_MAPPING.keys()) == set(range(NUM_LOBES))


# ── Error-path tests ───────────────────────────────────────────────────────────

class TestValidateLobeMappingErrorPath:
    """validate_lobe_mapping() must raise ValueError on a malformed mapping."""

    def test_wrong_number_of_regions_raises(self, monkeypatch):
        """Should raise when the number of regions != NUM_LOBES."""
        import src.core.config as cfg

        bad_mapping = {k: v for k, v in LOBE_MAPPING.items() if k < NUM_LOBES - 1}
        monkeypatch.setattr(cfg, "LOBE_MAPPING", bad_mapping)
        with pytest.raises(ValueError, match="NUM_LOBES"):
            validate_lobe_mapping()

    def test_out_of_range_roi_raises(self, monkeypatch):
        """Should raise when an ROI index is outside [0, 169]."""
        import src.core.config as cfg

        # Patch: replace last lobe with one that contains OOB index 170 (1-indexed 171)
        patched = dict(LOBE_MAPPING)
        patched[NUM_LOBES - 1] = [170]  # 0-indexed 170 → 1-indexed 171, out of range
        monkeypatch.setattr(cfg, "LOBE_MAPPING", patched)
        with pytest.raises(ValueError, match="out-of-range"):
            validate_lobe_mapping()

    def test_duplicate_roi_raises(self, monkeypatch):
        """Should raise when the same ROI index appears in two lobes."""
        import src.core.config as cfg

        # Duplicate index 0 (ROI 1) into lobe 1 as well
        patched = {k: list(v) for k, v in LOBE_MAPPING.items()}
        patched[1] = patched[1] + [0]  # index 0 already in lobe 0
        monkeypatch.setattr(cfg, "LOBE_MAPPING", patched)
        with pytest.raises(ValueError, match="duplicate"):
            validate_lobe_mapping()

    def test_missing_roi_raises(self, monkeypatch):
        """Should raise when at least one ROI is not assigned to any lobe."""
        import src.core.config as cfg

        # Remove the last ROI from the last lobe so coverage is incomplete
        patched = {k: list(v) for k, v in LOBE_MAPPING.items()}
        last_key = max(patched.keys())
        patched[last_key] = patched[last_key][:-1]
        monkeypatch.setattr(cfg, "LOBE_MAPPING", patched)
        with pytest.raises(ValueError, match="does not cover"):
            validate_lobe_mapping()
