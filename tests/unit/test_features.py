"""
Unit tests for spectral / causal-inference feature functions.

Covers:
    * extract_band_power()  (src/features/extract_temporal.py)
    * compute_granger_causality()  (src/features/causal_inference.py)

Run:
    pytest tests/unit/test_features.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import ACTIVE_FREQ_BANDS
from src.features.causal_inference import compute_granger_causality
from src.features.extract_temporal import extract_band_power

# ══════════════════════════════════════════════════════════════════════════════
# extract_band_power
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractBandPower:
    """Tests for extract_band_power() using synthetic fMRI-range signals."""

    # Sampling frequency matching TR=2 s (fs = 1/TR = 0.5 Hz)
    FS = 0.5
    N = 512  # enough samples for reliable Welch PSD
    BANDS = list(ACTIVE_FREQ_BANDS.keys())

    # ── Helper ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pure_sine(freq_hz: float, n: int = 512, fs: float = 0.5) -> np.ndarray:
        """Return a pure sine wave at *freq_hz* (duration = n/fs seconds)."""
        t = np.arange(n) / fs
        return np.sin(2 * np.pi * freq_hz * t).astype(np.float64)

    # ── Happy-path ───────────────────────────────────────────────────────────

    def test_returns_expected_feature_count(self):
        """Function must return 2 features per active band plus 2 global features."""
        ts = np.random.randn(self.N)
        feats = extract_band_power(ts, fs=self.FS)
        expected = len(self.BANDS) * 2 + 2
        assert len(feats) == expected, f"Expected {expected} features, got {len(feats)}: {list(feats.keys())}"

    def test_delta_signal_dominates_delta_band(self):
        """A 0.02 Hz sine (within delta: 0.01–0.027 Hz) should have the highest power in delta."""
        ts = self._pure_sine(freq_hz=0.02, n=self.N, fs=self.FS)
        feats = extract_band_power(ts, fs=self.FS)
        band_powers = {b: feats[f"{b}_power"] for b in self.BANDS}
        assert band_powers["delta"] == max(band_powers.values()), (
            f"Expected delta to dominate, got: {band_powers}"
        )

    def test_peak_freq_in_delta_band(self):
        """A 0.02 Hz sine should report a peak frequency inside the delta band."""
        ts = self._pure_sine(freq_hz=0.02, n=self.N, fs=self.FS)
        feats = extract_band_power(ts, fs=self.FS)
        peak = feats["delta_peak_freq"]
        assert 0.01 <= peak <= 0.027, (
            f"Delta peak frequency {peak:.5f} Hz is outside the delta band [0.01, 0.027]."
        )

    def test_theta_signal_dominates_theta_band(self):
        """A 0.05 Hz sine (within theta: 0.027–0.073 Hz) should dominate theta."""
        ts = self._pure_sine(freq_hz=0.05, n=self.N, fs=self.FS)
        feats = extract_band_power(ts, fs=self.FS)
        band_powers = {b: feats[f"{b}_power"] for b in self.BANDS}
        assert band_powers["theta"] == max(band_powers.values()), (
            f"Expected theta to dominate, got: {band_powers}"
        )

    def test_all_powers_non_negative(self):
        """Band power values must be ≥ 0 (power spectral density is non-negative)."""
        ts = np.random.randn(self.N)
        feats = extract_band_power(ts, fs=self.FS)
        for band in self.BANDS:
            assert feats[f"{band}_power"] >= 0, (
                f"{band}_power is negative: {feats[f'{band}_power']}"
            )

    def test_spectral_entropy_finite_non_negative(self):
        """Spectral entropy must be a finite, non-negative scalar."""
        ts = np.random.randn(self.N)
        feats = extract_band_power(ts, fs=self.FS)
        se = feats["spectral_entropy"]
        assert np.isfinite(se), f"spectral_entropy is not finite: {se}"
        assert se >= 0, f"spectral_entropy is negative: {se}"

    def test_phase_std_finite_non_negative(self):
        """Phase standard deviation must be finite and non-negative."""
        ts = np.random.randn(self.N)
        feats = extract_band_power(ts, fs=self.FS)
        ps = feats["phase_std"]
        assert np.isfinite(ps), f"phase_std is not finite: {ps}"
        assert ps >= 0, f"phase_std is negative: {ps}"

    # ── Edge-cases ───────────────────────────────────────────────────────────

    def test_short_signal_returns_zeros(self):
        """Signals shorter than 10 samples must return the zero-feature dict."""
        ts = np.ones(5)
        feats = extract_band_power(ts, fs=self.FS)
        assert all(v == 0.0 for v in feats.values()), (
            f"Expected all zeros for short signal, got: {feats}"
        )

    def test_nan_input_returns_zeros(self):
        """NaN in the time series should trigger graceful fallback (all zeros)."""
        ts = np.full(self.N, np.nan)
        feats = extract_band_power(ts, fs=self.FS)
        assert all(v == 0.0 for v in feats.values()), (
            f"Expected all zeros for NaN input, got: {feats}"
        )

    def test_inf_input_returns_zeros(self):
        """Inf in the time series should trigger graceful fallback (all zeros)."""
        ts = np.full(self.N, np.inf)
        feats = extract_band_power(ts, fs=self.FS)
        assert all(v == 0.0 for v in feats.values()), (
            f"Expected all zeros for Inf input, got: {feats}"
        )

    def test_constant_signal_returns_zeros_or_finite(self):
        """A constant (zero-variance) signal should not raise; band powers must be ≥ 0."""
        ts = np.ones(self.N) * 3.0
        feats = extract_band_power(ts, fs=self.FS)
        for band in self.BANDS:
            p = feats[f"{band}_power"]
            assert p >= 0 and np.isfinite(p), (
                f"{band}_power not a non-negative finite for constant signal: {p}"
            )

    def test_custom_bands_respected(self):
        """Custom band definitions should be used when explicitly passed."""
        custom_bands = {"low": (0.01, 0.05), "high": (0.05, 0.25)}
        ts = np.random.randn(self.N)
        feats = extract_band_power(ts, fs=self.FS, bands=custom_bands)
        expected_keys = {"low_power", "low_peak_freq", "high_power", "high_peak_freq",
                         "spectral_entropy", "phase_std"}
        assert set(feats.keys()) == expected_keys, (
            f"Unexpected keys: {set(feats.keys()) ^ expected_keys}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# compute_granger_causality
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeGrangerCausality:
    """
    Tests for compute_granger_causality() using the canonical synthetic
    X→Y signal from causal_inference.py __main__ block.
    """

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        """
        Reproduce the test signal from causal_inference.py:
            X  – white noise
            Y  – strongly depends on past of X  (X → Y)
            Z  – independent white noise
        Returns (ts_matrix, gc_matrix).
        """
        np.random.seed(42)
        n = 200
        X = np.random.randn(n)
        Y = 0.5 * np.roll(X, 1) + 0.3 * np.random.randn(n)
        Z = np.random.randn(n)
        ts = np.column_stack([X, Y, Z])
        gc = compute_granger_causality(ts, max_lag=5)
        return ts, gc

    # ── Shape ────────────────────────────────────────────────────────────────

    def test_output_shape(self, synthetic_data):
        _, gc = synthetic_data
        assert gc.shape == (3, 3), f"Expected (3,3), got {gc.shape}"

    def test_diagonal_zero(self, synthetic_data):
        """Self-causation must be zero (no region causes itself)."""
        _, gc = synthetic_data
        for i in range(3):
            assert gc[i, i] == 0.0, f"gc[{i},{i}] = {gc[i, i]}, expected 0"

    # ── Directionality ───────────────────────────────────────────────────────

    def test_x_causes_y_stronger_than_y_causes_x(self, synthetic_data):
        """GC(X→Y) should exceed GC(Y→X) because Y was constructed from lagged X."""
        _, gc = synthetic_data
        assert gc[0, 1] > gc[1, 0], (
            f"Expected GC(X→Y)={gc[0,1]:.4f} > GC(Y→X)={gc[1,0]:.4f}"
        )

    def test_x_causes_y_stronger_than_x_causes_z(self, synthetic_data):
        """GC(X→Y) should exceed GC(X→Z) because Z is independent of X."""
        _, gc = synthetic_data
        assert gc[0, 1] > gc[0, 2], (
            f"Expected GC(X→Y)={gc[0,1]:.4f} > GC(X→Z)={gc[0,2]:.4f}"
        )

    def test_x_causes_y_is_significant(self, synthetic_data):
        """GC(X→Y) should be meaningfully above zero (strong causal signal)."""
        _, gc = synthetic_data
        assert gc[0, 1] > 0.5, (
            f"GC(X→Y)={gc[0,1]:.4f} is unexpectedly weak; signal may not be detected."
        )

    # ── Value properties ─────────────────────────────────────────────────────

    def test_all_values_non_negative(self, synthetic_data):
        """GC values are -log10(p) or 0; they must be ≥ 0 (within float epsilon)."""
        _, gc = synthetic_data
        assert (gc >= -1e-9).all(), f"Negative GC values found:\n{gc}"

    def test_no_nan_or_inf(self, synthetic_data):
        """Output must contain no NaN or Inf."""
        _, gc = synthetic_data
        assert np.isfinite(gc).all(), f"GC matrix contains non-finite values:\n{gc}"

    # ── Edge-cases ───────────────────────────────────────────────────────────

    def test_short_series_returns_zeros(self):
        """A time series too short for the requested lag must return zero matrix."""
        ts = np.random.randn(10, 3)  # only 10 points, need > lag+10=15 minimum
        gc = compute_granger_causality(ts, max_lag=5)
        assert gc.shape == (3, 3)
        assert (gc == 0.0).all(), f"Expected zero matrix for short series, got:\n{gc}"

    def test_nan_input_returns_zeros(self):
        """NaN in the time series must be handled gracefully."""
        ts = np.full((200, 3), np.nan)
        gc = compute_granger_causality(ts, max_lag=5)
        assert gc.shape == (3, 3)
        assert (gc == 0.0).all()

    def test_inf_input_returns_zeros(self):
        """Inf in the time series must be handled gracefully."""
        ts = np.full((200, 3), np.inf)
        gc = compute_granger_causality(ts, max_lag=5)
        assert gc.shape == (3, 3)
        assert (gc == 0.0).all()

    def test_lag_1_still_detects_causality(self):
        """Even with max_lag=1, the X→Y signal should be detectable."""
        np.random.seed(0)
        n = 200
        X = np.random.randn(n)
        Y = 0.8 * np.roll(X, 1) + 0.1 * np.random.randn(n)
        ts = np.column_stack([X, Y])
        gc = compute_granger_causality(ts, max_lag=1)
        assert gc[0, 1] > gc[1, 0], (
            f"Lag-1: expected GC(X→Y)={gc[0,1]:.4f} > GC(Y→X)={gc[1,0]:.4f}"
        )
