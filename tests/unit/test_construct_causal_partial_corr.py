"""Unit tests for the GraphicalLasso partial-correlation causal method.

Run:
    pytest tests/unit/test_construct_causal_partial_corr.py -v
"""


import torch

from src.core.config import NUM_LOBES
from src.features import construct_causal as cc
from src.features.construct_causal import compute_causality_matrix


class TestPartialCorrGlasso:
    """Validate partial-correlation graph construction behavior."""

    def test_returns_symmetric_matrix_and_metadata(self):
        """Method should output a finite, near-symmetric matrix and expected metadata."""
        torch.manual_seed(42)
        n_time = 220
        ts = torch.randn(n_time, NUM_LOBES)

        # Inject a clear dependency between two lobes to ensure non-zero recovery.
        ts[:, 1] = 0.85 * ts[:, 0] + 0.15 * torch.randn(n_time)

        matrix, metadata = compute_causality_matrix(
            ts,
            method="partial_corr_glasso",
            return_metadata=True,
        )

        assert matrix.shape == (NUM_LOBES, NUM_LOBES)
        assert torch.isfinite(matrix).all(), "partial_corr_glasso returned non-finite values"
        assert torch.allclose(
            torch.diag(matrix),
            torch.zeros(NUM_LOBES, device=matrix.device),
        ), "Diagonal must remain zero"

        assert "partial_corr_matrix" in metadata
        assert "precision_matrix" in metadata
        assert "pvalue_matrix" in metadata
        assert "fdr_significant_mask" in metadata

        sym_diff = torch.max(torch.abs(matrix - matrix.T)).item()
        assert sym_diff < 1e-4, f"Expected symmetric matrix, max asymmetry={sym_diff:.6f}"
        assert abs(float(matrix[0, 1])) > 1e-3, "Expected non-trivial edge for injected dependency"

    def test_short_series_returns_zeros(self):
        """Short series should trigger graceful zero-matrix fallback."""
        ts = torch.randn(20, NUM_LOBES)
        matrix, metadata = compute_causality_matrix(
            ts,
            method="partial_corr_glasso",
            return_metadata=True,
        )

        assert torch.all(matrix == 0), "Expected all-zero fallback for short series"
        assert "partial_corr_matrix" in metadata
        assert torch.all(metadata["partial_corr_matrix"] == 0)
        assert "pvalue_matrix" in metadata
        assert torch.all(metadata["pvalue_matrix"] == 1)

    def test_nan_input_returns_zeros(self):
        """NaN input should trigger graceful zero-matrix fallback."""
        ts = torch.randn(160, NUM_LOBES)
        ts[0, 0] = torch.nan

        matrix, metadata = compute_causality_matrix(
            ts,
            method="partial_corr_glasso",
            return_metadata=True,
        )

        assert torch.all(matrix == 0), "Expected all-zero fallback for NaN input"
        assert "low_confidence_mask" in metadata
        assert metadata["low_confidence_mask"].all()

    def test_fdr_mask_controls_retained_edges(self):
        """When FDR is enabled, non-significant off-diagonal entries must be zeroed."""
        torch.manual_seed(7)
        n_time = 260
        ts = torch.randn(n_time, NUM_LOBES)
        ts[:, 3] = 0.80 * ts[:, 2] + 0.20 * torch.randn(n_time)

        orig_fdr_enabled = cc.PARTIAL_CORR_FDR_ENABLED
        try:
            cc.PARTIAL_CORR_FDR_ENABLED = True
            matrix, metadata = compute_causality_matrix(
                ts,
                method="partial_corr_glasso",
                return_metadata=True,
            )
        finally:
            cc.PARTIAL_CORR_FDR_ENABLED = orig_fdr_enabled

        offdiag_mask = ~torch.eye(NUM_LOBES, dtype=torch.bool, device=matrix.device)
        sig_mask = metadata["fdr_significant_mask"].bool() & offdiag_mask
        nonsig_mask = (~sig_mask) & offdiag_mask

        assert torch.all(matrix[nonsig_mask] == 0), "Non-significant edges should be pruned by FDR"
        assert torch.isfinite(metadata["pvalue_matrix"]).all()
