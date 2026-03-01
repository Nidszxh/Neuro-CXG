"""
Integration test: single-subject graph construction pipeline.

Verifies the complete path:
    synthetic NPY  →  aggregate_to_lobes
                   →  compute_causality_matrix (lagged_pearson, fast)
                   →  adaptive_sparsification
                   →  torch.save  /  torch.load
                   →  valid graph dict with correct shapes

Run:
    pytest tests/integration/test_graph_construction.py -v

Notes:
* Uses 'lagged_pearson' as the causality method so the test runs in ~1 s even
  on a CPU-only machine (Granger causality with 12 regions × max_lag=5 is
  too slow for CI).
* No external data files are required; the time series is synthesised inline.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import NUM_LOBES, LOBE_MAPPING
from src.features.construct_causal import (
    aggregate_to_lobes,
    compute_causality_matrix,
    adaptive_sparsification,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_ts_tensor():
    """
    Return a realistic synthetic time series as a torch.Tensor.

    Shape: (200 timepoints, 170 ROIs)  – standard ABIDE fMRI format.
    Values are white noise; for pipeline shape/type tests this is sufficient.
    """
    torch.manual_seed(0)
    return torch.randn(200, 170, dtype=torch.float32)


@pytest.fixture(scope="module")
def aggregated(synthetic_ts_tensor):
    """Run aggregate_to_lobes() once and return (lobe_signals, internal_features)."""
    return aggregate_to_lobes(synthetic_ts_tensor)


@pytest.fixture(scope="module")
def causal_matrix(aggregated):
    """Compute a causal matrix using the fast lagged_pearson method."""
    lobe_signals, _ = aggregated
    return compute_causality_matrix(lobe_signals, method='lagged_pearson')


@pytest.fixture(scope="module")
def sparsified_matrix(causal_matrix):
    """Apply adaptive_sparsification with the 'fixed' method for determinism."""
    return adaptive_sparsification(causal_matrix, method='fixed')


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: aggregate_to_lobes
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregateToLobes:

    def test_output_shapes(self, aggregated):
        """lobe_signals must be (T, NUM_LOBES) and internal_features (NUM_LOBES, 2)."""
        lobe_signals, internal_features = aggregated
        assert lobe_signals.shape[1] == NUM_LOBES, (
            f"Expected {NUM_LOBES} lobe signals, got {lobe_signals.shape[1]}"
        )
        assert lobe_signals.shape[0] == 200, "Timepoints should be preserved (200)"
        assert internal_features.shape == (NUM_LOBES, 2), (
            f"Expected internal_features shape ({NUM_LOBES}, 2), got {internal_features.shape}"
        )

    def test_no_nan_inf_in_lobe_signals(self, aggregated):
        lobe_signals, _ = aggregated
        assert torch.isfinite(lobe_signals).all(), (
            "lobe_signals must not contain NaN/Inf after aggregation"
        )

    def test_no_nan_inf_in_internal_features(self, aggregated):
        _, internal_features = aggregated
        assert torch.isfinite(internal_features).all(), (
            "internal_features must not contain NaN/Inf"
        )

    def test_coherence_in_valid_range(self, aggregated):
        """Coherence values (first internal feature) must be in [-1, 1]."""
        _, internal_features = aggregated
        coherence = internal_features[:, 0]
        assert (coherence >= -1.0).all() and (coherence <= 1.0).all(), (
            f"Coherence out of [-1, 1]: min={coherence.min():.4f}, max={coherence.max():.4f}"
        )

    def test_spatial_variance_non_negative(self, aggregated):
        """Spatial variance (second internal feature) must be ≥ 0."""
        _, internal_features = aggregated
        spatial_var = internal_features[:, 1]
        assert (spatial_var >= 0).all(), (
            f"Spatial variance has negative values: {spatial_var}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: compute_causality_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeCausalityMatrix:

    def test_output_shape(self, causal_matrix):
        assert causal_matrix.shape == (NUM_LOBES, NUM_LOBES), (
            f"Expected ({NUM_LOBES}, {NUM_LOBES}), got {causal_matrix.shape}"
        )

    def test_output_is_float(self, causal_matrix):
        assert causal_matrix.dtype in (torch.float32, torch.float64)

    def test_no_nan_inf(self, causal_matrix):
        assert torch.isfinite(causal_matrix).all(), (
            "Causal matrix contains NaN/Inf"
        )

    def test_not_all_zeros(self, causal_matrix):
        """A non-trivial time series should produce at least some non-zero edges."""
        assert (causal_matrix != 0).any(), "Causal matrix is entirely zero"


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: adaptive_sparsification
# ══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveSparsification:

    def test_output_shape(self, causal_matrix):
        sparse = adaptive_sparsification(causal_matrix, method='fixed')
        assert sparse.shape == (NUM_LOBES, NUM_LOBES)

    def test_sparsified_is_subset_of_original(self, causal_matrix):
        """Every non-zero edge in the sparsified matrix must also be non-zero in the original."""
        sparse = adaptive_sparsification(causal_matrix, method='fixed')
        original_zero_mask = causal_matrix == 0
        assert not (sparse[original_zero_mask] != 0).any(), (
            "Sparsification introduced edges that didn't exist in the original matrix"
        )

    def test_minimum_edges_satisfied(self, causal_matrix):
        """The sparsified matrix should contain at least MIN_EDGES_PER_GRAPH edges."""
        from src.core.config import MIN_EDGES_PER_GRAPH
        sparse = adaptive_sparsification(causal_matrix, method='fixed')
        n_edges = (sparse != 0).sum().item()
        assert n_edges >= MIN_EDGES_PER_GRAPH, (
            f"Sparsified graph has {n_edges} edges, below minimum {MIN_EDGES_PER_GRAPH}"
        )

    def test_no_nan_inf(self, causal_matrix):
        sparse = adaptive_sparsification(causal_matrix, method='fixed')
        assert torch.isfinite(sparse).all()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4: torch.save / torch.load round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphSaveLoadRoundTrip:

    def test_saved_dict_has_required_keys(self, aggregated, sparsified_matrix):
        """Saved graph dict must contain 'adj', 'internal_features', 'subject_id', 'lobe_order'."""
        _, internal_features = aggregated
        graph_dict = {
            'adj': sparsified_matrix,
            'internal_features': internal_features,
            'subject_id': 'SYN_TEST_001',
            'lobe_order': list(range(NUM_LOBES)),
        }
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = Path(f.name)

        try:
            torch.save(graph_dict, path)
            loaded = torch.load(path, weights_only=False)

            for key in ('adj', 'internal_features', 'subject_id', 'lobe_order'):
                assert key in loaded, f"Key '{key}' missing from loaded graph dict"

            assert loaded['subject_id'] == 'SYN_TEST_001'
            assert torch.equal(loaded['adj'], sparsified_matrix)
            assert loaded['adj'].shape == (NUM_LOBES, NUM_LOBES)
            assert loaded['internal_features'].shape == (NUM_LOBES, 2)
        finally:
            path.unlink(missing_ok=True)

    def test_adj_edges_non_zero(self, sparsified_matrix):
        """The saved adjacency matrix must have at least one non-zero entry."""
        assert (sparsified_matrix != 0).any(), "Sparsified adjacency is entirely zero"
