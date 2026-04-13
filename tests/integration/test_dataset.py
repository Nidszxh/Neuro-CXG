"""
Integration test: ABIDECausalDataset.__getitem__ (graph_factory.py).

Strategy
--------
* Build a fully self-contained mock data environment in a temporary directory.
* Monkeypatch the config constants so that ABIDECausalDataset reads from the
  temp directory instead of the real data folder.
* Verify that get() returns a torch_geometric.data.Data object whose tensors
  have the correct shapes and data types.

Run:
    pytest tests/integration/test_dataset.py -v

Note: no GPU / real ABIDE data required.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import src.core.config as cfg
from src.core.config import (
    NUM_LOBES, LOBE_NAMES, NUM_TEMPORAL_FEATURES,
    NUM_SPATIAL_FEATURES, GNN_IN_CHANNELS, FEATURE_GROUPS,
)

# ── Constants for mock data ───────────────────────────────────────────────────

SUBJECT_ID = "TEST_0000001"
SPLIT = "train"
DX_GROUP = 2         # ASD
SITE_ID = "MOCK_SITE"
N_EDGES_MIN = 12     # must match construct_causal MIN_EDGES_PER_GRAPH


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_manifest(subject_id: str, split: str, dx_group: int = DX_GROUP) -> pd.DataFrame:
    return pd.DataFrame([{
        'subject_id': subject_id,
        'split': split,
        'DX_GROUP': dx_group,
        'SITE_ID': SITE_ID,
        'AGE_AT_SCAN': 12.5,
        'SEX': 1,
        'FIQ': 105,
    }])


def _make_temporal_features(subject_id: str) -> pd.DataFrame:
    """Build a harmonized temporal feature row with current temporal+frequency schema."""
    feature_cols = {}
    temporal_feature_names = FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["frequency"]
    for lobe_id in range(NUM_LOBES):
        lobe_name = LOBE_NAMES[lobe_id]
        for feat in temporal_feature_names:
            feature_cols[f"{lobe_name}_{feat}"] = np.random.rand()
    row = {"subject_id": subject_id, **feature_cols}
    return pd.DataFrame([row])


def _make_spatial_features(subject_id: str) -> pd.DataFrame:
    """Build a spatial feature row with all 6 columns per lobe."""
    spatial_cols = {"subject_id": subject_id}
    for lobe_id in range(NUM_LOBES):
        name = LOBE_NAMES[lobe_id]
        spatial_cols[f"{name}_x"] = float(np.random.rand() * 100)
        spatial_cols[f"{name}_y"] = float(np.random.rand() * 100)
        spatial_cols[f"{name}_z_depth"] = float(np.random.rand() * 50)
        spatial_cols[f"{name}_size"] = float(np.random.rand() * 2000)
        spatial_cols[f"{name}_conf_std"] = float(np.random.rand())
        spatial_cols[f"{name}_detection_count"] = float(np.random.randint(3, 8))
    return pd.DataFrame([spatial_cols])


def _make_sparse_adj(num_lobes: int, min_edges: int = 20) -> torch.Tensor:
    """
    Create a sparse directed adjacency matrix with at least *min_edges* non-zero entries.
    Uses -log10(p) style values (positive floats) like the Granger output.
    """
    adj = torch.zeros(num_lobes, num_lobes)
    # Randomly place edges until we have enough
    rng = np.random.default_rng(42)
    edges_placed = 0
    while edges_placed < min_edges:
        i = rng.integers(0, num_lobes)
        j = rng.integers(0, num_lobes)
        if i != j and adj[i, j] == 0:
            adj[i, j] = float(rng.uniform(0.5, 5.0))
            edges_placed += 1
    return adj


# ── Session-scoped fixture: build temp data directory ─────────────────────────

@pytest.fixture(scope="module")
def mock_data_dir(tmp_path_factory):
    """
    Writes all mock data files to a temporary directory and returns its Path.
    """
    root = tmp_path_factory.mktemp("mock_abide")

    # ── Directories ────────────────────────────────────────────────────────
    meta_dir = root / "metadata"
    graphs_dir = root / "causal_graphs"
    meta_dir.mkdir()
    graphs_dir.mkdir()

    np.random.seed(7)

    # ── Manifest ───────────────────────────────────────────────────────────
    manifest_df = _make_manifest(SUBJECT_ID, SPLIT)
    manifest_df.to_csv(meta_dir / "master_manifest.csv", index=False)

    # ── Temporal features ──────────────────────────────────────────────────
    temporal_df = _make_temporal_features(SUBJECT_ID)
    temporal_df.to_csv(meta_dir / "node_attributes_harmonized.csv", index=False)

    # ── Spatial features ───────────────────────────────────────────────────
    spatial_df = _make_spatial_features(SUBJECT_ID)
    spatial_df.to_csv(meta_dir / "node_features_3d.csv", index=False)

    # ── Causal graph (.pt) ─────────────────────────────────────────────────
    adj = _make_sparse_adj(NUM_LOBES, min_edges=N_EDGES_MIN)
    internal = torch.rand(NUM_LOBES, 2)
    graph_dict = {
        'adj': adj,
        'internal_features': internal,
        'subject_id': SUBJECT_ID,
        'lobe_order': list(range(NUM_LOBES)),
    }
    torch.save(graph_dict, graphs_dir / f"{SUBJECT_ID}_graph.pt")

    return root, meta_dir, graphs_dir


# ── Dataset fixture ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset(mock_data_dir):
    """
    Instantiate ABIDECausalDataset with all paths monkeypatched to the temp dir.

    Uses patch() as an explicit context manager (scope limit) so that only this
    fixture (and the tests that depend on it) see the patched values.
    """
    root, meta_dir, graphs_dir = mock_data_dir

    patches = {
        "src.features.graph_factory.MASTER_MANIFEST":    meta_dir / "master_manifest.csv",
        "src.features.graph_factory.NODE_ATTRIBUTES_HARMONIZED": meta_dir / "node_attributes_harmonized.csv",
        "src.features.graph_factory.NODE_FEATURES_3D":   meta_dir / "node_features_3d.csv",
        # Point the harmonized path at a non-existent file so graph_factory falls
        # back to NODE_FEATURES_3D (the patched mock) rather than using the real
        # node_features_3d_harmonized.csv that may exist on disk.
        "src.features.graph_factory.NODE_FEATURES_3D_HARMONIZED": meta_dir / "node_features_3d_harmonized.csv",
        "src.features.graph_factory.CAUSAL_GRAPHS_DIR":  graphs_dir,
    }

    with patch.multiple("src.features.graph_factory", **{
        k.split(".")[-1]: v for k, v in patches.items()
    }):
        from src.features.graph_factory import ABIDECausalDataset
        ds = ABIDECausalDataset(split=SPLIT)

    return ds


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestABIDECausalDataset:

    def test_dataset_not_empty(self, dataset):
        """The mock dataset should contain exactly 1 subject."""
        assert len(dataset) == 1, f"Expected 1 subject, got {len(dataset)}"

    def test_getitem_returns_data(self, dataset):
        """get(0) must return a non-None PyG Data object."""
        sample = dataset.get(0)
        assert sample is not None, "get(0) returned None — check logs for error details"

    def test_node_feature_shape(self, dataset):
        """x must have shape (NUM_LOBES, GNN_IN_CHANNELS)."""
        sample = dataset.get(0)
        assert sample is not None
        assert sample.x.shape == (NUM_LOBES, GNN_IN_CHANNELS), (
            f"Expected x.shape=({NUM_LOBES}, {GNN_IN_CHANNELS}), got {sample.x.shape}"
        )

    def test_edge_index_shape(self, dataset):
        """edge_index must be (2, num_edges) with at least N_EDGES_MIN edges."""
        sample = dataset.get(0)
        assert sample is not None
        assert sample.edge_index.shape[0] == 2, (
            f"edge_index first dim must be 2, got {sample.edge_index.shape[0]}"
        )
        assert sample.edge_index.shape[1] >= N_EDGES_MIN, (
            f"Expected at least {N_EDGES_MIN} edges, got {sample.edge_index.shape[1]}"
        )

    def test_edge_attr_shape(self, dataset):
        """edge_attr must be (num_edges, 1) to match GAT expectations."""
        sample = dataset.get(0)
        assert sample is not None
        n_edges = sample.edge_index.shape[1]
        assert sample.edge_attr.shape == (n_edges, 1), (
            f"Expected edge_attr.shape=({n_edges}, 1), got {sample.edge_attr.shape}"
        )

    def test_label_is_binary(self, dataset):
        """y must be 0 (Control) or 1 (ASD)."""
        sample = dataset.get(0)
        assert sample is not None
        label = sample.y.item()
        assert label in (0, 1), f"Label must be 0 or 1, got {label}"

    def test_label_matches_dx_group(self, dataset):
        """DX_GROUP==2 should map to label==1 (ASD)."""
        sample = dataset.get(0)
        assert sample is not None
        expected = 1  # DX_GROUP=2 (ASD) → 1
        assert sample.y.item() == expected, (
            f"DX_GROUP={DX_GROUP} should produce label={expected}, got {sample.y.item()}"
        )

    def test_node_features_are_finite(self, dataset):
        """x must not contain NaN or Inf after construction."""
        sample = dataset.get(0)
        assert sample is not None
        assert torch.isfinite(sample.x).all(), (
            f"Node feature tensor x contains non-finite values"
        )

    def test_edge_attr_are_finite(self, dataset):
        """edge_attr must not contain NaN or Inf."""
        sample = dataset.get(0)
        assert sample is not None
        assert torch.isfinite(sample.edge_attr).all()

    def test_edge_attr_non_negative(self, dataset):
        """
        Edge weights are -log10(p-value) for Granger or positive correlation values.
        They must be ≥ 0.  Training augmentation may zero individual weights (edge
        dropout), so strictly > 0 is not guaranteed on the train split.
        """
        sample = dataset.get(0)
        assert sample is not None
        assert (sample.edge_attr >= 0).all(), (
            f"Some edge weights are negative: {sample.edge_attr[sample.edge_attr < 0]}"
        )

    def test_pos_shape(self, dataset):
        """pos (centroid coordinates) must be (NUM_LOBES, 3)."""
        sample = dataset.get(0)
        assert sample is not None
        assert sample.pos.shape == (NUM_LOBES, 3), (
            f"Expected pos.shape=({NUM_LOBES}, 3), got {sample.pos.shape}"
        )

    def test_site_id_is_tensor(self, dataset):
        """site_id must be a long tensor."""
        sample = dataset.get(0)
        assert sample is not None
        assert sample.site_id.dtype == torch.long

    def test_demographic_tensors_are_float(self, dataset):
        """age, sex, fiq must be float32 tensors."""
        sample = dataset.get(0)
        assert sample is not None
        for attr in ("age", "sex", "fiq"):
            val = getattr(sample, attr)
            assert val.dtype == torch.float32, (
                f"{attr} dtype expected float32, got {val.dtype}"
            )


# ── Control label-encoding fixture + test ─────────────────────────────────────

@pytest.fixture(scope="module")
def mock_data_dir_control(tmp_path_factory):
    """Same as mock_data_dir but DX_GROUP=1 (ABIDE Control)."""
    root = tmp_path_factory.mktemp("mock_abide_ctrl")
    meta_dir   = root / "metadata"
    graphs_dir = root / "causal_graphs"
    meta_dir.mkdir()
    graphs_dir.mkdir()

    np.random.seed(13)
    subject_id = "TEST_CTRL_0000001"
    _make_manifest(subject_id, "train", dx_group=1).to_csv(
        meta_dir / "master_manifest.csv", index=False
    )
    _make_temporal_features(subject_id).to_csv(
        meta_dir / "node_attributes_harmonized.csv", index=False
    )
    _make_spatial_features(subject_id).to_csv(
        meta_dir / "node_features_3d.csv", index=False
    )
    adj      = _make_sparse_adj(NUM_LOBES, min_edges=N_EDGES_MIN)
    internal = torch.rand(NUM_LOBES, 2)
    torch.save(
        {"adj": adj, "internal_features": internal,
         "subject_id": subject_id, "lobe_order": list(range(NUM_LOBES))},
        graphs_dir / f"{subject_id}_graph.pt",
    )
    return root, meta_dir, graphs_dir


@pytest.fixture(scope="module")
def dataset_control(mock_data_dir_control):
    """ABIDECausalDataset built from Control (DX_GROUP=1) mock data."""
    _, meta_dir, graphs_dir = mock_data_dir_control
    with patch.multiple("src.features.graph_factory",
                        MASTER_MANIFEST=meta_dir / "master_manifest.csv",
                        NODE_ATTRIBUTES_HARMONIZED=meta_dir / "node_attributes_harmonized.csv",
                        NODE_FEATURES_3D=meta_dir / "node_features_3d.csv",
                        NODE_FEATURES_3D_HARMONIZED=meta_dir / "node_features_3d_harmonized.csv",
                        CAUSAL_GRAPHS_DIR=graphs_dir):
        from src.features.graph_factory import ABIDECausalDataset
        ds = ABIDECausalDataset(split="train")
    return ds


def test_label_encoding_control(dataset_control):
    """DX_GROUP=1 (ABIDE Control) must map to y=0 (GNN Control class)."""
    sample = dataset_control.get(0)
    assert sample is not None, "dataset_control.get(0) returned None"
    assert sample.y.item() == 0, (
        f"DX_GROUP=1 (Control) should encode as y=0, got y={sample.y.item()}"
    )
