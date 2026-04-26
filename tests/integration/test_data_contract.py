"""
Data Contract Test

Loads a sample processed graph and verifies all expected keys and shapes are present.
This ensures graph construction produces valid artifacts.

Run: pytest tests/integration/test_data_contract.py -v
"""
import pytest
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_sample_graph():
    """Find any processed graph file for testing."""
    graph_dir = Path("data/processed/causal_graphs")
    if not graph_dir.exists():
        pytest.skip(f"Graph directory not found: {graph_dir}")
    
    graphs = list(graph_dir.glob("*_graph.pt"))
    if not graphs:
        pytest.skip("No graph files found for testing")
    
    return graphs[0]


class TestGraphDataContract:
    """Verify processed graph files have expected structure."""
    
    def test_graph_has_required_keys(self):
        """Graph dict must have all required keys."""
        graph_path = find_sample_graph()
        graph = torch.load(graph_path)
        
        required_keys = {
            "adj",              # Adjacency matrix
            "subject_id",       # Subject identifier
            "sparsification_info",  # Metadata about graph construction
        }
        
        missing = required_keys - set(graph.keys())
        assert not missing, f"Graph missing keys: {missing}"
    
    def test_adj_shape(self):
        """Adjacency matrix should be square (12 lobes x 12 lobes)."""
        graph = torch.load(find_sample_graph())
        adj = graph["adj"]
        
        assert adj.ndim == 2, f"adj should be 2D, got {adj.ndim}"
        assert adj.shape[0] == adj.shape[1], "adj should be square"
        assert adj.shape[0] == 12, f"Expected 12x12 adj (12 lobes), got {adj.shape}"
    
    def test_adj_values(self):
        """Adjacency values can be positive or negative (z-scores indicate direction)."""
        graph = torch.load(find_sample_graph())
        adj = graph["adj"]
        
        # Values can be positive or negative - they represent z-scores
        # Positive: A predicts B; negative: negative predictive relationship
        assert adj.isfinite().all(), "adj should have finite values (no NaN/inf)"
    
    def test_internal_features_shape(self):
        """Internal features should exist with expected shape."""
        graph = torch.load(find_sample_graph())
        
        assert "internal_features" in graph, "Missing internal_features"
        internal = graph["internal_features"]
        
        # Should be (NUM_LOBES, NUM_TEMPORAL_FEATURES)
        # Expected: 12 lobes x ~24 features
        assert internal.shape[0] == 12, f"Expected 12 lobes, got {internal.shape[0]}"
    
    def test_subject_id_present(self):
        """Subject ID should be a string."""
        graph = torch.load(find_sample_graph())
        subject_id = graph["subject_id"]
        
        assert isinstance(subject_id, str), f"subject_id should be string, got {type(subject_id)}"
        assert len(subject_id) > 0, "subject_id should not be empty"
    
    def test_sparsification_info(self):
        """Sparsification info should be a dict with metadata."""
        graph = torch.load(find_sample_graph())
        
        assert "sparsification_info" in graph, "Missing sparsification_info"
        info = graph["sparsification_info"]
        
        assert isinstance(info, dict), "sparsification_info should be a dict"
        assert "triggered" in info, "sparsification_info should have 'triggered' key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])