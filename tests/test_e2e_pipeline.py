import sys
import logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineE2ETest:
    """End-to-end pipeline testing."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
    
    def run_test(self, test_name: str, test_func):
        """Execute a test and record results."""
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            test_func()
            self.test_results.append((test_name, True, None))
            logger.info(f"✓ PASSED: {test_name}")
            return True
        except AssertionError as e:
            self.test_results.append((test_name, False, str(e)))
            self.failed_tests.append(test_name)
            logger.error(f"✗ FAILED: {test_name}")
            logger.error(f"  Reason: {e}")
            return False
        except Exception as e:
            self.test_results.append((test_name, False, f"Exception: {e}"))
            self.failed_tests.append(test_name)
            logger.error(f"✗ ERROR: {test_name}")
            logger.error(f"  Exception: {e}")
            return False
    
    # TEST 1: Configuration Consistency
    
    def test_config_consistency(self):
        """Test configuration values are consistent."""
        from src.core.config import (
            NUM_LOBES, LOBE_MAPPING, NUM_TEMPORAL_FEATURES,
            NUM_SPATIAL_FEATURES, GNN_IN_CHANNELS, LOBE_NAMES
        )
        
        # Test 1.1: LOBE_MAPPING size matches NUM_LOBES
        assert len(LOBE_MAPPING) == NUM_LOBES, \
            f"LOBE_MAPPING has {len(LOBE_MAPPING)} entries but NUM_LOBES={NUM_LOBES}"
        
        # Test 1.2: LOBE_NAMES size matches NUM_LOBES
        assert len(LOBE_NAMES) == NUM_LOBES, \
            f"LOBE_NAMES has {len(LOBE_NAMES)} entries but NUM_LOBES={NUM_LOBES}"
        
        # Test 1.3: GNN_IN_CHANNELS matches feature count
        expected_channels = NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES
        assert GNN_IN_CHANNELS == expected_channels, \
            f"GNN_IN_CHANNELS={GNN_IN_CHANNELS} but expected {expected_channels}"
        
        # Test 1.4: LOBE_MAPPING covers all AAL3 ROIs
        all_rois = set()
        for lobe_id, roi_list in LOBE_MAPPING.items():
            for roi in roi_list:
                assert roi not in all_rois, f"Duplicate ROI {roi} in mapping"
                all_rois.add(roi)
        
        expected_rois = set(range(1, 171))
        assert all_rois == expected_rois, \
            f"ROI coverage incomplete: {len(all_rois)}/170"
        
        logger.info(f"  ✓ NUM_LOBES = {NUM_LOBES}")
        logger.info(f"  ✓ GNN_IN_CHANNELS = {GNN_IN_CHANNELS}")
        logger.info(f"  ✓ LOBE_MAPPING complete (170 ROIs)")
    
    # TEST 2: Data Loading
    
    def test_data_loading(self):
        """Test dataset can be loaded."""
        from src.features.graph_factory import ABIDECausalDataset
        
        # Test 2.1: Load training dataset
        train_dataset = ABIDECausalDataset(split='train')
        assert len(train_dataset) > 0, "Training dataset is empty"
        
        # Test 2.2: Load first sample
        sample = train_dataset[0]
        assert sample is not None, "First training sample is None"
        
        # Test 2.3: Validate sample structure
        from src.core.config import NUM_LOBES, GNN_IN_CHANNELS
        
        assert hasattr(sample, 'x'), "Sample missing node features (x)"
        assert hasattr(sample, 'edge_index'), "Sample missing edge index"
        assert hasattr(sample, 'edge_attr'), "Sample missing edge attributes"
        assert hasattr(sample, 'y'), "Sample missing label"
        
        # Test 2.4: Validate shapes
        assert sample.x.shape == (NUM_LOBES, GNN_IN_CHANNELS), \
            f"Wrong node feature shape: {sample.x.shape}, expected ({NUM_LOBES}, {GNN_IN_CHANNELS})"
        
        assert sample.edge_index.shape[0] == 2, \
            f"Wrong edge_index format: {sample.edge_index.shape}"
        
        assert sample.edge_index.shape[1] > 0, \
            "Graph has zero edges"
        
        assert sample.y.shape == (1,), \
            f"Wrong label shape: {sample.y.shape}"
        
        logger.info(f"  ✓ Loaded {len(train_dataset)} training samples")
        logger.info(f"  ✓ Sample shape: x={sample.x.shape}, edges={sample.edge_index.shape[1]}")
    
    # TEST 3: Model Architecture
    
    def test_model_architecture(self):
        """Test GNN model can be instantiated and run forward pass."""
        from src.models.causal_gnn import CausalBrainGNN
        from src.core.config import GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS_TUNED
        
        # Test 3.1: Instantiate model
        model = CausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=GNN_HIDDEN_CHANNELS_TUNED,
            num_classes=2,
            dropout=0.5,
            num_heads=2,
            use_site_embedding=True,
            use_demographics=True
        )
        
        assert model is not None, "Model instantiation failed"
        
        # Test 3.2: Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model has no parameters"
        assert trainable_params == total_params, "Some parameters are frozen"
        
        # Test 3.3: Forward pass with dummy data
        from src.core.config import NUM_LOBES
        
        batch_size = 4
        num_edges = 12
        
        x = torch.randn(NUM_LOBES * batch_size, GNN_IN_CHANNELS)
        edge_index = torch.randint(0, NUM_LOBES * batch_size, (2, num_edges * batch_size))
        edge_attr = torch.randn(num_edges * batch_size, 1)
        batch = torch.repeat_interleave(torch.arange(batch_size), NUM_LOBES)
        site_id = torch.randint(0, 20, (batch_size,))
        age = torch.randn(batch_size)
        sex = torch.randn(batch_size)
        fiq = torch.randn(batch_size)
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_attr, batch, site_id, age, sex, fiq)
        
        assert out.shape == (batch_size, 2), \
            f"Wrong output shape: {out.shape}, expected ({batch_size}, 2)"
        
        # Test 3.4: Verify output is valid probabilities after softmax
        probs = torch.softmax(out, dim=1)
        assert torch.all(probs >= 0) and torch.all(probs <= 1), \
            "Invalid probabilities"
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), \
            "Probabilities don't sum to 1"
        
        logger.info(f"  ✓ Model has {trainable_params:,} trainable parameters")
        logger.info(f"  ✓ Forward pass successful: {batch_size} samples → {out.shape}")
    
    # TEST 4: Feature Extraction
    
    def test_feature_dimensions(self):
        """Test feature extraction produces correct dimensions."""
        from src.core.config import (
            NODE_ATTRIBUTES_TEMPORAL, NODE_ATTRIBUTES_HARMONIZED,
            NODE_FEATURES_3D, NUM_LOBES, NUM_TEMPORAL_FEATURES, NUM_SPATIAL_FEATURES
        )
        
        # Test 4.1: Temporal features exist
        assert NODE_ATTRIBUTES_TEMPORAL.exists(), \
            "Temporal features file not found"
        
        temporal_df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
        temporal_cols = [c for c in temporal_df.columns if c != 'subject_id']
        
        # Expect 170 ROIs × 8 features = 1360 columns
        expected_temporal = 170 * NUM_TEMPORAL_FEATURES
        assert len(temporal_cols) == expected_temporal, \
            f"Wrong temporal feature count: {len(temporal_cols)}, expected {expected_temporal}"
        
        logger.info(f"  ✓ Temporal features: {len(temporal_df)} subjects × {len(temporal_cols)} features")
        
        # Test 4.2: Spatial features exist
        if NODE_FEATURES_3D.exists():
            spatial_df = pd.read_csv(NODE_FEATURES_3D)
            
            # Should have NUM_LOBES × NUM_SPATIAL_FEATURES columns
            expected_spatial = NUM_LOBES * NUM_SPATIAL_FEATURES
            
            # Exclude metadata columns: subject_id, split, DX_GROUP, AGE_AT_SCAN, SEX, SITE_ID, FIQ, HANDEDNESS_CATEGORY
            metadata_cols = ['subject_id', 'split', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'SITE_ID', 'FIQ', 'HANDEDNESS_CATEGORY']
            spatial_cols = [c for c in spatial_df.columns if c not in metadata_cols]
            lobe_feature_count = len(spatial_cols)
            
            assert lobe_feature_count == expected_spatial, \
                f"Wrong spatial feature count: {lobe_feature_count}, expected {expected_spatial}"
            
            logger.info(f"  ✓ Spatial features: {len(spatial_df)} subjects × {lobe_feature_count} features")
        
        # Test 4.3: Harmonized features exist and are clean
        if NODE_ATTRIBUTES_HARMONIZED.exists():
            harm_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            harm_cols = [c for c in harm_df.columns if c != 'subject_id']
            
            # Should have NUM_LOBES × NUM_TEMPORAL_FEATURES after aggregation
            expected_harm = NUM_LOBES * NUM_TEMPORAL_FEATURES
            assert len(harm_cols) == expected_harm, \
                f"Wrong harmonized feature count: {len(harm_cols)}, expected {expected_harm}"
            
            # Check for NaN/Inf
            nan_count = harm_df[harm_cols].isna().sum().sum()
            inf_count = np.isinf(harm_df[harm_cols].values).sum()
            
            assert nan_count == 0, f"Harmonized features have {nan_count} NaN values"
            assert inf_count == 0, f"Harmonized features have {inf_count} Inf values"
            
            logger.info(f"  ✓ Harmonized features: {len(harm_df)} subjects × {len(harm_cols)} features (clean)")
    
    # TEST 5: Graph Construction
    
    def test_graph_construction(self):
        """Test graph construction produces valid graphs."""
        from src.core.config import CAUSAL_GRAPHS_DIR, NUM_LOBES
        
        # Test 5.1: Graphs exist
        assert CAUSAL_GRAPHS_DIR.exists(), "Graph directory not found"
        
        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        assert len(graph_files) > 0, "No graph files found"
        
        # Test 5.2: Sample graphs are valid
        sample_size = min(10, len(graph_files))
        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            graph_data = torch.load(graph_file, weights_only=False)
            
            assert 'adj' in graph_data, f"{graph_file.name} missing adjacency matrix"
            
            adj = graph_data['adj']
            
            # Check shape
            assert adj.shape == (NUM_LOBES, NUM_LOBES), \
                f"{graph_file.name} has wrong shape: {adj.shape}"
            
            # Check for NaN/Inf
            assert not torch.isnan(adj).any(), f"{graph_file.name} contains NaN"
            assert not torch.isinf(adj).any(), f"{graph_file.name} contains Inf"
            
            # Check edge count
            num_edges = (adj != 0).sum().item()
            assert num_edges > 0, f"{graph_file.name} has zero edges"
        
        logger.info(f"  ✓ {len(graph_files)} graphs validated")
        logger.info(f"  ✓ Sample size: {sample_size}, all valid")
    
    # TEST 6: Training Pipeline
    
    def test_training_utilities(self):
        """Test training utility classes."""
        from src.models.training_utils import (
            EarlyStopping, WarmupScheduler, TrainingTracker, CheckpointManager
        )
        
        # Test 6.1: EarlyStopping
        early_stop = EarlyStopping(patience=3, mode='max')
        
        # Should not stop on improvements
        assert not early_stop(0.5), "Stopped on first value"
        assert not early_stop(0.6), "Stopped on improvement"
        assert not early_stop(0.65), "Stopped on improvement"
        
        # Should stop after patience epochs without improvement
        assert not early_stop(0.64), "Stopped too early (1/3)"
        assert not early_stop(0.63), "Stopped too early (2/3)"
        assert early_stop(0.62), "Didn't stop after patience"
        
        logger.info("  ✓ EarlyStopping works correctly")
        
        # Test 6.2: TrainingTracker
        tracker = TrainingTracker(k_folds=5)
        
        tracker.add_fold_result(
            fold=0, auc=0.75, f1=0.70, acc=0.68,
            threshold=0.55, best_epoch=45
        )
        tracker.add_fold_result(
            fold=1, auc=0.80, f1=0.72, acc=0.70,
            threshold=0.52, best_epoch=38
        )
        
        summary = tracker.get_summary()
        
        assert summary['mean_auc'] == 0.775, "Wrong mean AUC"
        assert len(summary['per_fold_aucs']) == 2, "Wrong fold count"
        
        logger.info("  ✓ TrainingTracker works correctly")
    
    # TEST 7: End-to-End Data Flow
    
    def test_end_to_end_dataflow(self):
        """Test complete data flow from loading to model inference."""
        from src.features.graph_factory import ABIDECausalDataset
        from src.models.causal_gnn import CausalBrainGNN
        from src.core.config import GNN_IN_CHANNELS, GNN_HIDDEN_CHANNELS_TUNED
        from torch_geometric.loader import DataLoader
        
        # Test 7.1: Load dataset
        dataset = ABIDECausalDataset(split='test')
        
        # Get first valid sample
        sample = None
        for i in range(min(10, len(dataset))):
            s = dataset[i]
            if s is not None:
                sample = s
                break
        
        assert sample is not None, "No valid samples in test set"
        
        # Test 7.2: Create data loader
        loader = DataLoader([sample], batch_size=1)
        
        # Test 7.3: Initialize model
        model = CausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=GNN_HIDDEN_CHANNELS_TUNED,
            num_classes=2,
            use_site_embedding=True,
            use_demographics=True
        )
        model.eval()
        
        # Test 7.4: Inference
        for batch in loader:
            with torch.no_grad():
                out = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                    getattr(batch, 'site_id', None),
                    getattr(batch, 'age', None),
                    getattr(batch, 'sex', None),
                    getattr(batch, 'fiq', None)
                )
            
            # Validate output
            assert out.shape == (1, 2), f"Wrong output shape: {out.shape}"
            
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1)
            
            assert pred.item() in [0, 1], f"Invalid prediction: {pred.item()}"
            
            logger.info(f"  ✓ End-to-end inference successful")
            logger.info(f"    Input: {batch.x.shape} → Output: {out.shape}")
            logger.info(f"    Prediction: {pred.item()} (prob={probs[0, pred].item():.3f})")
            
            break  # Only test first batch
    
    # REPORTING
    
    def generate_report(self):
        """Generate test report."""
        logger.info("\n" + "="*70)
        logger.info("TEST REPORT")
        logger.info("="*70)
        
        total = len(self.test_results)
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = total - passed
        
        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        
        if failed > 0:
            logger.error("\n❌ FAILED TESTS:")
            for name, success, error in self.test_results:
                if not success:
                    logger.error(f"  • {name}")
                    if error:
                        logger.error(f"    {error}")
        
        logger.info("\n" + "="*70)
        
        if failed == 0:
            logger.info("✅ ALL TESTS PASSED")
            return True
        else:
            logger.error(f"❌ {failed}/{total} TESTS FAILED")
            return False
    
    def run_all_tests(self):
        """Execute all tests."""
        logger.info("="*70)
        logger.info("NEURO-CXG END-TO-END PIPELINE TESTING")
        logger.info("="*70)
        
        # Run tests in order
        self.run_test("Configuration Consistency", self.test_config_consistency)
        self.run_test("Data Loading", self.test_data_loading)
        self.run_test("Model Architecture", self.test_model_architecture)
        self.run_test("Feature Dimensions", self.test_feature_dimensions)
        self.run_test("Graph Construction", self.test_graph_construction)
        self.run_test("Training Utilities", self.test_training_utilities)
        self.run_test("End-to-End Data Flow", self.test_end_to_end_dataflow)
        
        # Generate report
        return self.generate_report()


def main():
    """CLI entry point."""
    tester = PipelineE2ETest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
