"""Unit tests for checkpoint preprocessing metadata attachment."""

from __future__ import annotations

import torch

from src.models.training_utils import attach_feature_scaler_from_checkpoint


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

def test_attach_legacy_scaler_metadata():
    model = _DummyModel()
    ckpt = {
        "feature_mean": [0.1, 0.2, 0.3, 0.4],
        "feature_std": [1.0, 1.1, 1.2, 1.3],
    }

    attached = attach_feature_scaler_from_checkpoint(model, ckpt, expected_dim=4)
    assert attached is True
    assert hasattr(model, "_feature_mean")
    assert hasattr(model, "_feature_std")
    assert model._feature_mean.shape[0] == 4
    assert model._feature_std.shape[0] == 4

def test_attach_wave1_preprocessing_metadata():
    model = _DummyModel()
    ckpt = {
        "preprocessing_mode": "wave1",
        "site_normalization_mode": "within_site",
        "feature_mean": [0.0, 0.0, 0.0, 0.0],
        "feature_std": [1.0, 1.0, 1.0, 1.0],
        "feature_mask": [1.0, 0.0, 1.0, 0.0],
        "selected_feature_idx": [0, 2],
        "site_feature_means": {
            "0": [0.1, 0.1, 0.1, 0.1],
            "2": [0.2, 0.2, 0.2, 0.2],
        },
        "site_feature_stds": {
            "0": [1.0, 1.0, 1.0, 1.0],
            "2": [1.1, 1.1, 1.1, 1.1],
        },
    }

    attached = attach_feature_scaler_from_checkpoint(model, ckpt, expected_dim=4)
    assert attached is True
    assert hasattr(model, "_feature_mask")
    assert hasattr(model, "_selected_feature_idx")
    assert hasattr(model, "_site_feature_means")
    assert hasattr(model, "_site_feature_stds")
    assert getattr(model, "_preprocessing_mode", None) == "wave1"
    assert getattr(model, "_site_normalization_mode", None) == "within_site"
    assert model._feature_mask.shape[0] == 4
    assert model._selected_feature_idx == [0, 2]
    assert set(model._site_feature_means.keys()) == {0, 2}
    assert set(model._site_feature_stds.keys()) == {0, 2}

def test_graceful_when_metadata_missing():
    model = _DummyModel()
    attached = attach_feature_scaler_from_checkpoint(model, {}, expected_dim=4)
    assert attached is False
    assert not hasattr(model, "_feature_mean")
    assert not hasattr(model, "_feature_mask")
