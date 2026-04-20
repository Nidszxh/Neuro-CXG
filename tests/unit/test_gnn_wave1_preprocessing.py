"""Unit tests for Wave-1 fold preprocessing helpers in gnn_model.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import gnn_model as gm


class _Sample:
    def __init__(self, x: torch.Tensor, y: int, site_id: int):
        self.x = x.clone().float()
        self.y = torch.tensor([int(y)], dtype=torch.long)
        self.site_id = torch.tensor([int(site_id)], dtype=torch.long)


def _make_train_graphs() -> list:
    rng = np.random.default_rng(42)
    graphs = []
    for i in range(12):
        label = i % 2
        site = 0 if i < 6 else 1
        x = torch.tensor(rng.normal(size=(12, 8)), dtype=torch.float32)
        if label == 1:
            x[:, 0] += 1.5
            x[:, 3] += 1.0
        if site == 1:
            x[:, 1] += 3.0
        graphs.append(_Sample(x=x, y=label, site_id=site))
    return graphs


def test_fit_mi_feature_selection_returns_valid_mask(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    monkeypatch.setattr(gm, "GNN_MI_MIN_KEEP_RATIO", 0.25)
    monkeypatch.setattr(gm, "GNN_MI_MAX_KEEP_RATIO", 0.75)

    train = _make_train_graphs()
    selected_idx, mask, meta = gm._fit_mi_feature_selection(train)

    assert len(selected_idx) >= 2
    assert len(selected_idx) <= 6
    assert mask.shape[0] == 8
    assert int(mask.sum().item()) == len(selected_idx)
    assert meta["selected_features"] == len(selected_idx)


def test_fit_mi_feature_selection_keeps_all_when_scores_near_zero(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    monkeypatch.setattr(gm, "GNN_MI_MIN_KEEP_RATIO", 0.70)
    monkeypatch.setattr(gm, "GNN_MI_MAX_KEEP_RATIO", 1.00)

    def _zero_mi(_X, _y, random_state=None, n_neighbors=None):
        return np.zeros(8, dtype=np.float64)

    monkeypatch.setattr(gm, "mutual_info_classif", _zero_mi)

    train = _make_train_graphs()
    selected_idx, mask, meta = gm._fit_mi_feature_selection(train)

    assert len(selected_idx) == 8
    assert int(mask.sum().item()) == 8
    assert meta["selected_features"] == 8
    assert meta["score_max"] == 0.0


def test_fit_mi_feature_selection_keeps_all_for_single_class_fold(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    monkeypatch.setattr(gm, "GNN_MI_MIN_KEEP_RATIO", 0.70)
    monkeypatch.setattr(gm, "GNN_MI_MAX_KEEP_RATIO", 1.00)

    train = _make_train_graphs()
    for sample in train:
        sample.y = torch.tensor([1], dtype=torch.long)

    selected_idx, mask, meta = gm._fit_mi_feature_selection(train)

    assert len(selected_idx) == 8
    assert int(mask.sum().item()) == 8
    assert meta["selected_features"] == 8
    assert meta["candidate_k"] == 8


def test_apply_feature_mask_zeros_unselected_channels(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    samples = _make_train_graphs()[:2]
    original = samples[0].x.clone()
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)

    gm._apply_feature_mask(samples, mask)

    assert torch.allclose(samples[0].x[:, 1], torch.zeros_like(samples[0].x[:, 1]))
    assert torch.allclose(samples[0].x[:, 3], torch.zeros_like(samples[0].x[:, 3]))
    assert not torch.allclose(samples[0].x[:, 0], torch.zeros_like(samples[0].x[:, 0]))
    assert torch.allclose(samples[0].x[:, 0], original[:, 0])


def test_within_site_normalization_uses_global_fallback_for_unseen_site(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    train = _make_train_graphs()
    site_stats, global_stats = gm._fit_site_normalization_stats(train)

    unseen = _Sample(x=torch.ones(12, 8) * 5.0, y=0, site_id=999)
    gm._apply_site_normalization([unseen], site_stats, global_stats)

    global_mean, global_std = global_stats
    expected = (torch.ones(12, 8) * 5.0 - global_mean) / global_std
    assert torch.allclose(unseen.x, expected, atol=1e-5)


def test_site_stats_serialization_roundtrip(monkeypatch):
    monkeypatch.setattr(gm, "GNN_IN_CHANNELS", 8)
    train = _make_train_graphs()
    site_stats, _ = gm._fit_site_normalization_stats(train)
    means, stds = gm._site_stats_to_serializable(site_stats)

    assert set(means.keys()) == set(stds.keys())
    for key in means:
        assert len(means[key]) == 8
        assert len(stds[key]) == 8
