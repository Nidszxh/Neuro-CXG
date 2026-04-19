"""
Data Quality Experiments
=========================
Three experiments to audit data quality and identify pipeline bottlenecks:

  1. Cross-site generalisation: Per-site AUC evaluation on the test set.
     If one site dominates, investigate site effects more aggressively.

  2. Subject count audit: How many subjects survive each pipeline stage?
     Identifies the primary bottleneck when below the expected ~1000.

  3. Atlas-centroid spatial baseline: Replace YOLO-detected spatial coords with
     fixed AAL3 atlas centroids. If AUC improves, YOLO spatial noise is hurting.

Usage:
    python -m src.experiments.data_quality                     # all experiments
    python -m src.experiments.data_quality --experiments 1 2   # specific
    python -m src.experiments.data_quality --experiments 3     # atlas baseline only
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    DATA_FINAL,
    DATA_METADATA,
    DATA_PROCESSED,
    DATA_ROOT,
    DEVICE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    GNN_BATCH_SIZE,
    GNN_DROPOUT,
    GNN_EPOCHS,
    GNN_HIDDEN_CHANNELS,
    GNN_IN_CHANNELS,
    GNN_NUM_LAYERS,
    GNN_NUM_HEADS,
    GNN_USE_SITE_EMBEDDING,
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_GRL_ALPHA,
    GNN_EDGE_GATE,
    GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE,
    GNN_POOLING,
    GNN_WEIGHT_DECAY,
    K_FOLDS,
    LOBE_MAPPING,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    RESULTS_DATA_QUALITY_DIR,
    SITE_ROBUSTNESS_GATE_ENABLED,
    SITE_ROBUSTNESS_MIN_SITE_AUC,
    SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION,
    SITE_ROBUSTNESS_MIN_EVALUABLE_SITES,
    SITE_ROBUSTNESS_GATE_POLICY,
    USE_FOCAL_LOSS,
    USE_CLASS_WEIGHTS,
    get_active_checkpoint_dir,
)
from src.models.losses import FocalLoss
from src.models.factory import build_model
from src.models.training_utils import (
    make_loader,
    train_fold_with_onecycle,
    attach_feature_scaler_from_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = RESULTS_DATA_QUALITY_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: CROSS-SITE GENERALISATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_predictions(model, loader, device=DEVICE):
    """Collect (probs, labels, site_ids) from a DataLoader."""
    model.eval()
    all_probs, all_labels, all_sites = [], [], []
    for data in loader:
        if data is None:
            continue
        data = data.to(device)
        out = model.forward_batch(data) if hasattr(model, "forward_batch") else model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            getattr(data, "site_id", None),
            getattr(data, "age", None),
            getattr(data, "sex", None),
            getattr(data, "fiq", None),
        )
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = data.y.cpu().numpy()
        sites = getattr(data, "site_id", None)
        sites = sites.cpu().numpy() if sites is not None else np.zeros(len(probs))
        all_probs.append(probs)
        all_labels.append(labels)
        all_sites.append(sites)
    if not all_probs:
        return np.array([]), np.array([]), np.array([])
    return (np.concatenate(all_probs),
            np.concatenate(all_labels),
            np.concatenate(all_sites))


def experiment_cross_site_auc() -> pd.DataFrame:
    """
    Loads the best checkpoint from Fold 0 (representative model) and evaluates
    per-site AUC on the TEST set.  Reports sites with AUC < 0.55 (site-driven
    failure) and > 0.70 (strong local signal).
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: CROSS-SITE GENERALISATION (TEST SET)")
    logger.info("=" * 70)

    from src.features.graph_factory import ABIDECausalDataset

    # ── Load test dataset ─────────────────────────────────────────────────────
    try:
        test_ds = ABIDECausalDataset(split="test")
    except Exception as e:
        logger.error(f"Cannot load test dataset: {e}")
        return pd.DataFrame()

    if len(test_ds) == 0:
        logger.warning("Test dataset is empty — cross-site evaluation skipped")
        return pd.DataFrame()

    test_data = [test_ds[i] for i in range(len(test_ds)) if test_ds[i] is not None]
    test_loader = make_loader(test_data, batch_size=GNN_BATCH_SIZE)

    # ── Load checkpoint (fold 0, representative) ──────────────────────────────
    checkpoint_path = get_active_checkpoint_dir() / "best_model_fold0.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint fold0 not found: {checkpoint_path}")
        return pd.DataFrame()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    site_dim = 16 if GNN_USE_SITE_EMBEDDING else 0
    saved_in_features = state_dict["lin_in.weight"].shape[1]
    node_emb_dim = saved_in_features - GNN_IN_CHANNELS - site_dim

    model = build_model(
        device=DEVICE,
        use_site_embedding=GNN_USE_SITE_EMBEDDING,
        use_demographics=GNN_USE_DEMOGRAPHICS,
        use_grl=GNN_USE_GRL,
        grl_alpha=GNN_GRL_ALPHA,
        edge_gate=GNN_EDGE_GATE,
        node_emb_dim=node_emb_dim,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Checkpoint missing keys: %s", missing)
    if unexpected:
        logger.warning("Checkpoint unexpected keys: %s", unexpected)

    attach_feature_scaler_from_checkpoint(model, checkpoint, expected_dim=GNN_IN_CHANNELS)
    decision_threshold = float(checkpoint.get("threshold", 0.5)) if isinstance(checkpoint, dict) else 0.5
    if not np.isfinite(decision_threshold):
        decision_threshold = 0.5

    # ── Collect predictions ───────────────────────────────────────────────────
    probs, labels, site_ids = _collect_predictions(model, test_loader)
    if len(probs) == 0:
        logger.warning("No predictions collected")
        return pd.DataFrame()

    # ── Recover site names from manifest ─────────────────────────────────────
    # site_id in dataset is an integer index — map back via manifest
    manifest = pd.read_csv(MASTER_MANIFEST)
    site_names = sorted(manifest["SITE_ID"].unique())
    idx_to_site = {i: s for i, s in enumerate(site_names)}

    # Overall AUC
    overall_auc = roc_auc_score(labels, probs)
    logger.info(f"\n  Overall test AUC  : {overall_auc:.4f}  (n={len(labels)})")
    logger.info(
        "  Overall test F1   : %.4f  (threshold=%.3f)",
        f1_score(labels, (probs >= decision_threshold).astype(int), zero_division=0),
        decision_threshold,
    )

    # ── Per-site breakdown ────────────────────────────────────────────────────
    rows = []
    logger.info(f"\n  {'Site':<25} {'N':>5} {'Ctrl':>5} {'ASD':>5} {'AUC':>7}  Notes")
    logger.info("  " + "-" * 62)

    for site_idx in sorted(np.unique(site_ids).astype(int)):
        mask = (site_ids == site_idx)
        site_labels = labels[mask]
        site_probs  = probs[mask]
        site_name   = idx_to_site.get(site_idx, f"site_{site_idx}")
        n = mask.sum()
        n_ctrl = (site_labels == 0).sum()
        n_asd  = (site_labels == 1).sum()

        if n < 5 or len(np.unique(site_labels)) < 2:
            note = "📌 too few / single class"
            auc_str = "—"
            rows.append({"site": site_name, "n": n, "n_control": n_ctrl,
                         "n_asd": n_asd, "auc": float("nan"), "note": "skip"})
        else:
            site_auc = roc_auc_score(site_labels, site_probs)
            auc_str = f"{site_auc:.4f}"
            if site_auc >= 0.70:
                note = "✓ strong signal"
            elif site_auc < 0.55:
                note = "⚠  site-driven failure"
            else:
                note = ""
            rows.append({"site": site_name, "n": n, "n_control": n_ctrl,
                         "n_asd": n_asd, "auc": site_auc, "note": note})

        logger.info(f"  {site_name:<25} {n:>5} {n_ctrl:>5} {n_asd:>5} {auc_str:>7}  {note}")

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "cross_site_auc.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\n  Saved → {out_csv}")

    # Flag problematic sites
    bad_sites = df[df["auc"] < 0.55].dropna(subset=["auc"])
    if len(bad_sites) > 0:
        logger.warning(
            f"\n  ⚠ {len(bad_sites)} sites with AUC < 0.55: "
            f"{bad_sites['site'].tolist()}\n"
            "  → Investigate site-specific scanner parameters or demographics."
        )

    return df


def _apply_site_robustness_gate(site_auc_df: pd.DataFrame) -> Dict[str, object]:
    """Apply configurable site-robustness gate to cross-site AUC output."""
    result = {
        "enabled": bool(SITE_ROBUSTNESS_GATE_ENABLED),
        "policy": str(SITE_ROBUSTNESS_GATE_POLICY),
        "min_site_auc": float(SITE_ROBUSTNESS_MIN_SITE_AUC),
        "max_weak_site_fraction": float(SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION),
        "min_evaluable_sites": int(SITE_ROBUSTNESS_MIN_EVALUABLE_SITES),
        "evaluable_sites": 0,
        "weak_sites": 0,
        "weak_site_fraction": None,
        "status": "skipped",
    }

    if not SITE_ROBUSTNESS_GATE_ENABLED:
        return result

    policy = str(SITE_ROBUSTNESS_GATE_POLICY).strip().lower()
    if policy not in {"warn", "fail"}:
        logger.warning(
            "Unknown SITE_ROBUSTNESS_GATE_POLICY=%r; falling back to 'warn'",
            SITE_ROBUSTNESS_GATE_POLICY,
        )
        policy = "warn"
        result["policy"] = policy

    evaluable = site_auc_df.dropna(subset=["auc"]) if not site_auc_df.empty else pd.DataFrame()
    n_eval = int(len(evaluable))
    n_weak = int((evaluable["auc"] < SITE_ROBUSTNESS_MIN_SITE_AUC).sum()) if n_eval else 0
    weak_frac = float(n_weak / max(n_eval, 1))

    result.update(
        {
            "evaluable_sites": n_eval,
            "weak_sites": n_weak,
            "weak_site_fraction": weak_frac,
        }
    )

    logger.info(
        "Site robustness gate: evaluable_sites=%d, weak_sites=%d (AUC < %.2f), weak_fraction=%.2f, threshold=%.2f",
        n_eval,
        n_weak,
        SITE_ROBUSTNESS_MIN_SITE_AUC,
        weak_frac,
        SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION,
    )

    if n_eval < SITE_ROBUSTNESS_MIN_EVALUABLE_SITES:
        msg = (
            "Site robustness gate has insufficient evaluable sites: "
            f"{n_eval} < {SITE_ROBUSTNESS_MIN_EVALUABLE_SITES}"
        )
        result["status"] = "insufficient-sites"
        if policy == "fail":
            raise RuntimeError(msg)
        logger.warning(msg)
        return result

    if weak_frac > SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION:
        msg = (
            "Site robustness gate failed: weak-site fraction exceeds threshold "
            f"({weak_frac:.2f} > {SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION:.2f})"
        )
        result["status"] = "failed"
        if policy == "fail":
            raise RuntimeError(msg)
        logger.warning(msg)
        return result

    result["status"] = "passed"
    logger.info("Site robustness gate passed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: SUBJECT COUNT AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def experiment_subject_count_audit() -> pd.DataFrame:
    """
    Count subjects surviving each pipeline stage and identify the bottleneck.

    Stages:
      0. Raw phenotype CSV
      1. PNG images downloaded (data/images/ or data/final/*/images/)
      2. All 12 regions detected  (node_features_3d.csv)
      3. Time series available    (.npy files across train/val/test)
      4. Temporal features        (node_attributes_temporal.csv)
      5. Harmonized features      (node_attributes_harmonized.csv)
      6. Causal graphs built      (causal_graphs/*.pt)
      7. Train dataset loaded     (ABIDECausalDataset('train'))
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: SUBJECT COUNT AUDIT (PIPELINE BOTTLENECK)")
    logger.info("=" * 70)

    counts = {}

    # Stage 0: Phenotype CSV
    pheno_path = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"
    if pheno_path.exists():
        pheno = pd.read_csv(pheno_path)
        counts["0_phenotype_csv"] = len(pheno)
    else:
        counts["0_phenotype_csv"] = 0
        logger.warning(f"  Phenotype CSV not found: {pheno_path}")

    # Stage 1: PNG images downloaded
    # Count unique subjects from PNG filenames (format: {sub_id}_z{depth}.png)
    png_subjects = set()
    for split in ["train", "val", "test"]:
        img_dir = DATA_FINAL / split / "images"
        if img_dir.exists():
            for f in img_dir.glob("*.png"):
                sub = f.stem.rsplit("_z", 1)[0]
                png_subjects.add(sub)
    # Also check flat images directory
    flat_img = DATA_ROOT / "images"
    if flat_img.exists():
        for f in flat_img.glob("*.png"):
            sub = f.stem.rsplit("_z", 1)[0]
            png_subjects.add(sub)
    counts["1_png_downloaded"] = len(png_subjects)

    # Stage 2: All 12 regions detected
    if NODE_FEATURES_3D.exists():
        nf3d = pd.read_csv(NODE_FEATURES_3D)
        counts["2_all12_detected"] = len(nf3d["subject_id"].unique()) if "subject_id" in nf3d.columns else len(nf3d)
    else:
        counts["2_all12_detected"] = 0
        logger.warning(f"  node_features_3d.csv not found: {NODE_FEATURES_3D}")

    # Stage 3: Time series available
    ts_subjects = set()
    for split in ["train", "val", "test"]:
        ts_dir = DATA_FINAL / split / "time_series"
        if ts_dir.exists():
            for f in ts_dir.glob("*_ts.npy"):
                ts_subjects.add(f.stem.replace("_ts", ""))
    # Also flat processed directory
    for f in DATA_PROCESSED.glob("*_ts.npy"):
        ts_subjects.add(f.stem.replace("_ts", ""))
    counts["3_time_series_npy"] = len(ts_subjects)

    # Stage 4: Temporal features CSV
    if NODE_ATTRIBUTES_TEMPORAL.exists():
        temp = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
        counts["4_temporal_features"] = len(temp["subject_id"].unique()) if "subject_id" in temp.columns else len(temp)
    else:
        counts["4_temporal_features"] = 0

    # Stage 5: Harmonized features CSV
    if NODE_ATTRIBUTES_HARMONIZED.exists():
        harm = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
        counts["5_harmonized"] = len(harm["subject_id"].unique()) if "subject_id" in harm.columns else len(harm)
    else:
        counts["5_harmonized"] = 0

    # Stage 6: Causal graphs built
    if CAUSAL_GRAPHS_DIR.exists():
        counts["6_causal_graphs"] = sum(1 for _ in CAUSAL_GRAPHS_DIR.glob("*.pt"))
    else:
        counts["6_causal_graphs"] = 0

    # Stage 7: Train dataset loadable
    try:
        from src.features.graph_factory import ABIDECausalDataset
        train_ds = ABIDECausalDataset(split="train")
        counts["7_train_dataset"] = len(train_ds)
    except Exception as e:
        logger.warning(f"  ABIDECausalDataset('train') failed: {e}")
        counts["7_train_dataset"] = 0

    # ── Print table ───────────────────────────────────────────────────────────
    logger.info(f"\n  {'Stage':<35} {'Count':>7}  {'Drop':>7}  {'Loss%':>7}")
    logger.info("  " + "-" * 60)

    prev = None
    rows = []
    for stage, count in counts.items():
        drop = (prev - count) if (prev is not None and prev > 0) else 0
        pct  = (drop / max(prev, 1) * 100) if prev else 0
        flag = "  ←  ⚠ BOTTLENECK" if drop > 50 and pct > 20 else ""
        logger.info(f"  {stage:<35} {count:>7}  {drop:>7}  {pct:>6.1f}%{flag}")
        rows.append({"stage": stage, "count": count, "drop": drop, "loss_pct": round(pct, 1)})
        prev = count

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "subject_count_audit.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\n  Saved → {out_csv}")

    # Flag the biggest bottleneck
    bottleneck = df.sort_values("drop", ascending=False).iloc[0]
    if bottleneck["drop"] > 0:
        logger.warning(
            f"\n  Largest drop: Stage '{bottleneck['stage']}' "
            f"loses {bottleneck['drop']} subjects ({bottleneck['loss_pct']:.1f}%)"
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: ATLAS-CENTROID SPATIAL BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_atlas_centroids() -> Optional[np.ndarray]:
    """
    Compute per-lobe atlas centroids from roi_centroids.json or AAL3 atlas.
    Returns (NUM_LOBES, 6) array matching the spatial feature layout:
      [mean_x, mean_y, mean_z, mean_size, 0, 1]  (conf_std=0, det_count=1 as fixed defaults)
    Returns None if neither source is available.
    """
    import json
    centroid_path = DATA_METADATA / "roi_centroids.json"

    def _normalize_roi_centroids(payload) -> Dict[int, List[float]]:
        """Convert list/dict centroid payloads into roi_id -> [x, y, z]."""
        normalized: Dict[int, List[float]] = {}

        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                roi_id = entry.get("roi_id")
                if roi_id is None:
                    continue
                coords = [entry.get("x"), entry.get("y"), entry.get("z")]
                if all(value is not None for value in coords):
                    normalized[int(roi_id)] = [float(coords[0]), float(coords[1]), float(coords[2])]
            return normalized

        if isinstance(payload, dict):
            for key, entry in payload.items():
                roi_id = None
                if isinstance(entry, dict):
                    roi_id = entry.get("roi_id")
                    coords = [entry.get("x"), entry.get("y"), entry.get("z")]
                    if roi_id is None and str(key).isdigit():
                        roi_id = int(key)
                    if roi_id is not None and all(value is not None for value in coords):
                        normalized[int(roi_id)] = [float(coords[0]), float(coords[1]), float(coords[2])]
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3 and str(key).isdigit():
                    normalized[int(key)] = [float(entry[0]), float(entry[1]), float(entry[2])]
            return normalized

        return normalized

    if centroid_path.exists():
        logger.info(f"  Using ROI centroids from {centroid_path}")
        with open(centroid_path) as f:
            roi_cents = _normalize_roi_centroids(json.load(f))

        lobe_centroids = np.zeros((NUM_LOBES, NUM_SPATIAL_FEATURES), dtype=np.float32)
        for lobe_id, roi_indices in LOBE_MAPPING.items():
            # roi_indices are 0-based; ROI centroids may be 1-based or 0-based.
            coords = []
            for roi_idx in roi_indices:
                entry = roi_cents.get(int(roi_idx + 1), roi_cents.get(int(roi_idx), None))
                if entry is not None:
                    coords.append(entry[:3])
            if coords:
                arr = np.array(coords, dtype=np.float32)
                cx, cy, cz = arr.mean(axis=0)
                lobe_centroids[lobe_id, 0] = cx
                lobe_centroids[lobe_id, 1] = cy
                lobe_centroids[lobe_id, 2] = cz
                if NUM_SPATIAL_FEATURES > 3:
                    lobe_centroids[lobe_id, 3] = float(len(coords))  # size proxy
                if NUM_SPATIAL_FEATURES > 4:
                    lobe_centroids[lobe_id, 4] = 0.0                 # conf_std = 0
                if NUM_SPATIAL_FEATURES > 5:
                    lobe_centroids[lobe_id, 5] = 1.0                 # detection_count = 1
        return lobe_centroids

    # Attempt to compute from atlas NIfTI if nibabel is available
    try:
        import nibabel as nib
        from src.core.config import ATLAS_PATH
        if ATLAS_PATH.exists():
            logger.info(f"  Computing centroids from AAL3 atlas: {ATLAS_PATH}")
            img = nib.load(str(ATLAS_PATH))
            data_arr = np.array(img.get_fdata(), dtype=np.int32)
            affine = img.affine

            # Compute voxel centroid for each ROI, then map to world space
            roi_mni: Dict[int, List[float]] = {}
            for roi_id_1based in range(1, 171):
                mask = (data_arr == roi_id_1based)
                if not mask.any():
                    continue
                voxels = np.argwhere(mask)
                vox_centroid = voxels.mean(axis=0)
                world_xyz = nib.affines.apply_affine(affine, vox_centroid)
                roi_mni[roi_id_1based - 1] = list(world_xyz)  # 0-based key

            lobe_centroids = np.zeros((NUM_LOBES, NUM_SPATIAL_FEATURES), dtype=np.float32)
            for lobe_id, roi_indices in LOBE_MAPPING.items():
                coords = [roi_mni[r] for r in roi_indices if r in roi_mni]
                if coords:
                    arr = np.array(coords, dtype=np.float32)
                    cx, cy, cz = arr.mean(axis=0)
                    lobe_centroids[lobe_id, :3] = [cx, cy, cz]
                    if NUM_SPATIAL_FEATURES > 3:
                        lobe_centroids[lobe_id, 3] = float(len(coords))
                    if NUM_SPATIAL_FEATURES > 4:
                        lobe_centroids[lobe_id, 4] = 0.0
                    if NUM_SPATIAL_FEATURES > 5:
                        lobe_centroids[lobe_id, 5] = 1.0
            return lobe_centroids
    except ImportError:
        logger.warning("  nibabel not available — atlas centroid extraction skipped")
    except Exception as e:
        logger.warning(f"  Could not compute atlas centroids from NIfTI: {e}")

    return None


class AtlasCentroidDataset:
    """
    Wraps ABIDECausalDataset and replaces per-subject YOLO spatial features with
    fixed atlas-derived centroids (same for every subject in the same lobe).
    """
    _SPATIAL_START: int = GNN_IN_CHANNELS - NUM_SPATIAL_FEATURES  # e.g. 22

    def __init__(self, base_dataset, atlas_centroids: np.ndarray):
        self.ds = base_dataset
        # atlas_centroids: (NUM_LOBES, NUM_SPATIAL_FEATURES), normalised
        self._centroids = torch.tensor(atlas_centroids, dtype=torch.float32)
        logger.info(f"  AtlasCentroidDataset: spatial features replaced with atlas centroids")
        logger.info(f"  Centroid range: x=[{atlas_centroids[:,0].min():.1f}, {atlas_centroids[:,0].max():.1f}]  "
                    f"y=[{atlas_centroids[:,1].min():.1f}, {atlas_centroids[:,1].max():.1f}]  "
                    f"z=[{atlas_centroids[:,2].min():.1f}, {atlas_centroids[:,2].max():.1f}]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if sample is None:
            return None
        sample = sample.clone()
        sl = slice(self._SPATIAL_START, GNN_IN_CHANNELS)
        sample.x[:, sl] = self._centroids  # broadcast: same for all nodes
        return sample

    def get(self, idx):
        return self[idx]


def experiment_atlas_centroid_baseline() -> Dict:
    """
    Runs 5-fold CV using atlas-centroid spatial features instead of YOLO detections.
    Returns result dict with mean_auc etc.
    """
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: ATLAS-CENTROID SPATIAL BASELINE")
    logger.info("=" * 70)

    atlas_centroids = _compute_atlas_centroids()
    if atlas_centroids is None:
        logger.error(
            "  Cannot compute atlas centroids: no roi_centroids.json and nibabel unavailable.\n"
            "  Run `python src/pipelines/generate_labels.py` first to generate roi_centroids.json."
        )
        return {}

    # Normalise to roughly [-1, 1] (MNI space: roughly ±100mm range)
    for col in range(3):  # x, y, z
        rng = atlas_centroids[:, col].max() - atlas_centroids[:, col].min()
        if rng > 0:
            atlas_centroids[:, col] = (atlas_centroids[:, col] - atlas_centroids[:, col].mean()) / (rng / 2)

    from src.features.graph_factory import ABIDECausalDataset

    base_ds = ABIDECausalDataset(split="train")
    atlas_ds = AtlasCentroidDataset(base_ds, atlas_centroids)

    def gnn_factory():
        return build_model(
            device=DEVICE,
            use_site_embedding=GNN_USE_SITE_EMBEDDING,
            use_demographics=GNN_USE_DEMOGRAPHICS,
            use_grl=GNN_USE_GRL,
            grl_alpha=GNN_GRL_ALPHA,
            edge_gate=GNN_EDGE_GATE,
        )

    # Collect labels
    labels = []
    for i in range(len(atlas_ds)):
        d = atlas_ds[i]
        if d is not None:
            labels.append(int(d.y.item()))

    if not labels:
        logger.error("  No valid data for atlas baseline experiment")
        return {}

    class_weight_tensor = None
    if USE_CLASS_WEIGHTS:
        labels_arr = np.array(labels)
        n_control = max(int((labels_arr == 0).sum()), 1)
        n_asd = max(int((labels_arr == 1).sum()), 1)
        total = max(len(labels_arr), 1)
        class_weight_tensor = torch.tensor(
            [total / (2 * n_control), total / (2 * n_asd)],
            dtype=torch.float32,
            device=DEVICE,
        )

    if USE_FOCAL_LOSS:
        pos_weight = None
        if USE_CLASS_WEIGHTS:
            labels_arr = np.array(labels)
            n_control = max(int((labels_arr == 0).sum()), 1)
            n_asd = max(int((labels_arr == 1).sum()), 1)
            pos_weight = float(n_control / n_asd)
        criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_aucs: List[float] = []

    logger.info(f"  Total subjects: {len(labels)}")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        t0 = time.time()
        td = [atlas_ds[i] for i in tr_idx if atlas_ds[i] is not None]
        vd = [atlas_ds[i] for i in val_idx if atlas_ds[i] is not None]
        if not td or not vd:
            continue
        tl = make_loader(td, batch_size=GNN_BATCH_SIZE, shuffle=True)
        vl = make_loader(vd, batch_size=GNN_BATCH_SIZE)
        model = gnn_factory().to(DEVICE)
        _, best_metrics, _ = train_fold_with_onecycle(
            model=model, train_loader=tl, val_loader=vl,
            criterion=criterion, device=DEVICE, epochs=GNN_EPOCHS,
            max_lr=GNN_ONECYCLE_MAX_LR, patience=GNN_EARLY_STOPPING_PATIENCE,
            use_grl=True, grl_weight=0.2, fold=fold, weight_decay=GNN_WEIGHT_DECAY,
        )
        auc = best_metrics["auc"]
        fold_aucs.append(auc)
        logger.info(f"  Fold {fold+1}: AUC={auc:.4f}  (elapsed {time.time()-t0:.0f}s)")

    if not fold_aucs:
        logger.error("  All folds failed")
        return {}

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    logger.info(f"\n  Atlas-centroid AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"  Baseline (YOLO)   : 0.6300  (reference)")
    delta = mean_auc - 0.63
    if delta > 0.01:
        logger.warning(
            f"  ⚠ Atlas centroids improve AUC by {delta:+.4f}  → "
            "YOLO spatial noise is hurting performance. "
            "Consider using fixed atlas coordinates as spatial features."
        )
    else:
        logger.info("  ✓ YOLO spatial features are competitive with atlas centroids")

    return {
        "experiment": "Atlas-Centroid Spatial Baseline",
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

EXP_MAP = {
    "1": "Cross-site generalisation",
    "2": "Subject count audit",
    "3": "Atlas-centroid spatial baseline",
}


def main():
    parser = argparse.ArgumentParser(description="Data quality experiments")
    parser.add_argument(
        "--experiments", nargs="+", default=list(EXP_MAP.keys()),
        choices=list(EXP_MAP.keys()),
        help="Which experiments to run (default: all)"
    )
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("NEURO-CXG: DATA QUALITY EXPERIMENTS")
    logger.info("=" * 70)

    all_results = {}

    if "1" in args.experiments:
        all_results["cross_site"] = experiment_cross_site_auc()
        all_results["site_robustness_gate"] = _apply_site_robustness_gate(all_results["cross_site"])

    if "2" in args.experiments:
        all_results["subject_audit"] = experiment_subject_count_audit()

    if "3" in args.experiments:
        all_results["atlas_baseline"] = experiment_atlas_centroid_baseline()

    logger.info("\n" + "=" * 70)
    logger.info("DATA QUALITY EXPERIMENTS COMPLETE")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
