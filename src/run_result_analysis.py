#!/usr/bin/env python
"""
scripts/run_result_analysis.py
Phase 9.3  Result Interpretation & Analysis
============================================
Covers all ROADMAP Phase 9.3 deliverables:

    ✅ Per-subject predictions + confidence scores → CSV
    ✅ Misclassification analysis (confusion patterns, feature profiles)
    ✅ Site-effect investigation (per-site AUC + bias heatmap)
    ✅ Case studies — top-K correctly and incorrectly classified subjects
    ✅ Prediction confidence distribution (calibration)
    ✅ Relationship between prediction confidence and clinical severity

Usage
-----
    # Full analysis (all sections)
    python scripts/run_result_analysis.py

    # Quick run — fewer case studies, skip slow plots
    python scripts/run_result_analysis.py --n-cases 3 --no-heatmap

    # Custom checkpoint / output directory
    python scripts/run_result_analysis.py --output-dir results/analysis_v2

Outputs
-------
results/analysis/
    per_subject_predictions.csv      — predictions + confidence per subject
    misclassification_analysis.png   — feature profile of FP vs FN subjects
    site_effects.png                 — per-site AUC bar chart
    site_bias_heatmap.png            — diagnosis distribution by site
    calibration.png                  — confidence distribution & calibration
    case_studies.txt                 — human-readable summaries of notable cases
    case_studies.csv                 — structured case study table
    severity_correlation.png         — confidence vs FIQ / age scatter
    result_analysis_summary.json     — machine-readable complete summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.core.config import (
    ALL_FEATURE_NAMES,
    CHECKPOINT_DIR,
    get_active_checkpoint_dir,
    GNN_BATCH_SIZE,
    GNN_DROPOUT,
    GNN_EDGE_GATE,
    GNN_GRL_ALPHA,
    GNN_HIDDEN_CHANNELS,
    GNN_IN_CHANNELS,
    GNN_NUM_LAYERS,
    GNN_NUM_HEADS,
    GNN_POOLING,
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_USE_SITE_EMBEDDING,
    K_FOLDS,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NUM_LOBES,
    RESULTS_DIR,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.causal_gnn import CausalBrainGNN
from src.models.factory import build_model
from src.models.training_utils import make_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR  = RESULTS_DIR / "analysis"
LOBE_LABELS = {v: k for k, v in LOBE_NAMES.items()} if isinstance(LOBE_NAMES, dict) else {}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_model(fold_id: int) -> CausalBrainGNN:
    model = build_model(
        device=DEVICE,
        use_grl=GNN_USE_GRL,
        grl_alpha=GNN_GRL_ALPHA,
    )
    ckpt_path = get_active_checkpoint_dir() / f"best_model_fold{fold_id}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def _collect_per_subject(
    graphs: List, fold_aucs: List[float]
) -> pd.DataFrame:
    """
    Build a per-subject DataFrame with columns:
    subject_id, true_label, pred_label, prob_asd, confidence,
    site_id, age_years, sex_code, fiq, correct
    """
    records, fold_probs_list = [], []

    for fold_id in range(K_FOLDS):
        try:
            model = _load_model(fold_id)
        except FileNotFoundError:
            continue
        loader = make_loader(graphs, batch_size=1, shuffle=False)
        probs  = []
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(DEVICE)
            out = model.forward_batch(batch) if hasattr(model, "forward_batch") else model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                site_id=getattr(batch, "site_id", None),
                age=batch.age if hasattr(batch, "age") else None,
                sex=batch.sex if hasattr(batch, "sex") else None,
                fiq=batch.fiq if hasattr(batch, "fiq") else None,
            )
            p = torch.softmax(out, dim=1)[:, 1].cpu().item()
            probs.append(p)
        fold_probs_list.append(np.array(probs))
        del model

    if not fold_probs_list:
        raise RuntimeError("No fold checkpoints found — run training first.")

    weights = np.array(fold_aucs[:len(fold_probs_list)])
    weights = weights / weights.sum()
    ens_probs = np.average(np.stack(fold_probs_list, axis=0), axis=0, weights=weights)

    for idx, g in enumerate(graphs):
        if g is None:
            continue
        prob    = float(ens_probs[idx])
        label   = int(g.y.item())
        pred    = int(prob >= 0.5)
        conf    = max(prob, 1 - prob)
        age_raw = float(g.age.item()) if hasattr(g, "age") and g.age is not None else 0.0
        sex_raw = float(g.sex.item()) if hasattr(g, "sex") and g.sex is not None else 0.0
        site_id = int(g.site_id.item()) if hasattr(g, "site_id") and g.site_id is not None else -1
        fiq_raw = float(g.fiq.item()) if hasattr(g, "fiq") and g.fiq is not None else 0.0
        sub_id  = str(g.sub_id) if hasattr(g, "sub_id") else f"subj_{idx}"
        records.append({
            "subject_id":   sub_id,
            "true_label":   label,
            "true_class":   "ASD" if label == 1 else "Control",
            "pred_label":   pred,
            "pred_class":   "ASD" if pred == 1 else "Control",
            "prob_asd":     round(prob, 4),
            "confidence":   round(conf, 4),
            "correct":      int(pred == label),
            "error_type":   _error_type(label, pred),
            "site_id":      site_id,
            "age_years":    round(age_raw * 20.0 + 15.0, 1),
            "sex_code":     round(sex_raw + 1.5),        # 1=M, 2=F
            "fiq":          round(fiq_raw * 30.0 + 100.0, 1),
        })

    return pd.DataFrame(records)


def _error_type(label: int, pred: int) -> str:
    if label == pred:
        return "TP" if label == 1 else "TN"
    return "FN" if label == 1 else "FP"


def _ensemble_fold_aucs() -> List[float]:
    """Load fold AUCs from the JSON summary if available, else return uniform weights."""
    json_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        aucs = data.get("per_fold_metrics", [])
        if aucs:
            return [m.get("auc", 0.5) for m in aucs]
    return [1.0 / K_FOLDS] * K_FOLDS


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PER-SUBJECT PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_per_subject_analysis(
    test_graphs: List, output_dir: Path
) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("SECTION 1 — PER-SUBJECT PREDICTIONS")
    logger.info("=" * 60)

    fold_aucs = _ensemble_fold_aucs()
    df = _collect_per_subject(test_graphs, fold_aucs)

    acc  = df["correct"].mean()
    auc  = roc_auc_score(df["true_label"], df["prob_asd"])
    n_fp = (df["error_type"] == "FP").sum()
    n_fn = (df["error_type"] == "FN").sum()

    logger.info("  Subjects: %d  Accuracy: %.3f  AUC: %.4f", len(df), acc, auc)
    logger.info("  False Positives: %d  False Negatives: %d", n_fp, n_fn)

    csv_path = output_dir / "per_subject_predictions.csv"
    df.to_csv(csv_path, index=False)
    logger.info("  Saved → %s", csv_path)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MISCLASSIFICATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_misclassification_analysis(
    df: pd.DataFrame, test_graphs: List, output_dir: Path
) -> Dict:
    logger.info("=" * 60)
    logger.info("SECTION 2 — MISCLASSIFICATION ANALYSIS")
    logger.info("=" * 60)

    fp_subjects = set(df[df["error_type"] == "FP"]["subject_id"].tolist())
    fn_subjects = set(df[df["error_type"] == "FN"]["subject_id"].tolist())
    tn_subjects = set(df[df["error_type"] == "TN"]["subject_id"].tolist())
    tp_subjects = set(df[df["error_type"] == "TP"]["subject_id"].tolist())

    logger.info("  TP=%d  TN=%d  FP=%d  FN=%d", len(tp_subjects), len(tn_subjects), len(fp_subjects), len(fn_subjects))

    # ── Feature profiles: mean node features per error group ─────────────────
    groups  = {"TP": tp_subjects, "TN": tn_subjects, "FP": fp_subjects, "FN": fn_subjects}
    feat_ix = {k: [] for k in groups}

    for g in test_graphs:
        if g is None:
            continue
        sub_id = str(g.sub_id) if hasattr(g, "sub_id") else None
        if sub_id is None:
            continue
        for gname, sid_set in groups.items():
            if sub_id in sid_set:
                feat_ix[gname].append(g.x.cpu().numpy())

    profiles: Dict[str, np.ndarray] = {}
    for gname, feat_list in feat_ix.items():
        if feat_list:
            # Mean across subjects, then mean across nodes → (num_features,)
            profiles[gname] = np.stack([f.mean(axis=0) for f in feat_list]).mean(axis=0)

    # ── Plot: FP vs FN feature profiles ──────────────────────────────────────
    if "FP" in profiles and "FN" in profiles and len(ALL_FEATURE_NAMES) > 0:
        _plot_error_feature_profiles(profiles, ALL_FEATURE_NAMES, output_dir / "misclassification_analysis.png")

    # ── Demographic profile of errors ─────────────────────────────────────────
    demo_summary = {}
    for gname in ("FP", "FN", "TP", "TN"):
        sub_df = df[df["error_type"] == gname]
        if sub_df.empty:
            continue
        demo_summary[gname] = {
            "n_subjects":      len(sub_df),
            "mean_age":        round(float(sub_df["age_years"].mean()), 1),
            "pct_male":        round(float((sub_df["sex_code"] == 1).mean() * 100), 1),
            "mean_fiq":        round(float(sub_df["fiq"].mean()), 1),
            "mean_confidence": round(float(sub_df["confidence"].mean()), 3),
        }
        logger.info(
            "  %s (n=%d): age=%.1f  pct_male=%.0f%%  FIQ=%.1f  conf=%.3f",
            gname, demo_summary[gname]["n_subjects"],
            demo_summary[gname]["mean_age"], demo_summary[gname]["pct_male"],
            demo_summary[gname]["mean_fiq"], demo_summary[gname]["mean_confidence"],
        )

    return {"feature_profiles": {k: v.tolist() for k, v in profiles.items()}, "demographics": demo_summary}


def _plot_error_feature_profiles(
    profiles: Dict[str, np.ndarray], feature_names: List[str], save_path: Path
) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        feature_names = list(feature_names)[:28]

        for ax, (a_name, b_name) in zip(axes, [("FP", "TN"), ("FN", "TP")]):
            if a_name not in profiles or b_name not in profiles:
                ax.set_title(f"{a_name} vs {b_name} (no data)")
                continue
            diff = profiles[a_name] - profiles[b_name]
            colors = ["#e74c3c" if d > 0 else "#3498db" for d in diff]
            y = np.arange(len(diff))
            ax.barh(y, diff, color=colors, alpha=0.8, edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(feature_names[:len(diff)], fontsize=7)
            ax.set_xlabel(f"Mean feature diff ({a_name} − {b_name})", fontsize=10)
            ax.set_title(f"{a_name} − {b_name}: Feature Profile Difference", fontsize=11, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)

        plt.suptitle("Misclassification Feature Profiles", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Misclassification plot saved → %s", save_path)
    except Exception as e:
        logger.warning("  Misclassification plot failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SITE EFFECTS INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════════

def run_site_effects(
    df: pd.DataFrame, output_dir: Path, no_heatmap: bool = False
) -> Dict:
    logger.info("=" * 60)
    logger.info("SECTION 3 — SITE EFFECTS INVESTIGATION")
    logger.info("=" * 60)

    def _safe_auc(sub_df):
        if len(sub_df) < 5 or sub_df["true_label"].nunique() < 2:
            return float("nan")
        return float(roc_auc_score(sub_df["true_label"], sub_df["prob_asd"]))

    sites = df[df["site_id"] >= 0]["site_id"].unique()
    site_stats = []

    for site in sorted(sites):
        sdf      = df[df["site_id"] == site]
        auc      = _safe_auc(sdf)
        n_asd    = int((sdf["true_label"] == 1).sum())
        n_ctrl   = int((sdf["true_label"] == 0).sum())
        acc      = float(sdf["correct"].mean())
        site_stats.append({
            "site_id": int(site), "n_total": len(sdf),
            "n_asd": n_asd, "n_control": n_ctrl,
            "auc": round(auc, 4), "accuracy": round(acc, 3),
        })
        logger.info("  Site %-3d  n=%-4d  ASD=%-3d  Ctrl=%-3d  AUC=%.4f  Acc=%.3f",
                    site, len(sdf), n_asd, n_ctrl, auc, acc)

    # ── Per-site AUC bar chart ─────────────────────────────────────────────────
    _plot_site_auc(site_stats, output_dir / "site_effects.png")

    # ── ASD prevalence heatmap ─────────────────────────────────────────────────
    if not no_heatmap:
        _plot_site_bias(site_stats, output_dir / "site_bias_heatmap.png")

    return {"per_site": site_stats}


def _plot_site_auc(site_stats: List[Dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        valid = [s for s in site_stats if not np.isnan(s["auc"])]
        valid.sort(key=lambda x: -x["auc"])
        x_labels = [f"Site {s['site_id']}" for s in valid]
        aucs     = [s["auc"] for s in valid]
        ns       = [s["n_total"] for s in valid]

        fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.7), 5))
        colors  = ["#e74c3c" if a >= 0.6 else "#f39c12" if a >= 0.5 else "#bdc3c7" for a in aucs]
        bars    = ax.bar(x_labels, aucs, color=colors, alpha=0.85, edgecolor="white")
        for bar, auc, n in zip(bars, aucs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{auc:.2f}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, color="gray", lw=1.2, ls="--", label="Chance (0.50)")
        ax.set_ylim(0.3, 0.95)
        ax.set_ylabel("AUC", fontsize=12, fontweight="bold")
        ax.set_title("Per-Site AUC on Test Set", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Site AUC plot saved → %s", save_path)
    except Exception as e:
        logger.warning("  Site AUC plot failed: %s", e)


def _plot_site_bias(site_stats: List[Dict], save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        sorted_sites = sorted(site_stats, key=lambda x: x["site_id"])
        labels = [f"Site {s['site_id']}" for s in sorted_sites]
        asd_pct = [100 * s["n_asd"] / s["n_total"] if s["n_total"] > 0 else 50 for s in sorted_sites]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
        norm    = mcolors.TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
        colors  = [plt.cm.RdYlGn(1 - norm(p)) for p in asd_pct]
        bars    = ax.bar(labels, asd_pct, color=colors, alpha=0.9, edgecolor="white")
        for bar, pct in zip(bars, asd_pct):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.axhline(50, color="gray", lw=1.2, ls="--", label="50% balanced")
        ax.set_ylim(0, 110)
        ax.set_ylabel("% ASD subjects", fontsize=12, fontweight="bold")
        ax.set_title("ASD Prevalence per Site (Potential Site Bias)", fontsize=13, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Site bias heatmap saved → %s", save_path)
    except Exception as e:
        logger.warning("  Site bias heatmap failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PREDICTION CONFIDENCE & CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def run_calibration_analysis(df: pd.DataFrame, output_dir: Path) -> Dict:
    logger.info("=" * 60)
    logger.info("SECTION 4 — PREDICTION CONFIDENCE & CALIBRATION")
    logger.info("=" * 60)

    # Fraction correct by confidence bin
    bins  = np.linspace(0.5, 1.0, 11)
    bin_labels, frac_correct = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        if mask.sum() > 0:
            bin_labels.append(f"{lo:.2f}–{hi:.2f}")
            frac_correct.append(float(df[mask]["correct"].mean()))

    mean_conf = float(df["confidence"].mean())
    pct_high  = float((df["confidence"] >= 0.75).mean() * 100)
    logger.info("  Mean confidence: %.3f  Pct high-conf (≥0.75): %.1f%%", mean_conf, pct_high)

    _plot_calibration(df, bin_labels, frac_correct, output_dir / "calibration.png")

    return {
        "mean_confidence": mean_conf,
        "pct_high_confidence": pct_high,
        "calibration_bins":    bin_labels,
        "fraction_correct":    frac_correct,
    }


def _plot_calibration(
    df: pd.DataFrame,
    bin_labels: List[str],
    frac_correct: List[float],
    save_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(14, 5))
        gs  = GridSpec(1, 3, figure=fig)

        # ── Left: confidence distribution by true class ────────────────────
        ax1 = fig.add_subplot(gs[0])
        asd_conf  = df[df["true_label"] == 1]["prob_asd"]
        ctrl_conf = df[df["true_label"] == 0]["prob_asd"]
        ax1.hist(asd_conf,  bins=20, alpha=0.7, color="#e74c3c",  label="ASD (true)",     density=True)
        ax1.hist(ctrl_conf, bins=20, alpha=0.7, color="#3498db",  label="Control (true)", density=True)
        ax1.axvline(0.5, color="black", lw=1, ls="--")
        ax1.set_xlabel("P(ASD)", fontsize=11)
        ax1.set_ylabel("Density", fontsize=11)
        ax1.set_title("Confidence Distribution\nby True Class", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)

        # ── Middle: confidence by prediction correctness ───────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.hist(df[df["correct"] == 1]["confidence"], bins=20, alpha=0.8, color="#2ecc71",  label="Correct", density=True)
        ax2.hist(df[df["correct"] == 0]["confidence"], bins=20, alpha=0.8, color="#e74c3c",  label="Wrong",   density=True)
        ax2.set_xlabel("Confidence", fontsize=11)
        ax2.set_title("Confidence vs\nCorrectness", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)

        # ── Right: reliability diagram ─────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        if bin_labels:
            x   = np.arange(len(bin_labels))
            ax3.bar(x, frac_correct, color="#9b59b6", alpha=0.8, edgecolor="white")
            ax3.axhline(1.0, color="gray", lw=0.8, ls="--")
            ax3.set_xticks(x)
            ax3.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=7)
            ax3.set_ylabel("Fraction Correct", fontsize=11)
            ax3.set_ylim(0, 1.1)
            ax3.set_title("Reliability Diagram\n(Confidence → Accuracy)", fontsize=11, fontweight="bold")
            ax3.grid(axis="y", alpha=0.3)

        plt.suptitle("Prediction Confidence Analysis", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Calibration plot saved → %s", save_path)
    except Exception as e:
        logger.warning("  Calibration plot failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SEVERITY CORRELATION (CONFIDENCE vs FIQ / AGE)
# ══════════════════════════════════════════════════════════════════════════════

def run_severity_correlation(df: pd.DataFrame, output_dir: Path) -> Dict:
    logger.info("=" * 60)
    logger.info("SECTION 5 — SEVERITY CORRELATION (CONFIDENCE vs FIQ / AGE)")
    logger.info("=" * 60)

    results = {}
    try:
        from scipy import stats

        asd_df  = df[df["true_label"] == 1].dropna(subset=["fiq", "age_years"])
        all_df  = df.dropna(subset=["fiq", "age_years"])

        for col, label in [("fiq", "FIQ"), ("age_years", "Age")]:
            for group, gdf in [("ASD", asd_df), ("All", all_df)]:
                r, p = stats.pearsonr(gdf[col], gdf["confidence"])
                results[f"{label}_{group}"] = {"r": round(r, 3), "p": round(p, 4)}
                logger.info(
                    "  %s vs confidence (%s): r=%.3f  p=%.4f %s",
                    label, group, r, p, "✓" if p < 0.05 else ""
                )

    except Exception as e:
        logger.warning("  Scipy not available or correlation failed: %s", e)

    # ── Scatter: FIQ / Age vs confidence ──────────────────────────────────────
    _plot_severity_scatter(df, output_dir / "severity_correlation.png")
    return results


def _plot_severity_scatter(df: pd.DataFrame, save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, col, xlabel in zip(axes, ["age_years", "fiq"], ["Age (years)", "FIQ"]):
            for label, color, marker in [(1, "#e74c3c", "^"), (0, "#3498db", "o")]:
                mask = df["true_label"] == label
                ax.scatter(
                    df[mask][col], df[mask]["confidence"],
                    c=color, marker=marker, alpha=0.55, s=30,
                    label="ASD" if label == 1 else "Control",
                )
            ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
            ax.set_ylabel("Prediction Confidence", fontsize=12, fontweight="bold")
            ax.set_title(f"Confidence vs {xlabel}", fontsize=12, fontweight="bold")
            ax.set_ylim(0.45, 1.05)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.suptitle("Prediction Confidence vs Clinical Variables", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Severity correlation plot saved → %s", save_path)
    except Exception as e:
        logger.warning("  Severity scatter failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CASE STUDIES
# ══════════════════════════════════════════════════════════════════════════════

def run_case_studies(df: pd.DataFrame, test_graphs: List, n_cases: int, output_dir: Path) -> List[Dict]:
    """
    Identify and describe notable subjects:
      - Top-N most confident correct predictions (TP + TN)
      - Top-N most confident wrong predictions   (FP + FN)
      - Top-N most ambiguous (confidence closest to 0.5)
    """
    logger.info("=" * 60)
    logger.info("SECTION 6 — CASE STUDIES (n=%d per group)", n_cases)
    logger.info("=" * 60)

    graph_map = {}
    for g in test_graphs:
        if g is None:
            continue
        sub_id = str(g.sub_id) if hasattr(g, "sub_id") else None
        if sub_id:
            graph_map[sub_id] = g

    cases: List[Dict] = []

    groups = {
        "HIGH_CONF_CORRECT": df[df["correct"] == 1].nlargest(n_cases, "confidence"),
        "HIGH_CONF_WRONG":   df[df["correct"] == 0].nlargest(n_cases, "confidence"),
        "AMBIGUOUS":         df.iloc[(df["confidence"] - 0.5).abs().argsort()].head(n_cases),
    }

    for group_name, sub_df in groups.items():
        for _, row in sub_df.iterrows():
            g   = graph_map.get(row["subject_id"])
            case = {
                "group":       group_name,
                "subject_id":  row["subject_id"],
                "true_class":  row["true_class"],
                "pred_class":  row["pred_class"],
                "prob_asd":    row["prob_asd"],
                "confidence":  row["confidence"],
                "age_years":   row["age_years"],
                "sex":         "M" if row["sex_code"] == 1 else "F",
                "fiq":         row["fiq"],
                "site_id":     row["site_id"],
                "error_type":  row["error_type"],
            }
            if g is not None:
                x_np = g.x.cpu().numpy()
                mean_node_feats = x_np.mean(axis=0)
                top3_feat_idx   = np.argsort(np.abs(mean_node_feats))[-3:][::-1]
                feat_names      = list(ALL_FEATURE_NAMES)
                case["top3_features"] = [
                    {"feature": feat_names[i] if i < len(feat_names) else f"feat_{i}",
                     "mean_value": round(float(mean_node_feats[i]), 4)}
                    for i in top3_feat_idx
                ]
            cases.append(case)
            logger.info(
                "  [%s] %s  true=%s  pred=%s  conf=%.3f  age=%.0f  sex=%s  FIQ=%.0f",
                group_name, row["subject_id"], row["true_class"], row["pred_class"],
                row["confidence"], row["age_years"], case["sex"], row["fiq"],
            )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_cases = pd.DataFrame([
        {k: v for k, v in c.items() if k != "top3_features"} for c in cases
    ])
    df_cases.to_csv(output_dir / "case_studies.csv", index=False)

    # ── Save human-readable text report ───────────────────────────────────────
    txt_path = output_dir / "case_studies.txt"
    with open(txt_path, "w") as f:
        f.write("CASE STUDIES — NEURO-CXG ASD CLASSIFICATION\n")
        f.write("=" * 60 + "\n\n")
        for c in cases:
            f.write(f"[{c['group']}]\n")
            f.write(f"  Subject  : {c['subject_id']}\n")
            f.write(f"  Diagnosis: {c['true_class']}  →  Predicted: {c['pred_class']}\n")
            f.write(f"  P(ASD)   : {c['prob_asd']:.4f}   Confidence: {c['confidence']:.4f}\n")
            f.write(f"  Age      : {c['age_years']:.0f}  Sex: {c['sex']}  FIQ: {c['fiq']:.0f}  Site: {c['site_id']}\n")
            if "top3_features" in c:
                f.write("  Top-3 features (by |mean| across 12 regions):\n")
                for ft in c["top3_features"]:
                    f.write(f"    {ft['feature']}: {ft['mean_value']:.4f}\n")
            f.write("\n")
    logger.info("  Case studies saved → %s and %s", output_dir / "case_studies.csv", txt_path)

    return cases


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SAVE SUMMARY JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_summary(
    df: pd.DataFrame,
    misclass_result: Dict,
    site_result: Dict,
    calib_result: Dict,
    severity_result: Dict,
    case_studies: List[Dict],
    output_dir: Path,
) -> None:
    summary = {
        "n_subjects":            len(df),
        "overall_accuracy":      float(df["correct"].mean()),
        "overall_auc":           float(roc_auc_score(df["true_label"], df["prob_asd"])),
        "misclassification":     misclass_result.get("demographics", {}),
        "site_effects":          site_result,
        "calibration":           calib_result,
        "severity_correlation":  severity_result,
        "n_case_studies":        len(case_studies),
    }
    json_path = output_dir / "result_analysis_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary JSON saved → %s", json_path)

    logger.info("\n" + "═" * 60)
    logger.info("RESULT ANALYSIS COMPLETE")
    logger.info("═" * 60)
    logger.info("  Subjects: %d  |  Accuracy: %.3f  |  AUC: %.4f",
                summary["n_subjects"], summary["overall_accuracy"], summary["overall_auc"])
    logger.info("  Output → %s", output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9.3 Result Interpretation & Analysis for Neuro-CXG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-cases",    type=int,  default=5,
                        help="Case studies per group (high-conf correct, wrong, ambiguous).")
    parser.add_argument("--no-heatmap", action="store_true", default=False,
                        help="Skip site-bias heatmap.")
    parser.add_argument("--no-severity", action="store_true", default=False,
                        help="Skip severity correlation plots.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device     : %s", DEVICE)
    logger.info("Output dir : %s", args.output_dir)

    # ── Load test dataset ─────────────────────────────────────────────────────
    logger.info("Loading test dataset…")
    test_dataset = ABIDECausalDataset(split="test")
    test_graphs  = [g for g in test_dataset if g is not None]
    logger.info("  Test subjects: %d", len(test_graphs))

    if not test_graphs:
        logger.error("Test set is empty — run the full pipeline first.")
        import sys
        sys.exit(1)

    # ── Section 1: Per-subject predictions ────────────────────────────────────
    df = run_per_subject_analysis(test_graphs, args.output_dir)

    # ── Section 2: Misclassification analysis ─────────────────────────────────
    misclass_result = run_misclassification_analysis(df, test_graphs, args.output_dir)

    # ── Section 3: Site effects ───────────────────────────────────────────────
    site_result = run_site_effects(df, args.output_dir, no_heatmap=args.no_heatmap)

    # ── Section 4: Calibration ────────────────────────────────────────────────
    calib_result = run_calibration_analysis(df, args.output_dir)

    # ── Section 5: Severity correlation ───────────────────────────────────────
    severity_result = {}
    if not args.no_severity:
        severity_result = run_severity_correlation(df, args.output_dir)

    # ── Section 6: Case studies ───────────────────────────────────────────────
    case_studies = run_case_studies(df, test_graphs, args.n_cases, args.output_dir)

    # ── Section 7: Summary JSON ───────────────────────────────────────────────
    save_summary(df, misclass_result, site_result, calib_result, severity_result, case_studies, args.output_dir)


if __name__ == "__main__":
    main()
