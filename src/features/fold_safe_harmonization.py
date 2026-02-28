"""
Fold-safe neuroHarmonize harmonization for Neuro-CXG.

Pipeline
--------
1. Validate + repair raw temporal features (NaN/Inf/outlier handling).
2. Aggregate 170 AAL ROIs -> 12 brain regions (vectorised mean).
3. 5-fold CV harmonization via neuroHarmonize (ComBat).
4. Quality-check variance retention across folds.
5. Write combined leave-one-fold-out output for graph_factory.py.

External interface (unchanged):
    harmonize_cv_safe_fold(features_df, manifest_df, ...) -> List[HarmonizationFold]
    main()  # called by run_pipeline.py
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    LOBE_MAPPING,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NUM_LOBES,
    NUM_TEMPORAL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── constants ─────────────────────────────────────────────────────────────────
FEATURE_TYPES = [
    "mean", "std", "skew", "kurt", "psd", "mssd", "range", "autocorr",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "delta_peak", "theta_peak", "alpha_peak", "beta_peak", "gamma_peak",
    "spectral_entropy", "phase_std",
]

NAN_REMOVAL_THRESHOLD = 0.5   # drop subjects with >50 % NaN
OUTLIER_STD_THRESHOLD = 5     # cap outliers beyond ±5 σ
VARIANCE_WARNING_THRESHOLD = 30.0  # warn when >30 % features lose/gain variance
VARIANCE_RETENTION_LOW = 0.7     # flag features retaining <70 % of original variance
VARIANCE_RETENTION_HIGH = 1.3    # flag features gaining  >30 % of original variance


@dataclass
class HarmonizationFold:
    """Container for harmonization results of a single CV fold."""
    fold: int
    train: pd.DataFrame
    val: pd.DataFrame
    train_idx: np.ndarray
    val_idx: np.ndarray
    model: Optional[object]


# ── feature-column helpers ────────────────────────────────────────────────────

def _feat_cols(df: pd.DataFrame) -> List[str]:
    """Return all columns except subject_id."""
    return [c for c in df.columns if c != "subject_id"]


# ── 1. Validation ───────────────────────────────────────────────────────────────────

def validate_features(df: pd.DataFrame) -> None:
    """Log a compact validation summary for the feature matrix."""
    cols = _feat_cols(df)
    vals = df[cols].replace([np.inf, -np.inf], np.nan)

    nan_total = int(vals.isna().sum().sum())
    inf_total = int(np.isinf(df[cols].values).sum())
    subj_nan = int(vals.isna().any(axis=1).sum())
    feat_nan = int(vals.isna().any(axis=0).sum())
    nan_pct = 100.0 * nan_total / max(vals.size, 1)

    logger.info(
        "Feature validation: %d subjects × %d features | "
        "NaN: %d (%.1f%%) | Inf: %d | Subjects with NaN: %d | Features with NaN: %d",
        len(df), len(cols), nan_total, nan_pct, inf_total, subj_nan, feat_nan,
    )


# ── 2. Repair ─────────────────────────────────────────────────────────────────────

def repair_features(df: pd.DataFrame, *, impute_nans: bool = True) -> pd.DataFrame:
    """
    Clean feature matrix in four vectorised steps.

    1. Drop subjects with >NAN_REMOVAL_THRESHOLD fraction of NaN values.
    2. Replace ±Inf with the column 1st/99th percentile.
    3. Log-transform spectral power features (reduces heavy tails).
    4. Impute remaining NaNs with per-column medians; cap ±OUTLIER_STD_THRESHOLD σ.

    Returns a cleaned copy; original is not modified.
    """
    df = df.copy()
    cols = _feat_cols(df)
    n_before = len(df)

    # Step 1 — drop subjects with too many NaNs
    min_valid = int(len(cols) * (1.0 - NAN_REMOVAL_THRESHOLD))
    df = df.dropna(subset=cols, thresh=min_valid).reset_index(drop=True)
    removed = n_before - len(df)
    if removed:
        logger.warning(
            "Dropped %d subjects with >%d%% NaN values", removed, int(NAN_REMOVAL_THRESHOLD * 100)
        )

    # Step 2 — replace ±Inf with quantile bounds (column-wise)
    inf_replaced = 0
    for col in cols:
        mask_pos = df[col] == np.inf
        mask_neg = df[col] == -np.inf
        if mask_pos.any() or mask_neg.any():
            valid = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            df.loc[mask_pos, col] = valid.quantile(0.99) if len(valid) else 0.0
            df.loc[mask_neg, col] = valid.quantile(0.01) if len(valid) else 0.0
            inf_replaced += int(mask_pos.sum()) + int(mask_neg.sum())
    if inf_replaced:
        logger.warning("Replaced %d ±Inf values with 1st/99th-percentile bounds", inf_replaced)

    # Step 3 — log-transform spectral power (positive-valued, heavy-tailed)
    spectral = ["delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"]
    spectral_cols = [c for c in cols if any(s in c for s in spectral)]
    for col in spectral_cols:
        mask = df[col] > 0
        df.loc[mask, col] = np.log1p(df.loc[mask, col])
    if spectral_cols:
        logger.info("Log-transformed %d spectral-power columns", len(spectral_cols))

    # Step 4a — impute NaNs with per-column median
    if impute_nans:
        medians = df[cols].median()
        df[cols] = df[cols].fillna(medians)

    # Step 4b — clip outliers beyond ±OUTLIER_STD_THRESHOLD σ (vectorised)
    means = df[cols].mean()
    stds = df[cols].std().replace(0, np.nan)   # avoid clipping constant cols
    lower = means - OUTLIER_STD_THRESHOLD * stds
    upper = means + OUTLIER_STD_THRESHOLD * stds
    df[cols] = df[cols].clip(lower=lower, upper=upper, axis=1)

    logger.info("Repair complete: %d subjects remaining (removed %d)", len(df), removed)
    return df


# ── 3. ROI → Lobe aggregation ─────────────────────────────────────────────

def aggregate_to_lobes(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 170-ROI → 12-lobe feature matrix (vectorised).

    Input:  subject_id + (170 ROIs × 20 features) ≈ 3401 columns
    Output: subject_id + (NUM_LOBES × len(FEATURE_TYPES)) columns
    """
    logger.info("Aggregating 170 ROIs \u2192 %d regions\u2026", NUM_LOBES)
    result: Dict[str, object] = {"subject_id": df["subject_id"]}
    for lobe_id in range(NUM_LOBES):
        lobe_name = LOBE_NAMES[lobe_id]
        roi_ids = [i + 1 for i in LOBE_MAPPING[lobe_id]]
        for feat in FEATURE_TYPES:
            present = [f"roi{r}_{feat}" for r in roi_ids if f"roi{r}_{feat}" in df.columns]
            result[f"{lobe_name}_{feat}"] = df[present].mean(axis=1) if present else 0.0
    out = pd.DataFrame(result)
    expected_cols = 1 + NUM_LOBES * len(FEATURE_TYPES)
    if len(out.columns) != expected_cols:
        logger.warning(
            "aggregate_to_lobes: expected %d columns, got %d", expected_cols, len(out.columns)
        )
    logger.info(
        "Aggregated \u2192 %d subjects \u00d7 %d lobe-features", len(out), len(out.columns) - 1
    )
    return out


# ── 4. Harmonization helpers ──────────────────────────────────────────────

def _prepare_covariates(manifest: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Build covariate DataFrame for neuroHarmonize (requires exact 'SITE' column)."""
    cov = manifest[["subject_id", "SITE_ID", "AGE_AT_SCAN", "SEX"]].copy()
    cov = cov.merge(features_df[["subject_id"]], on="subject_id", how="inner")
    cov = cov.rename(columns={"SITE_ID": "SITE"})
    if cov["AGE_AT_SCAN"].isna().any():
        cov["AGE_AT_SCAN"] = cov["AGE_AT_SCAN"].fillna(cov["AGE_AT_SCAN"].median())
    if cov["SEX"].isna().any():
        cov["SEX"] = cov["SEX"].fillna(cov["SEX"].mode().iloc[0])
    cov["SITE"] = cov["SITE"].astype(str)
    cov["SEX"] = pd.to_numeric(cov["SEX"], errors="coerce").fillna(1).astype(int)
    return cov.drop(columns=["subject_id"])


def _remove_constant_features(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Drop zero-variance columns that would cause ComBat to fail."""
    variances = features.var()
    constant_cols = variances[variances == 0].index.tolist()
    kept_cols = [c for c in features.columns if c not in constant_cols]
    if constant_cols:
        logger.warning("Dropping %d constant features before harmonization", len(constant_cols))
    return features[kept_cols], kept_cols, constant_cols


def _restore_constant_features(
    harmonized: pd.DataFrame,
    original: pd.DataFrame,
    kept_cols: List[str],
    dropped_cols: List[str],
) -> pd.DataFrame:
    """Re-attach zero-variance columns that were removed pre-harmonization."""
    result = pd.DataFrame(harmonized.values, columns=kept_cols)
    for col in dropped_cols:
        result[col] = original[col].values if col in original.columns else 0.0
    all_cols = [c for c in original.columns if c != "subject_id"]
    return result[[c for c in all_cols if c in result.columns]]


def _harmonize_fold(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    train_covariates: pd.DataFrame,
    val_covariates: pd.DataFrame,
) -> Tuple[Optional[object], pd.DataFrame, pd.DataFrame]:
    """Harmonize one CV fold via neuroHarmonize; falls back to raw data on error."""
    try:
        # Tiny epsilon noise prevents singular-matrix errors for near-constant columns
        rng = np.random.default_rng(seed=0)
        tf, vf = train_features.copy(), val_features.copy()
        for col in train_features.columns:
            if train_features[col].std() < 1e-5:
                tf[col] += rng.normal(0, 1e-8, len(tf))
                vf[col] += rng.normal(0, 1e-8, len(vf))
        model, train_harm = harmonizationLearn(tf.values, train_covariates)
        val_harm = harmonizationApply(vf.values, val_covariates, model)
        return (
            model,
            pd.DataFrame(train_harm, columns=train_features.columns, index=train_features.index),
            pd.DataFrame(val_harm, columns=val_features.columns, index=val_features.index),
        )
    except Exception as exc:
        logger.error(
            "Harmonization failed (%s); using unharmonized features for this fold", exc
        )
        return None, train_features, val_features


# ── 5. Quality verification ───────────────────────────────────────────────

def _check_harmonization_quality(
    original_df: pd.DataFrame,
    harmonized_folds: List[HarmonizationFold],
) -> Dict[str, object]:
    """Check variance retention and NaN introduction across all harmonized folds."""
    logger.info("=" * 60)
    logger.info("HARMONIZATION QUALITY CHECK")
    logger.info("=" * 60)

    all_harmonized = pd.concat(
        [pd.concat([f.train, f.val], ignore_index=True) for f in harmonized_folds],
        ignore_index=True,
    )
    harm_cols = _feat_cols(all_harmonized)

    orig_cols = _feat_cols(original_df)
    common = [c for c in orig_cols if c in harm_cols]
    if not common:
        logger.info("No overlapping columns — aggregating originals to lobes for comparison")
        try:
            original_df = aggregate_to_lobes(original_df)
            orig_cols = _feat_cols(original_df)
            common = [c for c in orig_cols if c in harm_cols]
        except Exception as exc:
            logger.warning("Lobe aggregation failed for quality check: %s", exc)

    if not common:
        logger.warning("No overlapping feature columns — skipping variance retention check")
        return {"variance_retention": np.nan, "nans_introduced": 0, "quality": "check_warnings"}

    orig_var = original_df[common].var().mean()
    harm_var = all_harmonized[common].var().mean()
    var_retention = harm_var / orig_var if orig_var != 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_feat_ret = (
            all_harmonized[common].var() / original_df[common].var()
        ).replace([np.inf, -np.inf], np.nan).dropna()

    n_total = len(per_feat_ret)
    n_low = int((per_feat_ret < VARIANCE_RETENTION_LOW).sum())
    n_high = int((per_feat_ret > VARIANCE_RETENTION_HIGH).sum())
    n_within = n_total - n_low - n_high

    orig_nans = int(original_df[orig_cols].isna().sum().sum())
    harm_nans = int(all_harmonized[harm_cols].isna().sum().sum())
    nans_introduced = harm_nans - orig_nans

    logger.info("  Original variance  : %.4f", orig_var)
    logger.info("  Harmonized variance: %.4f", harm_var)
    logger.info("  Variance retention : %.2f%%", var_retention * 100)
    if n_total:
        logger.info(
            "  Per-feature: %.1f%% within, %.1f%% low, %.1f%% high",
            100 * n_within / n_total,
            100 * n_low / n_total,
            100 * n_high / n_total,
        )
        if 100 * n_low / n_total > VARIANCE_WARNING_THRESHOLD:
            logger.warning("  Many features lost >30%% variance after harmonization")
        if 100 * n_high / n_total > VARIANCE_WARNING_THRESHOLD:
            logger.warning("  Many features gained >30%% variance after harmonization")
    if nans_introduced > 0:
        logger.warning("  Harmonization introduced %d NaN values", nans_introduced)
    else:
        logger.info("  No NaN values introduced")

    is_ok = (
        n_total > 0
        and (100 * n_low / n_total) <= VARIANCE_WARNING_THRESHOLD
        and (100 * n_high / n_total) <= VARIANCE_WARNING_THRESHOLD
        and nans_introduced <= 0
    )
    return {
        "variance_retention": var_retention,
        "nans_introduced": nans_introduced,
        "per_feature_stats": {
            "retention_within": n_within,
            "retention_low": n_low,
            "retention_high": n_high,
            "retention_total": n_total,
            "retention_low_pct": 100 * n_low / max(n_total, 1),
            "retention_high_pct": 100 * n_high / max(n_total, 1),
            "retention_within_pct": 100 * n_within / max(n_total, 1),
        },
        "quality": "good" if is_ok else "check_warnings",
    }


def harmonize_cv_safe_fold(
    features_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    output_dir: Optional[Path] = None,
    full_output_path: Optional[Path] = None,
) -> List[HarmonizationFold]:
    """
    Perform fold-safe harmonization with CV-aware train/val splitting.
    
    Args:
        features_df: DataFrame with temporal features
        manifest_df: DataFrame with subject metadata
        cv_splits: Optional pre-computed CV splits
        output_dir: Optional directory for saving fold outputs
        full_output_path: Optional path for saving combined output
    
    Returns:
        List of HarmonizationFold objects containing harmonized data
    """
    logger.info("=" * 80)
    logger.info("FOLD-SAFE HARMONIZATION")
    logger.info("=" * 80)
    
    # Validate and repair features
    validate_features(features_df)
    features_clean = repair_features(features_df, impute_nans=True)
    
    # Prepare for CV
    if cv_splits is None:
        from sklearn.model_selection import StratifiedKFold
        
        labels = manifest_df.set_index("subject_id")["DX_GROUP"]
        labels = labels.reindex(features_clean["subject_id"])
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(skf.split(features_clean, labels))
    
    # Process each fold
    harmonized_folds = []
    
    for fold_id, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info("=" * 80)
        logger.info("PROCESSING FOLD %d / %d", fold_id + 1, len(cv_splits))
        logger.info("=" * 80)
        
        # Split data
        train_data = features_clean.iloc[train_idx].reset_index(drop=True)
        val_data = features_clean.iloc[val_idx].reset_index(drop=True)
        
        train_manifest = manifest_df[
            manifest_df["subject_id"].isin(train_data["subject_id"])
        ].reset_index(drop=True)
        val_manifest = manifest_df[
            manifest_df["subject_id"].isin(val_data["subject_id"])
        ].reset_index(drop=True)
        
        # Prepare covariates
        train_covariates = _prepare_covariates(train_manifest, train_data)
        val_covariates = _prepare_covariates(val_manifest, val_data)
        
        # Remove constant features
        train_features = train_data.drop(columns=["subject_id"])
        val_features = val_data.drop(columns=["subject_id"])
        
        train_features, kept_cols, dropped_cols = _remove_constant_features(train_features)
        val_features = val_features[kept_cols]
        
        # Harmonize
        model, train_harmonized, val_harmonized = _harmonize_fold(
            train_features,
            val_features,
            train_covariates,
            val_covariates,
        )
        
        # Restore constant features
        train_df = _restore_constant_features(
            train_harmonized,
            train_data,
            kept_cols,
            dropped_cols,
        )
        train_df = pd.concat(
            [train_data[["subject_id"]].reset_index(drop=True), train_df.reset_index(drop=True)],
            axis=1,
        )
        
        val_df = _restore_constant_features(
            val_harmonized,
            val_data,
            kept_cols,
            dropped_cols,
        )
        val_df = pd.concat(
            [val_data[["subject_id"]].reset_index(drop=True), val_df.reset_index(drop=True)],
            axis=1,
        )
        
        # Aggregate to lobes
        logger.info("  Aggregating ROIs to lobes...")
        train_lobes = aggregate_to_lobes(train_df)
        val_lobes = aggregate_to_lobes(val_df)
        
        # Add metadata
        train_meta = train_manifest[["subject_id", "SITE_ID", "DX_GROUP"]]
        val_meta = val_manifest[["subject_id", "SITE_ID", "DX_GROUP"]]
        
        train_lobes = train_lobes.merge(train_meta, on="subject_id", how="left")
        val_lobes = val_lobes.merge(val_meta, on="subject_id", how="left")
        
        # Create fold result
        fold_result = HarmonizationFold(
            fold=fold_id,
            train=train_lobes,
            val=val_lobes,
            train_idx=train_idx,
            val_idx=val_idx,
            model=model,
        )
        
        # Save if output directory specified
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            train_path = output_dir / f"fold{fold_id}_train_lobes.csv"
            val_path = output_dir / f"fold{fold_id}_val_lobes.csv"
            train_lobes.to_csv(train_path, index=False)
            val_lobes.to_csv(val_path, index=False)
            logger.info("  Saved: %s", train_path)
            logger.info("  Saved: %s", val_path)
        
        harmonized_folds.append(fold_result)
        logger.info("  Fold %d complete", fold_id + 1)
    
    logger.info("=" * 80)
    logger.info("ALL FOLDS HARMONIZED")
    logger.info("=" * 80)

    # Write combined lobe-level output for graph_factory.py
    # Each subject appears in exactly one val fold in k-fold CV, so combining
    # val sets gives the full dataset harmonized in a leave-one-fold-out manner.
    if full_output_path is not None and harmonized_folds:
        all_val_frames = [fold_result.val for fold_result in harmonized_folds]
        combined_df = pd.concat(all_val_frames, ignore_index=True)

        # Drop metadata columns — graph_factory expects only subject_id + feature cols
        meta_cols = ["SITE_ID", "DX_GROUP"]
        combined_df = combined_df.drop(
            columns=[c for c in meta_cols if c in combined_df.columns]
        )

        # Deduplicate (safety: each subject should appear in exactly one val fold)
        combined_df = combined_df.drop_duplicates(subset=["subject_id"])

        Path(full_output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(full_output_path, index=False)
        logger.info(
            "Saved combined harmonized lobe-level features → %s (%d subjects, %d columns)",
            full_output_path,
            len(combined_df),
            len(combined_df.columns),
        )

    return harmonized_folds


def main():
    """Main execution function."""
    if not NODE_ATTRIBUTES_TEMPORAL.exists():
        logger.error("Temporal features not found: %s", NODE_ATTRIBUTES_TEMPORAL)
        sys.exit(1)
    
    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest not found: %s", MASTER_MANIFEST)
        sys.exit(1)
    
    features = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    output_dir = Path("data") / "metadata" / "harmonized_folds_cv"
    
    harmonized_folds = harmonize_cv_safe_fold(
        features_df=features,
        manifest_df=manifest,
        cv_splits=None,
        output_dir=output_dir,
        full_output_path=NODE_ATTRIBUTES_HARMONIZED,
    )
    
    if not harmonized_folds:
        logger.error("Harmonization failed")
        sys.exit(1)
    
    _check_harmonization_quality(features, harmonized_folds)


if __name__ == "__main__":
    main()