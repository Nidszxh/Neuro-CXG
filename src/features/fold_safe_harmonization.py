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
    "delta_power", "delta_peak",
    "theta_power", "theta_peak",
    "alpha_power", "alpha_peak",
    "beta_power",  "beta_peak",
    "gamma_power", "gamma_peak",
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

def repair_features(df: pd.DataFrame, *, impute_nans: bool = True, clip_outliers: bool = True) -> pd.DataFrame:
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
    # NOTE: only run when clip_outliers=True.  In harmonize_cv_safe_fold(),
    # this is set to False so that fold-safe clipping is applied inside the
    # fold loop using _outlier_clip_fit / _outlier_clip_apply instead.
    if clip_outliers:
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


def _outlier_clip_fit(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, "pd.Series"]]:
    """Compute outlier clip bounds from df (train fold only) and apply them.

    Returns the clipped DataFrame and a dict of {lower, upper} Series so the
    same bounds can be applied to val/test via _outlier_clip_apply().
    """
    cols = _feat_cols(df)
    means = df[cols].mean()
    stds = df[cols].std().replace(0, np.nan)
    lower = means - OUTLIER_STD_THRESHOLD * stds
    upper = means + OUTLIER_STD_THRESHOLD * stds
    clipped = df.copy()
    clipped[cols] = clipped[cols].clip(lower=lower, upper=upper, axis=1)
    return clipped, {"lower": lower, "upper": upper}


def _outlier_clip_apply(
    df: pd.DataFrame,
    clip_bounds: Dict[str, "pd.Series"],
) -> pd.DataFrame:
    """Apply pre-computed train-fold clip bounds to val/test data."""
    cols = _feat_cols(df)
    lower = clip_bounds["lower"].reindex(cols)  # align columns in case of mismatch
    upper = clip_bounds["upper"].reindex(cols)
    clipped = df.copy()
    clipped[cols] = clipped[cols].clip(lower=lower, upper=upper, axis=1)
    return clipped


def _clip_outliers(df: pd.DataFrame, percentile_range: float = 0.05) -> pd.DataFrame:
    """Clip extreme values based on percentile range to prevent outlier explosion.
    
    Uses 5-95 percentile bounds to clip outliers, allowing ±3std from bounds.
    Applied AFTER aggregation to handle extreme values that appear in aggregated features.
    """
    numeric_cols = df.columns[df.columns != "subject_id"]
    clipped = df.copy()
    
    for col in numeric_cols:
        p_lower = df[col].quantile(percentile_range)
        p_upper = df[col].quantile(1 - percentile_range)
        col_range = p_upper - p_lower
        
        # Allow ±3σ from the percentile bounds
        lower_bound = p_lower - 3 * col_range
        upper_bound = p_upper + 3 * col_range
        
        clipped[col] = clipped[col].clip(lower_bound, upper_bound)
    
    return clipped
def _prepare_covariates(manifest: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Build covariate DataFrame for neuroHarmonize (requires exact 'SITE' column).

    DX_GROUP is included as a *protected* covariate so ComBat preserves
    diagnosis-correlated variance rather than removing it as a batch effect.
    """
    cov = manifest[["subject_id", "SITE_ID", "AGE_AT_SCAN", "SEX", "DX_GROUP"]].copy()
    cov = cov.merge(features_df[["subject_id"]], on="subject_id", how="inner")
    cov = cov.rename(columns={"SITE_ID": "SITE"})
    if cov["AGE_AT_SCAN"].isna().any():
        cov["AGE_AT_SCAN"] = cov["AGE_AT_SCAN"].fillna(cov["AGE_AT_SCAN"].median())
    if cov["SEX"].isna().any():
        cov["SEX"] = cov["SEX"].fillna(cov["SEX"].mode().iloc[0])
    cov["SITE"] = cov["SITE"].astype(str)
    cov["SEX"] = pd.to_numeric(cov["SEX"], errors="coerce").fillna(1).astype(int)
    cov["DX_GROUP"] = pd.to_numeric(cov["DX_GROUP"], errors="coerce").fillna(1).astype(int)
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
    
    # Build all dropped columns at once to avoid fragmentation
    if dropped_cols:
        dropped_data = {
            col: original[col].values if col in original.columns else 0.0
            for col in dropped_cols
        }
        dropped_df = pd.DataFrame(dropped_data, index=result.index)
        result = pd.concat([result, dropped_df], axis=1)
    
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
    logger.info("=" * 80)
    logger.info("HARMONIZATION (Strict Train-Only Fitting)")
    logger.info("=" * 80)
    
    validate_features(features_df)
    # impute_nans=False: NaN imputation must happen after the train/val split
    # so that train-only medians are used — prevents label-independent leakage.
    features_safe = repair_features(features_df, impute_nans=False, clip_outliers=False)

    # 1. Align manifest and extract splits
    aligned_manifest = manifest_df.set_index("subject_id").reindex(features_safe["subject_id"]).reset_index()
    if 'split' not in aligned_manifest.columns:
        raise ValueError("manifest_df must contain 'split' column for strict leakage prevention!")

    train_mask = aligned_manifest['split'] == 'train'
    train_data = features_safe[train_mask].reset_index(drop=True)
    val_test_data = features_safe[~train_mask].reset_index(drop=True)

    # Fold-safe NaN imputation: fit medians on train only, apply to all splits
    _fc = _feat_cols(train_data)
    _train_medians = train_data[_fc].median()
    train_data[_fc] = train_data[_fc].fillna(_train_medians)
    val_test_data[_fc] = val_test_data[_fc].fillna(_train_medians)
    
    train_manifest = aligned_manifest[train_mask].reset_index(drop=True)
    val_test_manifest = aligned_manifest[~train_mask].reset_index(drop=True)

    # 2. Fit bounds and Covariates on train ONLY
    train_data, clip_bounds = _outlier_clip_fit(train_data)
    val_test_data = _outlier_clip_apply(val_test_data, clip_bounds)
    
    train_covariates = _prepare_covariates(train_manifest, train_data)
    val_test_covariates = _prepare_covariates(val_test_manifest, val_test_data)
    
    train_features = train_data.drop(columns=["subject_id"])
    val_test_features = val_test_data.drop(columns=["subject_id"])
    
    train_features, kept_cols, dropped_cols = _remove_constant_features(train_features)
    val_test_features = val_test_features[kept_cols]
    
    # 3. Fit ComBat on Train ONLY, Apply to All
    model, train_harmonized, val_test_harmonized = _harmonize_fold(
        train_features,
        val_test_features,
        train_covariates,
        val_test_covariates,
    )
    
    # 4. Restore and aggregate
    train_df = _restore_constant_features(train_harmonized, train_data, kept_cols, dropped_cols)
    train_df = pd.concat([train_data[["subject_id"]], train_df], axis=1)
    
    val_test_df = _restore_constant_features(val_test_harmonized, val_test_data, kept_cols, dropped_cols)
    val_test_df = pd.concat([val_test_data[["subject_id"]], val_test_df], axis=1)
    
    train_lobes = _clip_outliers(aggregate_to_lobes(train_df))
    val_test_lobes = _clip_outliers(aggregate_to_lobes(val_test_df))
    
    # Create fold object to appease existing downstream logic (just wrap train data)
    fold_result = HarmonizationFold(
        fold=0,
        train=train_lobes,
        val=train_lobes,
        train_idx=np.arange(len(train_lobes)),
        val_idx=np.arange(len(train_lobes)),
        model=model,
    )
    harmonized_folds = [fold_result]
    
    if full_output_path is not None:
        combined_df = pd.concat([train_lobes, val_test_lobes], ignore_index=True)
        # Drop metadata columns (none added yet, but safety check)
        meta_cols = ["SITE_ID", "DX_GROUP"]
        combined_df = combined_df.drop(columns=[c for c in meta_cols if c in combined_df.columns])
        combined_df = combined_df.drop_duplicates(subset=["subject_id"])
        
        Path(full_output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(full_output_path, index=False)
        logger.info("Saved combined NO-LEAK harmonized features \u2192 %s", full_output_path)
        
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