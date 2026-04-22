"""
Fold-safe neuroHarmonize harmonization for Neuro-CXG.

Pipeline
--------
1. Validate + repair raw temporal features (NaN/Inf/outlier handling).
2. Aggregate 170 AAL ROIs -> 12 brain regions (vectorised mean).
3. 5-fold CV harmonization via neuroHarmonize (ComBat).
4. Quality-check variance retention across folds.
5. Write per-fold harmonized outputs plus combined no-leak output for graph_factory.py.

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
    FEATURE_GROUPS,
    HARMONIZED_FOLDS_DIR,
    HARMONIZATION_UNSEEN_SITE_POLICY,
    K_FOLDS,
    LOBE_MAPPING,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NODE_FEATURES_3D_HARMONIZED,
    NUM_LOBES,
    NUM_TEMPORAL_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── constants ─────────────────────────────────────────────────────────────────
FEATURE_TYPES = FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["frequency"]

NAN_REMOVAL_THRESHOLD = 0.5   # drop subjects with >50 % NaN
OUTLIER_STD_THRESHOLD = 5     # cap outliers beyond ±5 σ
VARIANCE_WARNING_THRESHOLD = 30.0  # warn when >30 % features lose/gain variance
VARIANCE_RETENTION_LOW = 0.7     # flag features retaining <70 % of original variance
VARIANCE_RETENTION_HIGH = 1.3    # flag features gaining  >30 % of original variance
COMBAT_MIN_VARIANCE = 1e-8       # treat near-constant channels as constant for ComBat stability
QUALITY_MIN_REFERENCE_VARIANCE = 1e-8  # avoid unstable retention ratios from tiny denominators

COMBAT_SHRINKAGE = 0.0  # Disabled - neuroHarmonize doesn't support shrink param directly
                      # Use EB.extend = False for less aggressive harmonization instead


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
    # NOTE: only run when clip_outliers=True. In harmonize_cv_safe_fold(),
    # this is set to False so fold-safe clipping is applied inside the fold loop.
    if clip_outliers:
        means = df[cols].mean()
        stds = df[cols].std().replace(0, np.nan)
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
    """Drop (near-)constant columns that would destabilize ComBat."""
    variances = features.var()
    constant_cols = variances[variances <= COMBAT_MIN_VARIANCE].index.tolist()
    kept_cols = [c for c in features.columns if c not in constant_cols]
    if constant_cols:
        logger.info(
            "Dropping %d near-constant features before harmonization (var <= %.1e); "
            "restored unchanged by _restore_constant_features.",
            len(constant_cols),
            COMBAT_MIN_VARIANCE,
        )
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


def _safe_harmonization_apply(
    apply_features: pd.DataFrame,
    apply_covariates: pd.DataFrame,
    model: object,
    *,
    seen_sites: Optional[set] = None,
    context: str = "apply",
) -> np.ndarray:
    """Apply ComBat safely when SITE levels in apply data are unseen in train.

    neuroHarmonize can error when categorical levels appear at apply time but
    were absent during fit. This helper applies harmonization only to rows from
    seen SITE levels and leaves unseen rows unchanged.
    """
    if apply_features.empty:
        return np.empty((0, apply_features.shape[1]), dtype=np.float64)

    raw_values = apply_features.values
    harmonized = raw_values.copy()

    if seen_sites is None or "SITE" not in apply_covariates.columns:
        return harmonizationApply(raw_values, apply_covariates, model)

    site_vals = apply_covariates["SITE"].astype(str).to_numpy()
    seen_mask = np.isin(site_vals, list(seen_sites))

    if (~seen_mask).any():
        unseen_sites = sorted(set(site_vals[~seen_mask].tolist()))
        logger.warning(
            "%s contains %d/%d rows from unseen SITE levels %s; "
            "keeping those rows unharmonized",
            context,
            int((~seen_mask).sum()),
            len(site_vals),
            unseen_sites,
        )

    if seen_mask.any():
        seen_idx = np.where(seen_mask)[0]
        try:
            seen_harm = harmonizationApply(
                raw_values[seen_idx],
                apply_covariates.iloc[seen_idx].reset_index(drop=True),
                model,
            )
            harmonized[seen_idx] = seen_harm
        except Exception as exc:
            logger.error(
                "%s harmonization failed on seen-site subset (%s); "
                "falling back to unharmonized rows for this subset",
                context,
                exc,
            )

    return harmonized


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
                if not vf.empty:
                    vf[col] += rng.normal(0, 1e-8, len(vf))
        model, train_harm = harmonizationLearn(
            tf.values,
            train_covariates,
        )
        if vf.empty:
            val_harm = np.empty((0, len(train_features.columns)))
        else:
            train_sites = set(train_covariates["SITE"].astype(str).tolist())
            val_harm = _safe_harmonization_apply(
                vf,
                val_covariates,
                model,
                seen_sites=train_sites,
                context="Fold validation apply",
            )
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


def _harmonize_train_apply_pair(
    train_data: pd.DataFrame,
    apply_data: pd.DataFrame,
    train_manifest: pd.DataFrame,
    apply_manifest: pd.DataFrame,
) -> Tuple[Optional[object], pd.DataFrame, pd.DataFrame]:
    """Fit harmonization on train_data and apply to apply_data (strict no-leak)."""
    train_data = train_data.copy().reset_index(drop=True)
    apply_data = apply_data.copy().reset_index(drop=True)
    train_manifest = train_manifest.copy().reset_index(drop=True)
    apply_manifest = apply_manifest.copy().reset_index(drop=True)

    cols = _feat_cols(train_data)

    # Fold-safe NaN imputation: fit medians on train only, apply to all splits.
    train_medians = train_data[cols].median()
    train_data[cols] = train_data[cols].fillna(train_medians)
    if not apply_data.empty:
        apply_data[cols] = apply_data[cols].fillna(train_medians)

    # Fold-safe outlier clipping: fit on train only, apply to target split.
    train_data, clip_bounds = _outlier_clip_fit(train_data)
    if not apply_data.empty:
        apply_data = _outlier_clip_apply(apply_data, clip_bounds)

    train_covariates = _prepare_covariates(train_manifest, train_data)
    if apply_data.empty:
        apply_covariates = pd.DataFrame(columns=train_covariates.columns)
    else:
        apply_covariates = _prepare_covariates(apply_manifest, apply_data)

    train_features = train_data.drop(columns=["subject_id"])
    if apply_data.empty:
        apply_features = pd.DataFrame(columns=train_features.columns, index=apply_data.index)
    else:
        apply_features = apply_data.drop(columns=["subject_id"])

    train_features, kept_cols, dropped_cols = _remove_constant_features(train_features)
    apply_features = apply_features.reindex(columns=kept_cols, fill_value=0.0)

    model, train_harmonized, apply_harmonized = _harmonize_fold(
        train_features,
        apply_features,
        train_covariates,
        apply_covariates,
    )

    train_restored = _restore_constant_features(train_harmonized, train_data, kept_cols, dropped_cols)
    train_restored = pd.concat([train_data[["subject_id"]], train_restored], axis=1)
    train_lobes = _clip_outliers(aggregate_to_lobes(train_restored))

    if apply_data.empty:
        apply_lobes = pd.DataFrame(columns=train_lobes.columns)
    else:
        apply_restored = _restore_constant_features(apply_harmonized, apply_data, kept_cols, dropped_cols)
        apply_restored = pd.concat([apply_data[["subject_id"]], apply_restored], axis=1)
        apply_lobes = _clip_outliers(aggregate_to_lobes(apply_restored))

    return model, train_lobes, apply_lobes


def _write_ordered_subject_csv(df: pd.DataFrame, subject_order: List[str], output_path: Path) -> None:
    """Write CSV ordered by subject_id according to subject_order."""
    if df.empty:
        logger.warning("No rows to save for %s", output_path)
        return

    dedup = df.drop_duplicates(subset=["subject_id"], keep="first").set_index("subject_id")
    keep_order = [sid for sid in subject_order if sid in dedup.index]
    ordered = dedup.reindex(keep_order).dropna(how="all").reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(output_path, index=False)
    logger.info("Saved harmonized features -> %s (%d subjects)", output_path, len(ordered))


# ── 5. Quality verification ───────────────────────────────────────────────

def _check_harmonization_quality(
    original_df: pd.DataFrame,
    harmonized_folds: List[HarmonizationFold],
) -> Dict[str, object]:
    """Check variance retention and NaN introduction across all harmonized folds."""
    logger.info("=" * 60)
    logger.info("HARMONIZATION QUALITY CHECK")
    logger.info("=" * 60)

    val_only = [f.val for f in harmonized_folds if not f.val.empty]
    if val_only:
        # Validation slices are disjoint across folds and avoid repeated train subjects.
        all_harmonized = pd.concat(val_only, ignore_index=True).drop_duplicates(subset=["subject_id"])
    else:
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

    # Match the preprocessing scale used before ComBat so retention ratios are
    # numerically meaningful (raw-vs-log comparisons otherwise overstate loss).
    original_for_quality = original_df[common].copy()
    spectral = ("delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power")
    spectral_cols = [c for c in common if any(s in c for s in spectral)]
    for col in spectral_cols:
        mask = original_for_quality[col] > 0
        original_for_quality.loc[mask, col] = np.log1p(original_for_quality.loc[mask, col])

    orig_var_series = original_for_quality.var()
    harm_var_series = all_harmonized[common].var()
    orig_var = orig_var_series.mean()
    harm_var = harm_var_series.mean()
    var_retention = harm_var / orig_var if orig_var != 0 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        per_feat_ret = (harm_var_series / orig_var_series).replace([np.inf, -np.inf], np.nan)

    retention_df = pd.DataFrame(
        {
            "feature": common,
            "orig_variance": orig_var_series.reindex(common).values,
            "harm_variance": harm_var_series.reindex(common).values,
            "retention": per_feat_ret.reindex(common).values,
        }
    )
    retention_df["status"] = "within"
    retention_df.loc[retention_df["retention"] < VARIANCE_RETENTION_LOW, "status"] = "low"
    retention_df.loc[retention_df["retention"] > VARIANCE_RETENTION_HIGH, "status"] = "high"

    report_path = HARMONIZED_FOLDS_DIR / "harmonization_variance_retention.csv"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        retention_df.sort_values("retention", ascending=True).to_csv(report_path, index=False)
        logger.info("  Saved per-feature variance retention report -> %s", report_path)
    except Exception as exc:
        logger.warning("  Failed to save variance retention report: %s", exc)

    stable_mask = orig_var_series > QUALITY_MIN_REFERENCE_VARIANCE
    stable_ret = per_feat_ret[stable_mask].dropna()
    if stable_ret.empty:
        # Fallback: if all channels are tiny-variance, report on whatever is finite.
        stable_ret = per_feat_ret.dropna()
        logger.warning(
            "No channels above reference variance %.1e; using all finite retention ratios",
            QUALITY_MIN_REFERENCE_VARIANCE,
        )

    n_total = len(stable_ret)
    n_low = int((stable_ret < VARIANCE_RETENTION_LOW).sum())
    n_high = int((stable_ret > VARIANCE_RETENTION_HIGH).sum())
    n_within = n_total - n_low - n_high

    orig_nans = int(original_df[orig_cols].isna().sum().sum())
    harm_nans = int(all_harmonized[harm_cols].isna().sum().sum())
    nans_introduced = harm_nans - orig_nans

    logger.info("  Original variance  : %.4f", orig_var)
    logger.info("  Harmonized variance: %.4f", harm_var)
    logger.info("  Variance retention : %.2f%%", var_retention * 100)
    logger.info(
        "  Stable-channel denominator mask: %d/%d channels (var > %.1e)",
        int(stable_mask.sum()),
        len(common),
        QUALITY_MIN_REFERENCE_VARIANCE,
    )
    if n_total:
        logger.info(
            "  Per-feature: %.1f%% within, %.1f%% low, %.1f%% high",
            100 * n_within / n_total,
            100 * n_low / n_total,
            100 * n_high / n_total,
        )
        if 100 * n_low / n_total > VARIANCE_WARNING_THRESHOLD:
            logger.info(
                "  Many features lost >30%% variance after harmonization "
                "(expected: inputs are pre-z-scored so ComBat site removal naturally reduces total variance; "
                "lobe-level signal is preserved)."
            )
            low_examples = retention_df[retention_df["status"] == "low"]
            if not low_examples.empty:
                logger.warning(
                    "  Lowest-retention features (<%.2f): %s",
                    VARIANCE_RETENTION_LOW,
                    low_examples.sort_values("retention").head(10)["feature"].tolist(),
                )
        if 100 * n_high / n_total > VARIANCE_WARNING_THRESHOLD:
            logger.warning("  Many features gained >30%% variance after harmonization")
            high_examples = retention_df[retention_df["status"] == "high"]
            if not high_examples.empty:
                logger.warning(
                    "  Highest-retention features (>%.2f): %s",
                    VARIANCE_RETENTION_HIGH,
                    high_examples.sort_values("retention", ascending=False).head(10)["feature"].tolist(),
                )
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
    # so that train-only medians are used - prevents label-independent leakage.
    features_safe = repair_features(features_df, impute_nans=False, clip_outliers=False)

    if "split" not in manifest_df.columns:
        raise ValueError("manifest_df must contain 'split' column for strict leakage prevention!")

    # Align manifest to feature ordering and drop rows missing manifest metadata.
    aligned_manifest = manifest_df.set_index("subject_id").reindex(features_safe["subject_id"])
    missing_manifest = int(aligned_manifest["split"].isna().sum())
    if missing_manifest:
        logger.warning("Dropping %d subjects missing manifest split/site metadata", missing_manifest)
        keep_mask = ~aligned_manifest["split"].isna().to_numpy()
        features_safe = features_safe.loc[keep_mask].reset_index(drop=True)
        aligned_manifest = aligned_manifest.loc[keep_mask]
    aligned_manifest = aligned_manifest.reset_index()

    train_mask = aligned_manifest["split"] == "train"
    train_data_all = features_safe[train_mask].reset_index(drop=True)
    train_manifest_all = aligned_manifest[train_mask].reset_index(drop=True)
    holdout_data = features_safe[~train_mask].reset_index(drop=True)
    holdout_manifest = aligned_manifest[~train_mask].reset_index(drop=True)

    if train_data_all.empty:
        raise ValueError("No training subjects available after manifest alignment")

    # Build CV splits on train subjects only.
    if cv_splits is None:
        if "cv_fold" not in train_manifest_all.columns:
            raise ValueError(
                "manifest_df must contain 'cv_fold' for per-fold harmonization artifacts"
            )
        fold_values = pd.to_numeric(train_manifest_all["cv_fold"], errors="coerce")
        if fold_values.isna().any():
            raise ValueError("cv_fold contains non-numeric or missing values")
        fold_values = fold_values.astype(int).to_numpy()
        if fold_values.min() < 0 or fold_values.max() >= K_FOLDS:
            raise ValueError(
                f"Invalid cv_fold values [{fold_values.min()}, {fold_values.max()}], expected [0, {K_FOLDS - 1}]"
            )
        cv_splits = [
            (np.where(fold_values != f)[0], np.where(fold_values == f)[0])
            for f in range(K_FOLDS)
        ]

    unseen_policy = str(HARMONIZATION_UNSEEN_SITE_POLICY).strip().lower()
    if unseen_policy not in {"passthrough", "fail"}:
        logger.warning(
            "Unknown HARMONIZATION_UNSEEN_SITE_POLICY=%r; falling back to 'passthrough'",
            HARMONIZATION_UNSEEN_SITE_POLICY,
        )
        unseen_policy = "passthrough"

    # Audit unseen SITE coverage up-front so policy violations fail before any fold
    # harmonization runs.
    fold_site_audit: List[Dict[str, object]] = []
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_train_manifest = train_manifest_all.iloc[train_idx].reset_index(drop=True)
        fold_val_manifest = train_manifest_all.iloc[val_idx].reset_index(drop=True)
        train_sites = set(fold_train_manifest["SITE_ID"].astype(str).tolist())
        val_sites = set(fold_val_manifest["SITE_ID"].astype(str).tolist())
        unseen_val_sites = sorted(list(val_sites - train_sites))
        unseen_rows = int(fold_val_manifest["SITE_ID"].astype(str).isin(unseen_val_sites).sum())
        fold_site_audit.append(
            {
                "fold": fold,
                "train_sites": len(train_sites),
                "val_sites": len(val_sites),
                "unseen_sites": unseen_val_sites,
                "unseen_site_count": len(unseen_val_sites),
                "unseen_row_count": unseen_rows,
                "val_rows": int(len(fold_val_manifest)),
            }
        )

    folds_with_unseen = [r for r in fold_site_audit if r["unseen_site_count"] > 0]
    total_unseen_rows = int(sum(r["unseen_row_count"] for r in folds_with_unseen))
    all_rows_unseen = bool(
        fold_site_audit
        and all(
            int(row["val_rows"]) > 0 and int(row["unseen_row_count"]) == int(row["val_rows"])
            for row in fold_site_audit
        )
    )
    logger.info(
        "Fold unseen-site audit: %d/%d folds have unseen validation sites (%d total unseen rows), policy=%s",
        len(folds_with_unseen),
        len(fold_site_audit),
        total_unseen_rows,
        unseen_policy,
    )
    for row in folds_with_unseen:
        logger.warning(
            "Fold %d unseen SITE audit: %d/%d val rows from unseen sites %s",
            row["fold"],
            row["unseen_row_count"],
            row["val_rows"],
            row["unseen_sites"],
        )

    if all_rows_unseen:
        logger.error(
            "All validation rows across all folds are unseen SITE levels. "
            "This configuration is incompatible with fold-safe harmonization and will "
            "produce train/val feature-space mismatch under passthrough behavior."
        )

    audit_dir = Path(output_dir) if output_dir is not None else HARMONIZED_FOLDS_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "fold_unseen_site_audit.csv"
    audit_df = pd.DataFrame(
        [
            {
                "fold": row["fold"],
                "train_site_count": row["train_sites"],
                "val_site_count": row["val_sites"],
                "unseen_site_count": row["unseen_site_count"],
                "unseen_row_count": row["unseen_row_count"],
                "val_row_count": row["val_rows"],
                "unseen_sites": "|".join(row["unseen_sites"]),
            }
            for row in fold_site_audit
        ]
    )
    audit_df.to_csv(audit_path, index=False)
    logger.info("Saved fold unseen-site audit → %s", audit_path)

    if folds_with_unseen and unseen_policy == "fail":
        raise RuntimeError(
            "Fold harmonization aborted by HARMONIZATION_UNSEEN_SITE_POLICY='fail': "
            f"{len(folds_with_unseen)} fold(s) contain validation-only SITE levels. "
            "Recommended (Option A): use standard StratifiedKFold CV for publication runs "
            "(do not pass --site-stratified-cv)."
        )

    harmonized_folds: List[HarmonizationFold] = []
    train_subject_order = train_data_all["subject_id"].tolist()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Per-fold harmonization: fit on fold-train, apply to fold-val only.
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning("Skipping fold %d due to empty train/val split", fold)
            continue

        fold_train_data = train_data_all.iloc[train_idx].reset_index(drop=True)
        fold_val_data = train_data_all.iloc[val_idx].reset_index(drop=True)
        fold_train_manifest = train_manifest_all.iloc[train_idx].reset_index(drop=True)
        fold_val_manifest = train_manifest_all.iloc[val_idx].reset_index(drop=True)

        unseen_val_sites = fold_site_audit[fold]["unseen_sites"]
        if unseen_val_sites:
            logger.warning(
                "Fold %d has %d validation-only SITE levels not present in fold-train: %s",
                fold,
                len(unseen_val_sites),
                unseen_val_sites,
            )

        model, train_lobes, val_lobes = _harmonize_train_apply_pair(
            fold_train_data,
            fold_val_data,
            fold_train_manifest,
            fold_val_manifest,
        )

        harmonized_folds.append(
            HarmonizationFold(
                fold=fold,
                train=train_lobes,
                val=val_lobes,
                train_idx=np.asarray(train_idx),
                val_idx=np.asarray(val_idx),
                model=model,
            )
        )

        if output_dir is not None:
            fold_df = pd.concat([train_lobes, val_lobes], ignore_index=True)
            fold_path = output_dir / f"harmonized_fold_{fold}.csv"
            _write_ordered_subject_csv(fold_df, train_subject_order, fold_path)

    # Global no-leak output: fit on full train split, apply to holdout (val+test).
    if full_output_path is not None:
        _, train_lobes_full, holdout_lobes = _harmonize_train_apply_pair(
            train_data_all,
            holdout_data,
            train_manifest_all,
            holdout_manifest,
        )
        combined_df = pd.concat([train_lobes_full, holdout_lobes], ignore_index=True)
        full_subject_order = features_safe["subject_id"].tolist()
        _write_ordered_subject_csv(combined_df, full_subject_order, Path(full_output_path))

    return harmonized_folds


# ── 6. Spatial feature harmonization (conf_std / detection_count) ────────────

# These two features encode scanner quality rather than anatomy:
#   conf_std         — std-dev of YOLO detection confidence across 7 fMRI slices
#   detection_count  — number of slices where the region was detected
# Kruskal-Wallis test (March 2026) confirms 14/24 of these columns have
# highly significant site effects (p<0.001), making them scanner proxies.
# x, y, z_depth, and size are kept unchanged because they represent physical brain anatomy.
_SPATIAL_SITE_COLS = (
    [f"{name}_conf_std" for name in LOBE_NAMES.values()]
    + [f"{name}_detection_count" for name in LOBE_NAMES.values()]
)


def harmonize_spatial_features(
    spatial_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    output_path: Path = NODE_FEATURES_3D_HARMONIZED,
) -> pd.DataFrame:
    """Fold-safe ComBat harmonization of conf_std + detection_count spatial features.

    Fits on the train split subjects only and applies to val + test, then writes
    a combined CSV to *output_path*. The x / y / z_depth / size columns are
    copied through without any modification.

    Args:
        spatial_df:  Full node_features_3d.csv DataFrame (index NOT set).
        manifest_df: Master manifest with 'subject_id', 'split', 'SITE_ID',
                     'AGE_AT_SCAN', 'SEX', 'DX_GROUP' columns.
        output_path: Where to write the harmonized CSV.

    Returns:
        The harmonized DataFrame (all subjects, same columns as *spatial_df*).
    """
    logger.info("=" * 60)
    logger.info("SPATIAL FEATURE HARMONIZATION (conf_std / detection_count)")
    logger.info("=" * 60)

    spatial_df = spatial_df.copy()

    # Identify which site-proxy columns are actually present in this CSV.
    site_cols = [c for c in _SPATIAL_SITE_COLS if c in spatial_df.columns]
    if not site_cols:
        logger.warning("No conf_std / detection_count columns found — skipping spatial harmonization")
        return spatial_df

    logger.info("  Harmonizing %d site-proxy columns: %s …", len(site_cols), site_cols[:4])

    # Merge with manifest to get SITE_ID / DX_GROUP / split.
    # Drop any manifest-derived columns already present in spatial_df to avoid _x/_y collisions.
    _MANIFEST_COLS = ["SITE_ID", "split", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ",
                      "HANDEDNESS_CATEGORY", "TR", "cv_fold"]
    spatial_no_site = spatial_df.drop(
        columns=[c for c in _MANIFEST_COLS if c in spatial_df.columns]
    )
    merged = spatial_no_site.merge(
        manifest_df[["subject_id", "split", "SITE_ID", "AGE_AT_SCAN", "SEX", "DX_GROUP"]],
        on="subject_id",
        how="inner",
    )
    if merged.empty:
        logger.error("Spatial + manifest merge produced zero rows — skipping harmonization")
        return spatial_df

    train_mask = merged["split"] == "train"
    train_df = merged[train_mask].copy()
    other_df  = merged[~train_mask].copy()

    # Build neuroHarmonize covariate matrices: SITE, AGE_AT_SCAN, SEX, DX_GROUP.
    def _build_cov(df: pd.DataFrame) -> pd.DataFrame:
        cov = df[["SITE_ID", "AGE_AT_SCAN", "SEX", "DX_GROUP"]].copy()
        cov = cov.rename(columns={"SITE_ID": "SITE"})
        cov["SITE"] = cov["SITE"].astype(str)
        cov["AGE_AT_SCAN"] = pd.to_numeric(cov["AGE_AT_SCAN"], errors="coerce").fillna(
            cov["AGE_AT_SCAN"].median()
        )
        cov["SEX"] = pd.to_numeric(cov["SEX"], errors="coerce").fillna(1).astype(int)
        cov["DX_GROUP"] = pd.to_numeric(cov["DX_GROUP"], errors="coerce").fillna(1).astype(int)
        return cov

    train_cov = _build_cov(train_df)
    other_cov = _build_cov(other_df)

    train_feats = train_df[site_cols].fillna(0.0)
    other_feats = other_df[site_cols].fillna(0.0) if not other_df.empty else None

    # Drop constant columns (ComBat fails on zero-variance features).
    var = train_feats.var()
    constant = var[var == 0].index.tolist()
    active_cols = [c for c in site_cols if c not in constant]
    if constant:
        logger.info("  Dropping %d constant spatial columns before ComBat", len(constant))

    if not active_cols:
        logger.warning("All spatial site-proxy columns are constant — skipping harmonization")
        return spatial_df

    try:
        model, train_harm = harmonizationLearn(
            train_feats[active_cols].values,
            train_cov,
        )
        train_df = train_df.copy()
        train_df[active_cols] = train_harm

        if other_feats is not None and not other_df.empty:
            seen_sites = set(train_cov["SITE"].astype(str).tolist())
            other_harm = _safe_harmonization_apply(
                other_feats[active_cols],
                other_cov,
                model,
                seen_sites=seen_sites,
                context="Spatial feature apply",
            )
            other_df = other_df.copy()
            other_df[active_cols] = other_harm

        logger.info("  ComBat harmonization successful for %d subjects", len(merged))
    except Exception as exc:
        logger.error("  Spatial harmonization failed (%s) — writing raw features unchanged", exc)
        return spatial_df

    # Reassemble: harmonized train + other rows, then re-merge with original
    # non-site-proxy columns (x, y, z_depth, size, spatial_complete, etc.).
    harm_parts = [train_df[["subject_id"] + active_cols]]
    if not other_df.empty:
        harm_parts.append(other_df[["subject_id"] + active_cols])
    harm_site_cols = pd.concat(harm_parts, ignore_index=True)

    result = spatial_df.drop(columns=active_cols).merge(harm_site_cols, on="subject_id", how="left")
    # Subjects not in manifest keep original values.
    for col in active_cols:
        mask = result[col].isna()
        if mask.any():
            result.loc[mask, col] = spatial_df.loc[spatial_df["subject_id"].isin(result.loc[mask, "subject_id"]), col].values

    # Restore original column order.
    result = result[[c for c in spatial_df.columns if c in result.columns]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("  Saved harmonized spatial features → %s", output_path)

    return result


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
    
    output_dir = HARMONIZED_FOLDS_DIR
    
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

    # Stage 12b — spatial harmonization removed (conf_std/detection_count columns no longer exist)
    # Spatial features are copied directly without harmonization
    if NODE_FEATURES_3D.exists():
        spatial_df = pd.read_csv(NODE_FEATURES_3D)
        spatial_df.to_csv(NODE_FEATURES_3D_HARMONIZED, index=False)
        logger.info("Spatial features copied without harmonization (legacy columns removed)")


if __name__ == "__main__":
    main()
