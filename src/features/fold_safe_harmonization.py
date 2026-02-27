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


def _placeholder_repair_features_end():
    pass


class _FeatureRepairer_DELETED:
    """Handles repair operations on temporal features."""
    
    @staticmethod
    def repair(df: pd.DataFrame, impute_nans: bool = False) -> Tuple[pd.DataFrame, RepairStats]:
        """
        Repair temporal features with intelligent imputation.
        
        Strategy:
        1. Remove subjects with >50% NaN values
        2. Replace infinities with feature-specific bounds
        3. Impute remaining NaNs using feature-wise medians
        4. Cap outliers beyond 5 standard deviations
        """
        logger.info("Attempting to repair temporal features...")
        
        feature_cols = FeatureValidator._get_feature_columns(df)
        df_clean = df.copy()
        
        stats = RepairStats(subjects_before=len(df))
        
        # Step 1: Remove subjects with excessive NaN values
        df_clean, removed_count = FeatureRepairer._remove_invalid_subjects(
            df_clean, feature_cols, NAN_REMOVAL_THRESHOLD
        )
        stats.subjects_removed = removed_count
        
        # Step 2: Replace infinite values
        df_clean, inf_count = FeatureRepairer._replace_infinities(df_clean, feature_cols)
        stats.infs_replaced = inf_count
        
        # Step 3: Apply spectral transformations
        df_clean, spectral_count = FeatureRepairer._transform_spectral_features(
            df_clean, feature_cols
        )
        stats.spectral_log_transformed = spectral_count
        
        # Step 4: Impute NaN values if requested
        if impute_nans:
            df_clean, imputed_count = FeatureRepairer._impute_nans(df_clean, feature_cols)
            stats.nans_imputed = imputed_count
        
        # Step 5: Cap outliers
        df_clean, outlier_count = FeatureRepairer._cap_outliers(
            df_clean, feature_cols, OUTLIER_STD_THRESHOLD
        )
        stats.outliers_detected = outlier_count
        
        FeatureRepairer._log_repair_summary(stats)
        
        return df_clean, stats
    
    @staticmethod
    def _remove_invalid_subjects(
        df: pd.DataFrame,
        feature_cols: List[str],
        threshold: float
    ) -> Tuple[pd.DataFrame, int]:
        """Remove subjects with excessive NaN values."""
        nan_counts = df[feature_cols].isna().sum(axis=1)
        invalid_subjects = nan_counts > (len(feature_cols) * threshold)
        
        if invalid_subjects.sum() > 0:
            logger.warning(
                "Removing %d subjects with >%d%% NaN values",
                invalid_subjects.sum(),
                int(threshold * 100),
            )
            df = df[~invalid_subjects].reset_index(drop=True)
        
        return df, int(invalid_subjects.sum())
    
    @staticmethod
    def _replace_infinities(
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """Replace infinite values with feature-specific bounds."""
        inf_count = 0
        
        for col in feature_cols:
            pos_inf = df[col] == np.inf
            neg_inf = df[col] == -np.inf
            
            if pos_inf.sum() > 0:
                valid_vals = df.loc[~(pos_inf | neg_inf | df[col].isna()), col]
                if len(valid_vals) > 0:
                    upper_bound = valid_vals.quantile(0.99)
                    df.loc[pos_inf, col] = upper_bound
                    inf_count += pos_inf.sum()
            
            if neg_inf.sum() > 0:
                valid_vals = df.loc[~(pos_inf | neg_inf | df[col].isna()), col]
                if len(valid_vals) > 0:
                    lower_bound = valid_vals.quantile(0.01)
                    df.loc[neg_inf, col] = lower_bound
                    inf_count += neg_inf.sum()
        
        if inf_count > 0:
            logger.warning("Replaced %d infinite values", inf_count)
        
        return df, inf_count
    
    @staticmethod
    def _transform_spectral_features(
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """Apply log transformation to spectral power features."""
        spectral_features = [
            "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"
        ]
        
        transformed_count = 0
        for col in feature_cols:
            if any(sf in col for sf in spectral_features):
                mask = df[col] > 0
                if mask.sum() > 0:
                    df.loc[mask, col] = np.log1p(df.loc[mask, col])
                    transformed_count += 1
        
        if transformed_count > 0:
            logger.info("Applied log transformation to %d spectral features", transformed_count)
        
        return df, transformed_count
    
    @staticmethod
    def _impute_nans(
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, int]:
        """Impute NaN values using feature-wise medians."""
        nans_before = df[feature_cols].isna().sum().sum()
        
        for col in feature_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
        
        nans_after = df[feature_cols].isna().sum().sum()
        imputed_count = nans_before - nans_after
        
        if imputed_count > 0:
            logger.info("Imputed %d NaN values using feature medians", imputed_count)
        
        return df, imputed_count
    
    @staticmethod
    def _cap_outliers(
        df: pd.DataFrame,
        feature_cols: List[str],
        std_threshold: float
    ) -> Tuple[pd.DataFrame, int]:
        """Cap outliers beyond specified standard deviation threshold."""
        outlier_count = 0
        
        for col in feature_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            if pd.notna(col_mean) and pd.notna(col_std) and col_std > 0:
                lower_bound = col_mean - std_threshold * col_std
                upper_bound = col_mean + std_threshold * col_std
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                if outliers.sum() > 0:
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    outlier_count += outliers.sum()
        
        if outlier_count > 0:
            logger.info("Capped %d outliers beyond %d std", outlier_count, std_threshold)
        
        return df, outlier_count
    
    @staticmethod
    def _log_repair_summary(stats: RepairStats) -> None:
        """Log summary of repair operations."""
        logger.info("=" * 80)
        logger.info("REPAIR SUMMARY")
        logger.info("=" * 80)
        logger.info("  Subjects before: %d", stats.subjects_before)
        logger.info("  Subjects removed: %d", stats.subjects_removed)
        logger.info("  Infinities replaced: %d", stats.infs_replaced)
        logger.info("  Spectral features transformed: %d", stats.spectral_log_transformed)
        logger.info("  NaNs imputed: %d", stats.nans_imputed)
        logger.info("  Outliers capped: %d", stats.outliers_detected)
        logger.info("=" * 80)


class ROIAggregator:
    """Handles aggregation of ROI features to lobe-level features."""
    
    @staticmethod
    def aggregate_to_lobes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 170 ROI features to 12 region features.
        
        Input: subject_id + (170 ROIs x 20 features) = 3401 columns
        Output: subject_id + (12 regions x 20 features) = 241 columns
        """
        logger.info("Aggregating 170 ROIs to 12 regions...")
        
        aggregated_data = []
        
        for _, row in df.iterrows():
            subject_row = {"subject_id": row["subject_id"]}
            
            for lobe_id in range(NUM_LOBES):
                lobe_features = ROIAggregator._aggregate_lobe_features(
                    row, lobe_id, df.columns
                )
                subject_row.update(lobe_features)
            
            aggregated_data.append(subject_row)
        
        aggregated_df = pd.DataFrame(aggregated_data)
        ROIAggregator._validate_output(aggregated_df)
        
        return aggregated_df
    
    @staticmethod
    def _aggregate_lobe_features(
        row: pd.Series,
        lobe_id: int,
        columns: pd.Index
    ) -> Dict[str, float]:
        """Aggregate features for a single lobe."""
        lobe_name = LOBE_NAMES[lobe_id]
        roi_indices = LOBE_MAPPING[lobe_id]
        roi_ids = [roi_id + 1 for roi_id in roi_indices]
        
        weight_cols = [f"roi{roi_id}_volume" for roi_id in roi_ids]
        has_volume_weights = all(col in columns for col in weight_cols)
        
        lobe_features = {}
        
        for feat_name in FEATURE_TYPES:
            aggregated_value = ROIAggregator._aggregate_feature(
                row, roi_ids, feat_name, has_volume_weights
            )
            lobe_features[f"{lobe_name}_{feat_name}"] = aggregated_value
        
        return lobe_features
    
    @staticmethod
    def _aggregate_feature(
        row: pd.Series,
        roi_indices: List[int],
        feat_name: str,
        has_volume_weights: bool
    ) -> float:
        """Aggregate a single feature across ROIs."""
        roi_values = []
        roi_weights = []
        
        for roi_id in roi_indices:
            col_name = f"roi{roi_id}_{feat_name}"
            if col_name in row.index:
                val = row[col_name]
                if pd.notna(val) and np.isfinite(val):
                    roi_values.append(val)
                    if has_volume_weights:
                        w_val = row[f"roi{roi_id}_volume"]
                        roi_weights.append(w_val if np.isfinite(w_val) else np.nan)
        
        if not roi_values:
            return 0.0
        
        if has_volume_weights and len(roi_weights) == len(roi_values):
            return ROIAggregator._weighted_average(roi_values, roi_weights)
        
        return float(np.mean(roi_values))
    
    @staticmethod
    def _weighted_average(values: List[float], weights: List[float]) -> float:
        """Calculate weighted average with validation."""
        weights_array = np.array(weights, dtype=float)
        values_array = np.array(values, dtype=float)
        valid_mask = np.isfinite(weights_array)
        
        if valid_mask.any():
            return float(np.average(values_array[valid_mask], weights=weights_array[valid_mask]))
        
        return float(np.mean(values_array))
    
    @staticmethod
    def _validate_output(df: pd.DataFrame) -> None:
        """Validate aggregated output dimensions."""
        expected_cols = 1 + (NUM_LOBES * NUM_TEMPORAL_FEATURES)
        
        if len(df.columns) != expected_cols:
            logger.warning(
                "Unexpected column count: %d vs expected %d",
                len(df.columns),
                expected_cols,
            )
        
        logger.info(
            "Aggregated to %d regions with %d features each",
            NUM_LOBES,
            NUM_TEMPORAL_FEATURES,
        )
        logger.info(
            "Total columns: %d (subject_id + %d features)",
            len(df.columns),
            NUM_LOBES * NUM_TEMPORAL_FEATURES,
        )


class HarmonizationEngine:
    """Handles the core harmonization logic."""
    
    @staticmethod
    def prepare_covariates(manifest: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare covariate DataFrame for harmonization.

        neuroHarmonize (ComBat) requires a 'SITE' column (exact name) plus any
        continuous/categorical covariates.  We rename 'SITE_ID' → 'SITE' and drop
        'subject_id' so the returned DataFrame is ready to pass directly to
        harmonizationLearn / harmonizationApply.
        """
        covariates = manifest[["subject_id", "SITE_ID", "AGE_AT_SCAN", "SEX"]].copy()
        covariates = covariates.merge(features_df[["subject_id"]], on="subject_id", how="inner")

        # Rename to the column name expected by neuroHarmonize
        covariates = covariates.rename(columns={"SITE_ID": "SITE"})

        # Fill missing demographics with median/mode before ComBat
        if covariates["AGE_AT_SCAN"].isna().any():
            covariates["AGE_AT_SCAN"] = covariates["AGE_AT_SCAN"].fillna(
                covariates["AGE_AT_SCAN"].median()
            )
        if covariates["SEX"].isna().any():
            covariates["SEX"] = covariates["SEX"].fillna(
                covariates["SEX"].mode().iloc[0]
            )

        covariates["SITE"] = covariates["SITE"].astype(str)
        covariates["SEX"] = pd.to_numeric(covariates["SEX"], errors="coerce").fillna(1).astype(int)

        # Drop subject_id — neuroHarmonize only expects covariate columns
        covariates = covariates.drop(columns=["subject_id"])

        return covariates
    
    @staticmethod
    def remove_constant_features(
        features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Remove constant features that would cause harmonization to fail."""
        feature_cols = [c for c in features.columns if c != "subject_id"]
        
        variances = features[feature_cols].var()
        constant_cols = variances[variances == 0].index.tolist()
        kept_cols = [c for c in feature_cols if c not in constant_cols]
        
        if constant_cols:
            logger.warning("Dropping %d constant features", len(constant_cols))
        
        return features[kept_cols], kept_cols, constant_cols
    
    @staticmethod
    def restore_constant_features(
        harmonized: pd.DataFrame,
        original: pd.DataFrame,
        kept_cols: List[str],
        dropped_cols: List[str]
    ) -> pd.DataFrame:
        """Restore constant features that were dropped before harmonization."""
        result = pd.DataFrame(harmonized, columns=kept_cols)
        
        for col in dropped_cols:
            if col in original.columns:
                result[col] = original[col].values
            else:
                result[col] = 0.0
        
        # Restore original column order
        all_cols = [c for c in original.columns if c != "subject_id"]
        result = result[[c for c in all_cols if c in result.columns]]
        
        return result
    
    @staticmethod
    def harmonize_fold(
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        train_covariates: pd.DataFrame,
        val_covariates: pd.DataFrame
    ) -> Tuple[Optional[object], pd.DataFrame, pd.DataFrame]:
        """Harmonize a single fold using neuroHarmonize."""
        try:
            # Add epsilon noise to prevent singular matrix errors with 1e-6 constants
            # ComBat cannot handle zero-variance features that occur when entire ROIs
            # are constant (from the 1e-6 padding in abide_download.py)
            train_features_noisy = train_features.copy()
            val_features_noisy = val_features.copy()
            
            for col in train_features.columns:
                col_std = train_features[col].std()
                if col_std < 1e-5:  # Near-constant column
                    epsilon_noise = np.random.normal(0, 1e-8, size=len(train_features))
                    train_features_noisy[col] = train_features[col] + epsilon_noise
                    
                    val_epsilon = np.random.normal(0, 1e-8, size=len(val_features))
                    val_features_noisy[col] = val_features[col] + val_epsilon
            
            model, train_harmonized = harmonizationLearn(
                train_features_noisy.values,
                train_covariates,
            )
            
            val_harmonized = harmonizationApply(
                val_features_noisy.values,
                val_covariates,
                model,
            )
            
            train_harmonized = pd.DataFrame(
                train_harmonized,
                columns=train_features.columns,
                index=train_features.index,
            )
            
            val_harmonized = pd.DataFrame(
                val_harmonized,
                columns=val_features.columns,
                index=val_features.index,
            )
            
            return model, train_harmonized, val_harmonized
            
        except Exception as exc:
            logger.error("Harmonization failed: %s", exc)
            logger.error("Falling back to unharmonized features for this fold")
            return None, train_features, val_features


class QualityVerifier:
    """Handles quality verification of harmonization results."""
    
    @staticmethod
    def verify(
        original_df: pd.DataFrame,
        harmonized_folds: List[HarmonizationFold]
    ) -> Dict[str, object]:
        """Verify quality of harmonization across all folds."""
        logger.info("=" * 80)
        logger.info("HARMONIZATION QUALITY CHECK")
        logger.info("=" * 80)
        
        all_harmonized = QualityVerifier._combine_folds(harmonized_folds)
        
        harm_cols = [c for c in all_harmonized.columns if c != "subject_id"]
        original_for_var = QualityVerifier._prepare_original_data(original_df, harm_cols)
        orig_cols = [c for c in original_for_var.columns if c != "subject_id"]
        common_cols = [c for c in orig_cols if c in harm_cols]
        
        if not common_cols:
            logger.warning("No overlapping feature columns for variance retention check")
            return QualityVerifier._create_empty_quality_report(harm_cols, all_harmonized)
        
        variance_stats = QualityVerifier._calculate_variance_retention(
            original_for_var, all_harmonized, common_cols
        )
        
        nan_stats = QualityVerifier._check_nan_introduction(
            original_for_var, all_harmonized, orig_cols, harm_cols
        )
        
        QualityVerifier._log_quality_results(variance_stats, nan_stats)
        
        return QualityVerifier._create_quality_report(variance_stats, nan_stats)
    
    @staticmethod
    def _combine_folds(harmonized_folds: List[HarmonizationFold]) -> pd.DataFrame:
        """Combine all folds into a single DataFrame."""
        return pd.concat(
            [
                pd.concat([fold.train, fold.val], ignore_index=True)
                for fold in harmonized_folds
            ],
            ignore_index=True,
        )
    
    @staticmethod
    def _prepare_original_data(
        original_df: pd.DataFrame,
        harm_cols: List[str]
    ) -> pd.DataFrame:
        """Prepare original data for comparison."""
        orig_cols = [c for c in original_df.columns if c != "subject_id"]
        common_cols = [c for c in orig_cols if c in harm_cols]
        
        if not common_cols:
            logger.info("No overlapping columns. Aggregating originals to lobes for comparison")
            try:
                return ROIAggregator.aggregate_to_lobes(original_df)
            except Exception as exc:
                logger.warning("Lobe aggregation failed for quality check: %s", exc)
        
        return original_df
    
    @staticmethod
    def _calculate_variance_retention(
        original_df: pd.DataFrame,
        harmonized_df: pd.DataFrame,
        common_cols: List[str]
    ) -> Dict[str, float]:
        """Calculate variance retention statistics."""
        orig_var = original_df[common_cols].var().mean()
        harm_var = harmonized_df[common_cols].var().mean()
        var_retention = harm_var / orig_var if orig_var != 0 else 0.0
        
        orig_var_series = original_df[common_cols].var()
        harm_var_series = harmonized_df[common_cols].var()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            retention_series = harm_var_series / orig_var_series
        
        retention_series = retention_series.replace([np.inf, -np.inf], np.nan).dropna()
        
        per_feature_stats = QualityVerifier._calculate_per_feature_stats(retention_series)
        
        return {
            "orig_var": orig_var,
            "harm_var": harm_var,
            "var_retention": var_retention,
            **per_feature_stats
        }
    
    @staticmethod
    def _calculate_per_feature_stats(retention_series: pd.Series) -> Dict[str, float]:
        """Calculate per-feature retention statistics."""
        if len(retention_series) == 0:
            return {
                "retention_within": 0,
                "retention_low": 0,
                "retention_high": 0,
                "retention_total": 0,
                "retention_low_pct": np.nan,
                "retention_high_pct": np.nan,
                "retention_within_pct": np.nan,
            }
        
        low_mask = retention_series < VARIANCE_RETENTION_LOW
        high_mask = retention_series > VARIANCE_RETENTION_HIGH
        total = len(retention_series)
        low_count = int(low_mask.sum())
        high_count = int(high_mask.sum())
        within_count = total - low_count - high_count
        
        return {
            "retention_within": within_count,
            "retention_low": low_count,
            "retention_high": high_count,
            "retention_total": total,
            "retention_low_pct": (low_count / total) * 100,
            "retention_high_pct": (high_count / total) * 100,
            "retention_within_pct": (within_count / total) * 100,
        }
    
    @staticmethod
    def _check_nan_introduction(
        original_df: pd.DataFrame,
        harmonized_df: pd.DataFrame,
        orig_cols: List[str],
        harm_cols: List[str]
    ) -> Dict[str, int]:
        """Check if harmonization introduced NaN values."""
        orig_nans = original_df[orig_cols].isna().sum().sum()
        harm_nans = harmonized_df[harm_cols].isna().sum().sum()
        
        return {
            "orig_nans": orig_nans,
            "harm_nans": harm_nans,
            "nans_introduced": harm_nans - orig_nans,
        }
    
    @staticmethod
    def _log_quality_results(
        variance_stats: Dict[str, float],
        nan_stats: Dict[str, int]
    ) -> None:
        """Log quality verification results."""
        if np.isfinite(variance_stats["orig_var"]):
            logger.info("  Original variance: %.4f", variance_stats["orig_var"])
        if np.isfinite(variance_stats["harm_var"]):
            logger.info("  Harmonized variance: %.4f", variance_stats["harm_var"])
        if np.isfinite(variance_stats["var_retention"]):
            logger.info("  Variance retention: %.2f%%", variance_stats["var_retention"] * 100)
        
        if variance_stats["retention_total"] > 0:
            logger.info(
                "  Per-feature retention: %.1f%% within, %.1f%% low, %.1f%% high",
                variance_stats["retention_within_pct"],
                variance_stats["retention_low_pct"],
                variance_stats["retention_high_pct"],
            )
            
            if variance_stats["retention_low_pct"] > VARIANCE_WARNING_THRESHOLD:
                logger.warning("  Many features lost >30%% variance")
            if variance_stats["retention_high_pct"] > VARIANCE_WARNING_THRESHOLD:
                logger.warning("  Many features gained >30%% variance")
        
        if nan_stats["nans_introduced"] > 0:
            logger.warning("  Harmonization introduced NaN values")
        else:
            logger.info("  No NaN values introduced")
    
    @staticmethod
    def _create_quality_report(
        variance_stats: Dict[str, float],
        nan_stats: Dict[str, int]
    ) -> Dict[str, object]:
        """Create quality report dictionary."""
        is_good_quality = (
            variance_stats["retention_total"] > 0
            and variance_stats["retention_low_pct"] <= VARIANCE_WARNING_THRESHOLD
            and variance_stats["retention_high_pct"] <= VARIANCE_WARNING_THRESHOLD
            and nan_stats["nans_introduced"] <= 0
        )
        
        return {
            "variance_retention": variance_stats["var_retention"],
            "nans_introduced": nan_stats["nans_introduced"],
            "per_feature_stats": {
                k: v for k, v in variance_stats.items()
                if k.startswith("retention_")
            },
            "quality": "good" if is_good_quality else "check_warnings",
        }
    
    @staticmethod
    def _create_empty_quality_report(
        harm_cols: List[str],
        all_harmonized: pd.DataFrame
    ) -> Dict[str, object]:
        """Create quality report when no common columns exist."""
        return {
            "variance_retention": np.nan,
            "nans_introduced": 0,
            "per_feature_stats": {
                "retention_within": 0,
                "retention_low": 0,
                "retention_high": 0,
                "retention_total": 0,
                "retention_low_pct": np.nan,
                "retention_high_pct": np.nan,
                "retention_within_pct": np.nan,
            },
            "quality": "check_warnings",
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
    stats = FeatureValidator.validate(features_df)
    features_clean, repair_stats = FeatureRepairer.repair(features_df, impute_nans=True)
    
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
        train_covariates = HarmonizationEngine.prepare_covariates(train_manifest, train_data)
        val_covariates = HarmonizationEngine.prepare_covariates(val_manifest, val_data)
        
        # Remove constant features
        train_features = train_data.drop(columns=["subject_id"])
        val_features = val_data.drop(columns=["subject_id"])
        
        train_features, kept_cols, dropped_cols = HarmonizationEngine.remove_constant_features(
            train_features
        )
        val_features = val_features[kept_cols]
        
        # Harmonize
        model, train_harmonized, val_harmonized = HarmonizationEngine.harmonize_fold(
            train_features,
            val_features,
            train_covariates,
            val_covariates,
        )
        
        # Restore constant features
        train_df = HarmonizationEngine.restore_constant_features(
            train_harmonized,
            train_data,
            kept_cols,
            dropped_cols,
        )
        train_df = pd.concat(
            [train_data[["subject_id"]].reset_index(drop=True), train_df.reset_index(drop=True)],
            axis=1,
        )
        
        val_df = HarmonizationEngine.restore_constant_features(
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
        train_lobes = ROIAggregator.aggregate_to_lobes(train_df)
        val_lobes = ROIAggregator.aggregate_to_lobes(val_df)
        
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
    
    QualityVerifier.verify(features, harmonized_folds)


if __name__ == "__main__":
    main()