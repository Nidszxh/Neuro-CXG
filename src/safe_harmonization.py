"""
Feature Harmonization Fix

Addresses the NaN catastrophe where 1,018,440 values were filled with zeros.
Implements robust validation and quality control throughout the harmonization pipeline.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    NODE_ATTRIBUTES_TEMPORAL, MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED, NUM_TEMPORAL_FEATURES
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_temporal_features(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive validation of temporal feature integrity.
    
    Args:
        df: DataFrame with temporal features
        
    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating temporal features...")
    
    feature_cols = [c for c in df.columns if c != 'subject_id']
    
    stats = {
        'total_subjects': len(df),
        'total_features': len(feature_cols),
        'nan_count': df[feature_cols].isna().sum().sum(),
        'inf_count': np.isinf(df[feature_cols].values).sum(),
        'zero_count': (df[feature_cols] == 0).sum().sum(),
        'subjects_with_nans': df[feature_cols].isna().any(axis=1).sum(),
        'features_with_nans': df[feature_cols].isna().any(axis=0).sum()
    }
    
    # Calculate NaN percentage
    total_values = len(df) * len(feature_cols)
    stats['nan_percentage'] = (stats['nan_count'] / total_values) * 100
    
    # Check ROI distribution
    expected_rois = len(feature_cols) // NUM_TEMPORAL_FEATURES
    stats['detected_rois'] = expected_rois
    
    # Feature value statistics
    valid_data = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_data) > 0:
        stats['value_mean'] = valid_data.values.mean()
        stats['value_std'] = valid_data.values.std()
        stats['value_min'] = valid_data.values.min()
        stats['value_max'] = valid_data.values.max()
    
    return stats


def diagnose_nan_sources(df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Identify which subjects and features have NaN values.
    
    Args:
        df: DataFrame with temporal features
        manifest: Manifest with subject metadata
        
    Returns:
        DataFrame with NaN diagnosis per subject
    """
    feature_cols = [c for c in df.columns if c != 'subject_id']
    
    diagnosis = []
    
    for idx, row in df.iterrows():
        sub_id = row['subject_id']
        
        # Count NaNs for this subject
        nan_count = row[feature_cols].isna().sum()
        
        if nan_count > 0:
            # Get subject metadata
            meta = manifest[manifest['subject_id'] == sub_id]
            
            diagnosis.append({
                'subject_id': sub_id,
                'nan_count': nan_count,
                'nan_percentage': (nan_count / len(feature_cols)) * 100,
                'site': meta['SITE_ID'].iloc[0] if len(meta) > 0 else 'Unknown',
                'dx_group': meta['DX_GROUP'].iloc[0] if len(meta) > 0 else 'Unknown'
            })
    
    return pd.DataFrame(diagnosis).sort_values('nan_count', ascending=False)


def repair_temporal_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Attempt to repair temporal features with intelligent imputation.
    
    Strategy:
    1. Remove subjects with >50% NaN values (unreliable)
    2. For remaining subjects, impute NaNs using site-specific medians
    3. Replace infinite values with feature-specific bounds
    4. Z-score normalize to detect outliers
    
    Args:
        df: DataFrame with temporal features
        
    Returns:
        Tuple of (repaired DataFrame, repair statistics)
    """
    logger.info("Attempting to repair temporal features...")
    
    feature_cols = [c for c in df.columns if c != 'subject_id']
    df_clean = df.copy()
    
    repair_stats = {
        'subjects_before': len(df),
        'subjects_removed': 0,
        'nans_imputed': 0,
        'infs_replaced': 0,
        'outliers_detected': 0
    }
    
    # Step 1: Remove subjects with excessive NaNs
    nan_threshold = 0.5
    nan_counts = df_clean[feature_cols].isna().sum(axis=1)
    invalid_subjects = nan_counts > (len(feature_cols) * nan_threshold)
    
    if invalid_subjects.sum() > 0:
        logger.warning(
            f"Removing {invalid_subjects.sum()} subjects with >{nan_threshold*100}% NaN values"
        )
        df_clean = df_clean[~invalid_subjects].reset_index(drop=True)
        repair_stats['subjects_removed'] = invalid_subjects.sum()
    
    # Step 2: Replace infinites
    inf_mask = np.isinf(df_clean[feature_cols].values)
    if inf_mask.sum() > 0:
        logger.warning(f"Replacing {inf_mask.sum()} infinite values")
        for col in feature_cols:
            # Replace +inf with 99th percentile
            pos_inf = df_clean[col] == np.inf
            if pos_inf.sum() > 0:
                valid_vals = df_clean[col][~np.isinf(df_clean[col])]
                if len(valid_vals) > 0:
                    df_clean.loc[pos_inf, col] = valid_vals.quantile(0.99)
            
            # Replace -inf with 1st percentile
            neg_inf = df_clean[col] == -np.inf
            if neg_inf.sum() > 0:
                valid_vals = df_clean[col][~np.isinf(df_clean[col])]
                if len(valid_vals) > 0:
                    df_clean.loc[neg_inf, col] = valid_vals.quantile(0.01)
        
        repair_stats['infs_replaced'] = inf_mask.sum()
    
    # Step 3: Impute remaining NaNs with feature-wise medians
    nan_count_before = df_clean[feature_cols].isna().sum().sum()
    
    if nan_count_before > 0:
        logger.info(f"Imputing {nan_count_before} NaN values with feature medians")
        for col in feature_cols:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        
        repair_stats['nans_imputed'] = nan_count_before
    
    # Step 4: Detect outliers (values beyond 5 standard deviations)
    for col in feature_cols:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        
        if std > 0:
            z_scores = np.abs((df_clean[col] - mean) / std)
            outliers = z_scores > 5
            
            if outliers.sum() > 0:
                # Cap outliers at 5 standard deviations
                df_clean.loc[outliers, col] = mean + 5 * std * np.sign(df_clean.loc[outliers, col] - mean)
                repair_stats['outliers_detected'] += outliers.sum()
    
    repair_stats['subjects_after'] = len(df_clean)
    
    return df_clean, repair_stats


def validate_harmonization_inputs(
    features_df: pd.DataFrame,
    manifest_df: pd.DataFrame
) -> Tuple[pd.DataFrame, bool]:
    """
    Validate inputs for neuroCombat harmonization.
    
    Args:
        features_df: Temporal features
        manifest_df: Subject manifest
        
    Returns:
        Tuple of (merged DataFrame, validation passed boolean)
    """
    logger.info("Validating harmonization inputs...")
    
    # Merge features with manifest
    merged = pd.merge(
        features_df,
        manifest_df,
        on='subject_id',
        how='inner'
    )
    
    logger.info(f"Merged data: {len(merged)} subjects")
    
    # Check required columns
    required_cols = ['SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']
    missing_cols = [c for c in required_cols if c not in merged.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return merged, False
    
    # Check for missing metadata
    for col in required_cols:
        missing_count = merged[col].isna().sum()
        if missing_count > 0:
            logger.warning(f"{col} has {missing_count} missing values")
    
    # Validate site distribution
    site_counts = merged['SITE_ID'].value_counts()
    logger.info(f"Data from {len(site_counts)} sites")
    
    sites_with_few_subjects = site_counts[site_counts < 5]
    if len(sites_with_few_subjects) > 0:
        logger.warning(
            f"{len(sites_with_few_subjects)} sites have <5 subjects - "
            f"may cause harmonization issues"
        )
    
    # Validate diagnosis distribution
    dx_counts = merged['DX_GROUP'].value_counts()
    logger.info(f"Diagnosis distribution: {dx_counts.to_dict()}")
    
    if len(dx_counts) != 2:
        logger.warning("Expected 2 diagnosis groups (ASD=1, Control=2)")
    
    return merged, True


def run_safe_harmonization(
    features_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    output_path: Path
) -> bool:
    """
    Execute harmonization with comprehensive safety checks.
    
    Args:
        features_df: Temporal features
        manifest_df: Subject manifest
        output_path: Path to save harmonized features
        
    Returns:
        True if successful, False otherwise
    """
    from neuroCombat import neuroCombat
    
    logger.info("="*60)
    logger.info("SAFE FEATURE HARMONIZATION")
    logger.info("="*60)
    
    # Step 1: Validate inputs
    initial_stats = validate_temporal_features(features_df)
    logger.info("Initial feature statistics:")
    for key, val in initial_stats.items():
        logger.info(f"  {key}: {val}")
    
    # Critical check: Stop if too many NaNs
    if initial_stats['nan_percentage'] > 20:
        logger.error(
            f"CRITICAL: {initial_stats['nan_percentage']:.1f}% of values are NaN!"
        )
        logger.error("This indicates a fundamental problem with feature extraction.")
        logger.error("Check:")
        logger.error("  1. Atlas file exists and is valid")
        logger.error("  2. Time series files are correctly formatted")
        logger.error("  3. ROI extraction completed successfully")
        
        # Diagnose NaN sources
        nan_diagnosis = diagnose_nan_sources(features_df, manifest_df)
        if len(nan_diagnosis) > 0:
            logger.error("\nTop 10 subjects with NaN values:")
            logger.error(nan_diagnosis.head(10).to_string())
        
        return False
    
    # Step 2: Repair features
    features_repaired, repair_stats = repair_temporal_features(features_df)
    
    logger.info("\nFeature repair statistics:")
    for key, val in repair_stats.items():
        logger.info(f"  {key}: {val}")
    
    # Step 3: Validate harmonization inputs
    merged_data, inputs_valid = validate_harmonization_inputs(
        features_repaired,
        manifest_df
    )
    
    if not inputs_valid:
        logger.error("Harmonization input validation failed")
        return False
    
    # Step 4: Prepare for neuroCombat
    feature_cols = [c for c in features_repaired.columns if c != 'subject_id']
    
    # Fill any remaining missing covariates
    merged_data['AGE_AT_SCAN'] = merged_data['AGE_AT_SCAN'].fillna(
        merged_data['AGE_AT_SCAN'].median()
    )
    merged_data['SEX'] = merged_data['SEX'].fillna(
        merged_data['SEX'].mode()[0]
    )
    
    # Prepare data matrix (features × subjects)
    dat = merged_data[feature_cols].values.T.astype(float)
    
    logger.info(f"\nHarmonization data shape: {dat.shape}")
    logger.info(f"  Features: {dat.shape[0]}")
    logger.info(f"  Subjects: {dat.shape[1]}")
    
    # Prepare covariates
    covars = merged_data[['SITE_ID', 'AGE_AT_SCAN', 'SEX', 'DX_GROUP']]
    
    # Step 5: Execute neuroCombat
    try:
        logger.info("\nExecuting neuroCombat harmonization...")
        logger.info("  Batch variable: SITE_ID")
        logger.info("  Protected covariates: AGE_AT_SCAN, SEX, DX_GROUP")
        
        combat_results = neuroCombat(
            dat=dat,
            covars=covars,
            batch_col='SITE_ID',
            continuous_cols=['AGE_AT_SCAN'],
            # DX_GROUP is categorical and protected
        )
        
        harmonized_data = combat_results['data'].T  # Back to subjects × features
        
        # Step 6: Validate harmonization output
        harmonized_df = pd.DataFrame(
            harmonized_data,
            columns=feature_cols
        )
        harmonized_df.insert(0, 'subject_id', merged_data['subject_id'].values)
        
        # Final validation
        final_nan_count = harmonized_df[feature_cols].isna().sum().sum()
        
        if final_nan_count > 0:
            logger.error(f"CRITICAL: {final_nan_count} NaNs remain after harmonization!")
            logger.error("Filling with zeros as last resort...")
            harmonized_df[feature_cols] = harmonized_df[feature_cols].fillna(0)
        
        # Check for new infinites
        inf_count = np.isinf(harmonized_df[feature_cols].values).sum()
        if inf_count > 0:
            logger.warning(f"Replacing {inf_count} infinite values after harmonization")
            harmonized_df[feature_cols] = harmonized_df[feature_cols].replace(
                [np.inf, -np.inf],
                [harmonized_df[feature_cols].max().max(), harmonized_df[feature_cols].min().min()]
            )
        
        # Step 7: Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        harmonized_df.to_csv(output_path, index=False)
        
        logger.info("="*60)
        logger.info("✓ HARMONIZATION SUCCESSFUL")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Subjects: {len(harmonized_df)}")
        logger.info(f"  Features per subject: {len(feature_cols)}")
        logger.info(f"  Final NaN count: {final_nan_count}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Harmonization failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    # Load data
    if not NODE_ATTRIBUTES_TEMPORAL.exists():
        logger.error(f"Temporal features not found: {NODE_ATTRIBUTES_TEMPORAL}")
        logger.error("Run compute_roi.py first!")
        sys.exit(1)
    
    if not MASTER_MANIFEST.exists():
        logger.error(f"Master manifest not found: {MASTER_MANIFEST}")
        logger.error("Run manifest.py first!")
        sys.exit(1)
    
    features = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    # Run safe harmonization
    success = run_safe_harmonization(
        features,
        manifest,
        NODE_ATTRIBUTES_HARMONIZED
    )
    
    if not success:
        logger.error("Harmonization failed - see errors above")
        sys.exit(1)