import logging
import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import (
    DATA_PROCESSED, NODE_ATTRIBUTES_TEMPORAL, 
    MASTER_MANIFEST, NODE_ATTRIBUTES_HARMONIZED
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
FEATURES_PATH = NODE_ATTRIBUTES_TEMPORAL
MANIFEST_PATH = MASTER_MANIFEST
OUTPUT_PATH   = NODE_ATTRIBUTES_HARMONIZED


def run_harmonization():
    """
    Harmonize temporal features across sites using neuroCombat.
    
    Protects diagnosis label (DX_GROUP) as a covariate so harmonization
    doesn't remove disease-related signal. This is critical for journal publication.
    """
    logger.info("Starting neuroCombat batch effect harmonization")
    
    # 1. Load and Align
    if not FEATURES_PATH.exists() or not MANIFEST_PATH.exists():
        logger.error(f"Missing input files: features={FEATURES_PATH.exists()}, manifest={MANIFEST_PATH.exists()}")
        logger.error("Ensure signal extraction and manifest generation are done.")
        return

    try:
        features = pd.read_csv(FEATURES_PATH)
    except FileNotFoundError:
        logger.error(f"File not found: {FEATURES_PATH}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {FEATURES_PATH}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load features: {e}")
        raise
    
    try:
        manifest = pd.read_csv(MANIFEST_PATH)
    except FileNotFoundError:
        logger.error(f"File not found: {MANIFEST_PATH}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {MANIFEST_PATH}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise
    
    logger.info(f"Loaded features ({features.shape}) and manifest ({manifest.shape})")
    
    # Merge to ensure every feature row has a corresponding Site and Diagnosis
    try:
        data = pd.merge(features, manifest, on='subject_id')
        logger.info(f"Merged data: {data.shape} subjects with complete records")
    except KeyError as e:
        logger.error(f"Merge failed - missing key column: {e}")
        raise
    
    # 2. Robust Cleaning (Phase 2.2 Cleanup)
    # ComBat cannot handle NaNs. We fill missing Age/Sex with medians/modes if necessary
    try:
        age_filled = data['AGE_AT_SCAN'].fillna(data['AGE_AT_SCAN'].median())
        sex_filled = data['SEX'].fillna(data['SEX'].mode()[0])
        
        if (data['AGE_AT_SCAN'] != age_filled).sum() > 0:
            logger.info(f"Imputed {(data['AGE_AT_SCAN'] != age_filled).sum()} missing age values")
        if (data['SEX'] != sex_filled).sum() > 0:
            logger.info(f"Imputed {(data['SEX'] != sex_filled).sum()} missing sex values")
        
        data['AGE_AT_SCAN'] = age_filled
        data['SEX'] = sex_filled
    except Exception as e:
        logger.error(f"Failed to impute missing values: {e}")
        raise
    
    # 3. Prepare ComBat Inputs
    try:
        feature_cols = [c for c in features.columns if c != 'subject_id']
        dat = data[feature_cols].values.astype(float).T  # Shape: (Features x Subjects)
        
        # Define Covariates
        # CRITICAL: DX_GROUP is protected so harmonization doesn't remove disease signal!
        covars = data[['SITE_ID', 'AGE_AT_SCAN', 'SEX', 'DX_GROUP']]
    except KeyError as e:
        logger.error(f"Missing required column: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to prepare ComBat inputs: {e}")
        raise
    
    logger.info(f"Harmonizing {dat.shape[1]} subjects across {data['SITE_ID'].nunique()} sites")
    logger.info(f"Protecting biological variance: Age, Sex, and Diagnosis (ASD vs Control)")
    
    # 4. Execute neuroCombat (Phase 4.2 Rigor)
    try:
        combat_results = neuroCombat(
            dat=dat,
            covars=covars,
            batch_col='SITE_ID',
            continuous_cols=['AGE_AT_SCAN'],
            # We don't include DX_GROUP in continuous_cols because it's categorical (0 or 1)
        )
        
        harmonized_data = combat_results['data'].T  # Back to (Subjects x Features)
        
        # 5. Verify & Save
        harmonized_df = pd.DataFrame(harmonized_data, columns=feature_cols)
        harmonized_df.insert(0, 'subject_id', data['subject_id'].values)
        
        # Final safety check: No NaNs should remain
        nan_count_before = harmonized_df.isna().sum().sum()
        harmonized_df = harmonized_df.fillna(0)
        nan_count_after = harmonized_df.isna().sum().sum()
        
        if nan_count_before > 0:
            logger.warning(f"Filled {nan_count_before} NaN values with zeros")
        
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        harmonized_df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Successfully saved harmonized features to {OUTPUT_PATH}")
        logger.info(f"Output shape: {harmonized_df.shape}")
        
    except Exception as e:
        logger.error(f"Harmonization failed: {e}", exc_info=True)
        return

if __name__ == "__main__":
    run_harmonization()