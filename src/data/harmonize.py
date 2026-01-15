import pandas as pd
import numpy as np
from neuroCombat import neuroCombat
from pathlib import Path

# --- CONFIG ---
PROJECT_ROOT = Path("./data")
FEATURES_PATH = PROJECT_ROOT / "metadata" / "node_attributes_temporal.csv"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "master_manifest.csv"
OUTPUT_PATH   = PROJECT_ROOT / "metadata" / "node_attributes_harmonized.csv"

def run_harmonization():
    # 1. Load and Align
    if not FEATURES_PATH.exists() or not MANIFEST_PATH.exists():
        print("❌ Missing input files. Ensure signal extraction and manifest generation are done.")
        return

    features = pd.read_csv(FEATURES_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    
    # Merge to ensure every feature row has a corresponding Site and Diagnosis
    data = pd.merge(features, manifest, on='subject_id')
    
    # 2. Robust Cleaning (Phase 2.2 Cleanup)
    # ComBat cannot handle NaNs. We fill missing Age/Sex with medians/modes if necessary
    data['AGE_AT_SCAN'] = data['AGE_AT_SCAN'].fillna(data['AGE_AT_SCAN'].median())
    data['SEX'] = data['SEX'].fillna(data['SEX'].mode()[0])
    
    # 3. Prepare ComBat Inputs
    feature_cols = [c for c in features.columns if c != 'subject_id']
    dat = data[feature_cols].values.astype(float).T # Shape: (Features x Subjects)
    
    # Define Covariates
    # CRITICAL: We include DX_GROUP as a protected covariate so we don't 'harmonize away' the ASD!
    covars = data[['SITE_ID', 'AGE_AT_SCAN', 'SEX', 'DX_GROUP']]
    
    print(f"🧹 Harmonizing {dat.shape[1]} subjects across {data['SITE_ID'].nunique()} sites...")
    print(f"🧬 Protecting biological variance: Age, Sex, and Diagnosis (ASD vs Control)")
    
    # 4. Execute neuroCombat (Phase 4.2 Rigor)
    try:
        combat_results = neuroCombat(
            dat=dat,
            covars=covars,
            batch_col='SITE_ID',
            continuous_cols=['AGE_AT_SCAN'],
            # We don't include DX_GROUP in continuous_cols because it's categorical (1 or 2)
        )
        
        harmonized_data = combat_results['data'].T # Back to (Subjects x Features)
        
        # 5. Verify & Save
        harmonized_df = pd.DataFrame(harmonized_data, columns=feature_cols)
        harmonized_df.insert(0, 'subject_id', data['subject_id'].values)
        
        # Final safety check: No NaNs should remain
        harmonized_df = harmonized_df.fillna(0)
        
        harmonized_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Success! Harmonized features saved to {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"❌ Harmonization failed: {e}")

if __name__ == "__main__":
    run_harmonization()