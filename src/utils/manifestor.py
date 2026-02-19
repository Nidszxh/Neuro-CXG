import os, logging, sys
import pandas as pd
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_PROCESSED, DATA_FINAL, DATA_METADATA, MASTER_MANIFEST

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG ---
PHENO_PATH = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"

def create_manifest():
    if not PHENO_PATH.exists():
        logger.error(f"[Error]: Phenotypic file not found at {PHENO_PATH}")
        return
        
    # 1. Load and clean phenotypic data (Phase 2.1)
    df = pd.read_csv(PHENO_PATH)
    df.columns = df.columns.str.strip()
    # Strip whitespace from FILE_ID to prevent match failures
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip() 
    
    # 2. Map processed files to their specific splits (Phase 2.2)
    manifest_data = []
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Check time_series folder specifically as it's the core for the GNN
        ts_path = DATA_FINAL / split / 'time_series'
        
        if ts_path.exists():
            # Extract subject IDs from .npy files
            subjects = [f.replace('_ts.npy', '') for f in os.listdir(ts_path) if f.endswith('.npy')]
            for s in subjects:
                manifest_data.append({'subject_id': s, 'split': split})
    
    if not manifest_data:
        logger.error("[Error]: No processed data found. Ensure split.py was successful.")
        return

    manifest_df = pd.DataFrame(manifest_data)
    
    # 3. Select Causal & Clinical variables (Phase 7.1 & 8.4)
    # We include IQ and Handedness as they are major confounders in ASD research
    required_cols = [
        'FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 
        'SITE_ID', 'FIQ', 'HANDEDNESS_CATEGORY'
    ]
    
    # Filter only available columns to avoid merge errors
    available_cols = [c for c in required_cols if c in df.columns]
    
    # 4. Final Merge
    final_df = pd.merge(
        manifest_df, 
        df[available_cols], 
        left_on='subject_id', 
        right_on='FILE_ID',
        how='inner'
    ).drop(columns=['FILE_ID'])
    
    # 5. Data Integrity Check: Ensure no missing targets
    final_df = final_df.dropna(subset=['DX_GROUP'])
    
    DATA_METADATA.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(MASTER_MANIFEST, index=False)
    
    logger.info(f"Manifest successfully synchronized with {len(final_df)} subjects.")
    logger.info(f"Breakdown:\n{final_df.groupby(['split', 'DX_GROUP']).size().unstack(fill_value=0)}")

if __name__ == "__main__":
    create_manifest()