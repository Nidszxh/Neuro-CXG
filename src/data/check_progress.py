import os, logging, sys
import pandas as pd
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import DATA_PROCESSED, DATA_ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PHENO_PATH = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"
PNG_OUTPUT = DATA_ROOT / "images"

def check_health():
    # 1. Load Metadata
    if not PHENO_PATH.exists():
        logger.error(f"Error: {PHENO_PATH} not found.")
        return
    
    # Load and clean metadata to ensure FILE_ID matches the filenames
    df = pd.read_csv(PHENO_PATH)
    # Ensure FILE_ID is treated as a string for matching
    df['FILE_ID'] = df['FILE_ID'].astype(str)
    
    # 2. Get list of subjects currently downloaded
    if not PNG_OUTPUT.exists():
        logger.error(f"Error: Image folder {PNG_OUTPUT} not found.")
        return

    downloaded_files = [f for f in os.listdir(PNG_OUTPUT) if f.endswith('.png')]
    
    # CRITICAL FIX: Extract subject ID correctly. 
    # We split from the right side in case the FILE_ID itself contains underscores.
    completed_subs = set([f.rsplit('_z', 1)[0] for f in downloaded_files])
    
    logger.info("\n" + "="*40)
    logger.info(f"{'DATASET HEALTH REPORT':^40}")
    logger.info("="*40)
    
    logger.info(f"Unique Subjects:   {len(completed_subs)}")
    logger.info(f"Total PNG Slices:  {len(downloaded_files)}")
    
    if len(completed_subs) > 0:
        logger.info(f"Avg Slices/Sub:    {len(downloaded_files)/len(completed_subs):.1f} (Target: 5.0)")
    
    # 3. Analyze Balance
    # We filter the dataframe to only include subjects we actually have images for
    current_df = df[df['FILE_ID'].isin(completed_subs)].copy()
    
    if current_df.empty:
        logger.warning("\n[!] Warning: No matching metadata found for downloaded images.")
        logger.warning("Check if FILE_ID in CSV matches the prefix of your PNG files.")
        return

    # DX_GROUP: 1 = ASD, 2 = Control
    stats = current_df['DX_GROUP'].value_counts().to_dict()
    asd = stats.get(1, 0)
    tc = stats.get(2, 0)
    
    logger.info("-" * 40)
    logger.info(f"CLASS BALANCE")
    logger.info(f"  Autism (ASD):     {asd}")
    logger.info(f"  Controls (TC):    {tc}")
    
    if tc > 0:
        logger.info(f"  Ratio (ASD/TC):   {asd/tc:.2f}")

    # 4. Demographic Check
    logger.info("-" * 40)
    logger.info(f"DEMOGRAPHICS")
    if 'AGE_AT_SCAN' in current_df.columns:
        valid_age = current_df[current_df['AGE_AT_SCAN'] > 0]['AGE_AT_SCAN']
        if not valid_age.empty:
            logger.info(f"  Avg Age:          {valid_age.mean():.1f} years")
    
    if 'SEX' in current_df.columns:
        # 1 = Male, 2 = Female
        sex_stats = current_df['SEX'].value_counts().to_dict()
        males = sex_stats.get(1, 0)
        females = sex_stats.get(2, 0)
        logger.info(f"  Sex Ratio (M/F):  {males}/{females}")

    # 5. Site Distribution (Crucial for ABIDE to check for site-bias)
    logger.info("-" * 40)
    logger.info("TOP SITES")
    if 'SITE_ID' in current_df.columns:
        site_stats = current_df['SITE_ID'].value_counts().head(5)
        for site, count in site_stats.items():
            logger.info(f"  {str(site):<15}: {count} subjects")
    
    logger.info("="*40 + "\n")

if __name__ == "__main__":
    check_health()