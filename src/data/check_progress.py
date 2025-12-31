import os
import pandas as pd

PHENO_PATH = "./data/processed/Phenotypic_V1_0b_preprocessed1.csv"
PNG_OUTPUT = "./data/images" 

def check_health():
    # 1. Load Metadata
    if not os.path.exists(PHENO_PATH):
        print(f"Error: {PHENO_PATH} not found.")
        return
    
    # Load and clean metadata to ensure FILE_ID matches the filenames
    df = pd.read_csv(PHENO_PATH)
    # Ensure FILE_ID is treated as a string for matching
    df['FILE_ID'] = df['FILE_ID'].astype(str)
    
    # 2. Get list of subjects currently downloaded
    if not os.path.exists(PNG_OUTPUT):
        print(f"Error: Image folder {PNG_OUTPUT} not found.")
        return

    downloaded_files = [f for f in os.listdir(PNG_OUTPUT) if f.endswith('.png')]
    
    # CRITICAL FIX: Extract subject ID correctly. 
    # We split from the right side in case the FILE_ID itself contains underscores.
    completed_subs = set([f.rsplit('_z', 1)[0] for f in downloaded_files])
    
    print("\n" + "="*40)
    print(f"{'DATASET HEALTH REPORT':^40}")
    print("="*40)
    
    print(f"Unique Subjects:   {len(completed_subs)}")
    print(f"Total PNG Slices:  {len(downloaded_files)}")
    
    if len(completed_subs) > 0:
        print(f"Avg Slices/Sub:    {len(downloaded_files)/len(completed_subs):.1f} (Target: 5.0)")
    
    # 3. Analyze Balance
    # We filter the dataframe to only include subjects we actually have images for
    current_df = df[df['FILE_ID'].isin(completed_subs)].copy()
    
    if current_df.empty:
        print("\n[!] Warning: No matching metadata found for downloaded images.")
        print("Check if FILE_ID in CSV matches the prefix of your PNG files.")
        return

    # DX_GROUP: 1 = ASD, 2 = Control
    stats = current_df['DX_GROUP'].value_counts().to_dict()
    asd = stats.get(1, 0)
    tc = stats.get(2, 0)
    
    print("-" * 40)
    print(f"CLASS BALANCE")
    print(f"  Autism (ASD):     {asd}")
    print(f"  Controls (TC):    {tc}")
    
    if tc > 0:
        print(f"  Ratio (ASD/TC):   {asd/tc:.2f}")

    # 4. Demographic Check
    print("-" * 40)
    print(f"DEMOGRAPHICS")
    if 'AGE_AT_SCAN' in current_df.columns:
        valid_age = current_df[current_df['AGE_AT_SCAN'] > 0]['AGE_AT_SCAN']
        if not valid_age.empty:
            print(f"  Avg Age:          {valid_age.mean():.1f} years")
    
    if 'SEX' in current_df.columns:
        # 1 = Male, 2 = Female
        sex_stats = current_df['SEX'].value_counts().to_dict()
        males = sex_stats.get(1, 0)
        females = sex_stats.get(2, 0)
        print(f"  Sex Ratio (M/F):  {males}/{females}")

    # 5. Site Distribution (Crucial for ABIDE to check for site-bias)
    print("-" * 40)
    print("TOP SITES")
    if 'SITE_ID' in current_df.columns:
        site_stats = current_df['SITE_ID'].value_counts().head(5)
        for site, count in site_stats.items():
            print(f"  {str(site):<15}: {count} subjects")
    
    print("="*40 + "\n")

if __name__ == "__main__":
    check_health()