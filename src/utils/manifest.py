import os
import pandas as pd

# --- CONFIG ---
PHENO_PATH = "./data/processed/Phenotypic_V1_0b_preprocessed1.csv"
# Updated to match the TARGET_ROOT in your split.py
DATASET_ROOT = "./data/processed" 
OUTPUT_PATH = "./data/metadata/manifest_v1.csv"

def create_manifest():
    # 1. Load labels and clean column names
    if not os.path.exists(PHENO_PATH):
        print(f"Error: Phenotypic file not found at {PHENO_PATH}")
        return
        
    df = pd.read_csv(PHENO_PATH)
    df.columns = df.columns.str.strip() 
    
    # Ensure FILE_ID is treated as a string for merging
    id_col = 'FILE_ID'
    if id_col not in df.columns:
        print(f"Error: {id_col} column missing in CSV.")
        return
    df[id_col] = df[id_col].astype(str)

    # 2. Map subjects in each split
    manifest_data = []
    splits = ['train', 'val', 'test']
    
    print("Indexing processed splits...")
    for split in splits:
        img_path = os.path.join(DATASET_ROOT, split, 'images')
        
        if os.path.exists(img_path):
            # Get unique subject IDs from the filenames
            files = [f for f in os.listdir(img_path) if f.endswith('.png')]
            subs = set([f.rsplit('_z', 1)[0] for f in files])
            
            for s in subs:
                manifest_data.append({'ID_MATCH': str(s), 'Split': split})
    
    if not manifest_data:
        print("Error: No images found in data/processed/. Did you run split.py?")
        return

    manifest_df = pd.DataFrame(manifest_data)
    
    # 3. Merge with clinical/causal metadata
    # We include DX_GROUP (Target), AGE, SEX, and SITE_ID (Causal Confounders)
    final_df = pd.merge(
        manifest_df, 
        df[[id_col, 'DX_GROUP', 'AGE_AT_SCAN', 'SEX', 'SITE_ID']], 
        left_on='ID_MATCH', 
        right_on=id_col
    ).drop(columns=[id_col]) # Remove redundant column after merge
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # 4. Save the manifest
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n" + "="*40)
    print(f"{'MANIFEST GENERATED':^40}")
    print("="*40)
    print(f"Subjects Tracked: {len(final_df)}")
    print(f"Split Breakdown:\n{final_df['Split'].value_counts()}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("="*40)

if __name__ == "__main__":
    create_manifest()