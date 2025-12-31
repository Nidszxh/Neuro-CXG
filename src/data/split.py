import os
import shutil
import random

SOURCE_IMG = "./data/images"
SOURCE_LBL = "./data/labels"
TARGET_ROOT = "./data/processed"

# Ratios for Train, Validation, and Test
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15 # Remaining 0.15 goes to Test

def run_split():
    # 1. Check if source folders exist
    if not os.path.exists(SOURCE_IMG):
        print(f"Error: Source image directory {SOURCE_IMG} not found.")
        return

    # 2. Identify unique subjects to ensure subject-level splitting
    all_images = [f for f in os.listdir(SOURCE_IMG) if f.endswith('.png')]
    # Use rsplit to handle ABIDE IDs with underscores
    subject_ids = list(set([f.rsplit('_z', 1)[0] for f in all_images]))
    
    if not subject_ids:
        print("No subjects found to split. Check your data/images folder.")
        return

    # 3. Shuffle and Calculate Split Sizes
    random.seed(42) 
    random.shuffle(subject_ids)
    
    total_subs = len(subject_ids)
    n_train = int(total_subs * TRAIN_RATIO)
    n_val = int(total_subs * VAL_RATIO)
    
    splits = {
        'train': subject_ids[:n_train],
        'val': subject_ids[n_train:n_train + n_val],
        'test': subject_ids[n_train + n_val:]
    }

    # 4. Execute the move
    print(f"Total Subjects: {total_subs}")
    for name, subs in splits.items():
        img_dst = os.path.join(TARGET_ROOT, name, 'images')
        lbl_dst = os.path.join(TARGET_ROOT, name, 'labels')
        
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        # Remove old YOLO label caches to prevent training errors
        cache_file = os.path.join(TARGET_ROOT, name, 'labels.cache')
        if os.path.exists(cache_file):
            os.remove(cache_file)

        print(f" Processing {name}: {len(subs)} subjects...")
        
        for sub_id in subs:
            # Match all slices for this specific subject
            # We use sub_id + "_z" to avoid matching 'sub-10' when looking for 'sub-1'
            subject_files = [f for f in all_images if f.startswith(sub_id + "_z")]
            
            for f in subject_files:
                # 1. Move Image
                src_img_path = os.path.join(SOURCE_IMG, f)
                shutil.move(src_img_path, os.path.join(img_dst, f))
                
                # 2. Move Corresponding Label
                lbl_f = f.replace('.png', '.txt')
                src_lbl_path = os.path.join(SOURCE_LBL, lbl_f)
                
                if os.path.exists(src_lbl_path):
                    shutil.move(src_lbl_path, os.path.join(lbl_dst, lbl_f))
                else:
                    # For Causal Graphs, every image MUST have a label (the nodes)
                    print(f" [!] Warning: Missing label for {f}")

    print("\n[SUCCESS] Dataset split into Subject-level Train/Val/Test.")
    print(f"Results located in: {TARGET_ROOT}")

if __name__ == "__main__":
    run_split()