import os
from PIL import Image
from collections import Counter

IMAGE_DIR = "./data/images"
TARGET_SLICES = 5  # We expect 5 slices per subject

def check_dataset_integrity(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    corrupted = []
    subject_counts = Counter()
    total = len(files)
    
    if total == 0:
        print("No images found to check.")
        return

    print(f"Scanning {total} images for integrity and completeness...")

    for i, filename in enumerate(files):
        path = os.path.join(directory, filename)
        
        # 1. Check for physical corruption (file integrity)
        try:
            with Image.open(path) as img:
                img.verify()
            
            # 2. Track subject completeness if file is healthy
            # Extract ID from 'sub123_z20.png' -> 'sub123'
            sub_id = filename.rsplit('_z', 1)[0]
            subject_counts[sub_id] += 1

        except (IOError, SyntaxError):
            print(f" [!] Corrupted: {filename}")
            corrupted.append(path)
        
        if i % 500 == 0: # Increased interval for cleaner output
            print(f" Progress: {i}/{total} checked.")

    # 3. Identify Incomplete Subjects
    incomplete_subs = [s for s, count in subject_counts.items() if count < TARGET_SLICES]

    print("\n" + "="*40)
    print(f"FINAL INTEGRITY REPORT")
    print("="*40)
    print(f"Physically Corrupted: {len(corrupted)}")
    print(f"Incomplete Subjects:  {len(incomplete_subs)} (fewer than {TARGET_SLICES} slices)")
    print("-" * 40)

    # Action Logic
    if corrupted or incomplete_subs:
        print("OPTIONS:")
        print(" [1] Delete ONLY corrupted files")
        print(" [2] Delete EVERYTHING related to incomplete subjects (Clean slate for retry)")
        print(" [3] Do nothing")
        
        choice = input("\nSelect an option (1/2/3): ")

        if choice == '1':
            for p in corrupted: os.remove(p)
            print("Deleted corrupted files.")
            
        elif choice == '2':
            # Delete corrupted
            for p in corrupted: 
                if os.path.exists(p): os.remove(p)
            # Delete incomplete subjects so the downloader sees them as 'missing'
            for sub_id in incomplete_subs:
                for f in os.listdir(directory):
                    if f.startswith(sub_id):
                        os.remove(os.path.join(directory, f))
            print(f"Purged data for {len(incomplete_subs)} subjects to allow a clean re-download.")
            
    else:
        print("[CLEAN] All images are healthy and all subjects are complete!")

if __name__ == "__main__":
    if os.path.exists(IMAGE_DIR):
        check_dataset_integrity(IMAGE_DIR)
    else:
        print(f"Error: {IMAGE_DIR} not found.")