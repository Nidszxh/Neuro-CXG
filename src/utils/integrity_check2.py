import os
from collections import Counter

# Updated to look at the processed root
PROCESSED_ROOT = "./data/processed"
TARGET_SLICES = 5

def check_distribution():
    if not os.path.exists(PROCESSED_ROOT):
        print(f"Error: Path {PROCESSED_ROOT} does not exist. Run split.py first.")
        return

    splits = ['train', 'val', 'test']
    overall_stats = {}

    print(f"Dataset Completeness Report (Target: {TARGET_SLICES} slices/sub):")
    
    for split in splits:
        img_path = os.path.join(PROCESSED_ROOT, split, 'images')
        lbl_path = os.path.join(PROCESSED_ROOT, split, 'labels')
        
        if not os.path.exists(img_path):
            print(f"\n[!] Split '{split}' images folder missing.")
            continue

        files = [f for f in os.listdir(img_path) if f.endswith('.png')]
        subject_counts = {}
        
        for f in files:
            # Consistent splitting logic used in your other scripts
            sub_id = f.rsplit('_z', 1)[0]
            subject_counts[sub_id] = subject_counts.get(sub_id, 0) + 1
        
        # Analyze the distribution of slice counts
        dist = Counter(subject_counts.values())
        
        print(f"\nSplit: {split.upper()}")
        print(f"  Total Subjects: {len(subject_counts)}")
        
        for num_slices in sorted(dist.keys()):
            status = "✓" if num_slices == TARGET_SLICES else "X"
            print(f"  {status} {num_slices} slices: {dist[num_slices]} subjects")
        
        # Check for matching labels (Critical for YOLO training)
        if os.path.exists(lbl_path):
            labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]
            if len(labels) != len(files):
                print(f"  [!] ALERT: Image/Label Mismatch! ({len(files)} images vs {len(labels)} labels)")
            else:
                print(f"  ✓ Image/Label count matches.")


if __name__ == "__main__":
    check_distribution()