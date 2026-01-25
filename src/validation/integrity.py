"""
Combined data integrity validation module.

Provides comprehensive dataset validation and diagnostics:
1. check_dataset_integrity() - Post-download validation (checks PNGs, NPYs, incomplete subjects)
2. check_distribution() - Pre-GNN validation (checks dataset completeness across train/val/test splits)
3. analyze_class_distribution() - Class imbalance analysis with recommendations
4. generate_health_report() - Comprehensive dataset health report with metadata, demographics, features
"""

import os
import sys
import logging
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_PROCESSED, DATA_ROOT, DATA_METADATA, MASTER_MANIFEST, CAUSAL_GRAPHS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TARGET_SLICES = 5
VALID_ROI_RANGE = (164, 170)  # AAL3v1 atlas: 164-170 ROIs depending on template variant
PNG_DIR = DATA_ROOT / "images"
TS_DIR = DATA_PROCESSED


def check_dataset_integrity():
    """
    Post-download integrity check.
    
    Verifies:
    - PNG files are not corrupted
    - NPY time-series files are valid (no NaNs)
    - Subjects have all required data (slices + time series)
    
    Offers interactive cleanup options for corrupted/incomplete data.
    """
    logger.info("Starting post-download integrity check...")
    
    # 1. Check PNG Integrity
    png_files = list(PNG_DIR.glob("*.png"))
    corrupted_pngs = []
    subject_counts = Counter()
    
    if not png_files:
        logger.info("No images found to check.")
    else:
        logger.info(f"Scanning {len(png_files)} images for integrity...")
        for path in png_files:
            try:
                with Image.open(path) as img:
                    img.verify()
                sub_id = path.name.rsplit('_z', 1)[0]
                subject_counts[sub_id] += 1
            except Exception:
                logger.warning(f" [!] Corrupted PNG: {path.name}")
                corrupted_pngs.append(path)

    # 2. Check NPY Integrity
    npy_files = list(TS_DIR.glob("*.npy"))
    corrupted_npys = []
    ts_subjects = set()
    
    logger.info(f"Scanning {len(npy_files)} time-series files...")
    for path in npy_files:
        try:
            data = np.load(path)
            if np.isnan(data).any():
                raise ValueError("NaNs detected")
            num_rois = data.shape[1]
            if not (VALID_ROI_RANGE[0] <= num_rois <= VALID_ROI_RANGE[1]):
                raise ValueError(f"ROI mismatch: Found {num_rois}, expected {VALID_ROI_RANGE[0]}-{VALID_ROI_RANGE[1]}")
            ts_subjects.add(path.name.replace("_ts.npy", ""))
        except Exception as e:
            logger.warning(f" [!] Invalid NPY: {path.name} ({e})")
            corrupted_npys.append(path)

    # 3. Identify Incomplete Subjects
    all_subjects = set(subject_counts.keys()) | ts_subjects
    incomplete_subs = []
    for sub in all_subjects:
        if subject_counts[sub] < TARGET_SLICES or sub not in ts_subjects:
            incomplete_subs.append(sub)

    logger.info("\n" + "="*40)
    logger.info(f"POST-DOWNLOAD INTEGRITY REPORT")
    logger.info("="*40)
    logger.info(f"Corrupted PNGs:      {len(corrupted_pngs)}")
    logger.info(f"Corrupted NPYs:      {len(corrupted_npys)}")
    logger.info(f"Incomplete Subjects: {len(incomplete_subs)} (Missing slices or TS)")
    logger.info("-" * 40)

    if corrupted_pngs or corrupted_npys or incomplete_subs:
        logger.info("OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | [3] Exit")
        choice = input("Select (1/2/3): ")
        if choice == '1':
            for p in corrupted_pngs + corrupted_npys:
                os.remove(p)
                logger.info(f"Deleted: {p}")
        elif choice == '2':
            for sub in incomplete_subs:
                # Remove PNGs
                for p in PNG_DIR.glob(f"{sub}_z*.png"):
                    os.remove(p)
                # Remove NPYs
                for p in TS_DIR.glob(f"{sub}_ts.npy"):
                    os.remove(p)
                for p in TS_DIR.glob(f"{sub}_qc.json"):
                    os.remove(p)
            logger.info("Purge complete.")


def check_distribution():
    """
    Pre-GNN integrity check.
    
    Validates dataset completeness across train/val/test splits:
    - Checks each subject has target number of slices
    - Verifies image/label file pairing for YOLO training
    - Reports distribution of slice counts per split
    """
    logger.info("Starting pre-GNN integrity check...")
    
    PROCESSED_ROOT = str(DATA_PROCESSED)
    
    if not os.path.exists(PROCESSED_ROOT):
        logger.error(f"Path {PROCESSED_ROOT} does not exist. Run split.py first.")
        return

    splits = ['train', 'val', 'test']
    overall_stats = {}

    logger.info(f"Dataset Completeness Report (Target: {TARGET_SLICES} slices/subject)")
    
    for split in splits:
        img_path = os.path.join(PROCESSED_ROOT, split, 'images')
        lbl_path = os.path.join(PROCESSED_ROOT, split, 'labels')
        
        if not os.path.exists(img_path):
            logger.warning(f"[!] Split '{split}' images folder missing.")
            continue

        files = [f for f in os.listdir(img_path) if f.endswith('.png')]
        subject_counts = {}
        
        for f in files:
            # Consistent splitting logic used throughout codebase
            sub_id = f.rsplit('_z', 1)[0]
            subject_counts[sub_id] = subject_counts.get(sub_id, 0) + 1
        
        # Analyze the distribution of slice counts
        dist = Counter(subject_counts.values())
        
        logger.info(f"\nSplit: {split.upper()}")
        logger.info(f"  Total Subjects: {len(subject_counts)}")
        
        for num_slices in sorted(dist.keys()):
            status = "✓" if num_slices == TARGET_SLICES else "X"
            logger.info(f"  {status} {num_slices} slices: {dist[num_slices]} subjects")
        
        # Check for matching labels (Critical for YOLO training)
        if os.path.exists(lbl_path):
            labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]
            if len(labels) != len(files):
                logger.warning(f"  [!] ALERT: Image/Label Mismatch! ({len(files)} images vs {len(labels)} labels)")
            else:
                logger.info(f"  ✓ Image/Label count matches ({len(files)} files)")
    
    logger.info("\nPre-GNN integrity check complete.")


def analyze_class_distribution():
    """
    Comprehensive class distribution analysis.
    
    Analyzes:
    - Overall dataset class balance (ASD vs Control)
    - Per-split distribution (train/val/test)
    - Per-site distribution (top 10 sites)
    - Graph availability and its impact on class balance
    - Provides actionable recommendations based on imbalance severity
    """
    logger.info("="*70)
    logger.info("CLASS DISTRIBUTION ANALYSIS")
    logger.info("="*70)
    
    # Load manifest
    if not MASTER_MANIFEST.exists():
        logger.error(f"❌ Manifest not found: {MASTER_MANIFEST}")
        return
    
    try:
        df = pd.read_csv(MASTER_MANIFEST)
    except FileNotFoundError:
        logger.error(f"File not found: {MASTER_MANIFEST}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {MASTER_MANIFEST}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise
    
    # Overall distribution
    logger.info("\n1. OVERALL DATASET:")
    logger.info("-"*70)
    dx_counts = df['DX_GROUP'].value_counts()
    total = len(df)
    
    control_count = dx_counts.get(2, 0)  # DX_GROUP=2 is Control
    asd_count = dx_counts.get(1, 0)      # DX_GROUP=1 is ASD
    
    logger.info(f"Total subjects: {total}")
    logger.info(f"Control (0): {control_count} ({control_count/total*100:.1f}%)")
    logger.info(f"ASD (1): {asd_count} ({asd_count/total*100:.1f}%)")
    if asd_count > 0:
        logger.info(f"Imbalance ratio: {control_count/asd_count:.2f}:1")
    
    # Per-split distribution
    logger.info("\n2. DISTRIBUTION BY SPLIT:")
    logger.info("-"*70)
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if len(split_df) == 0:
            continue
        
        split_dx = split_df['DX_GROUP'].value_counts()
        split_control = split_dx.get(2, 0)
        split_asd = split_dx.get(1, 0)
        split_total = len(split_df)
        
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Total: {split_total}")
        logger.info(f"  Control: {split_control} ({split_control/split_total*100:.1f}%)")
        logger.info(f"  ASD: {split_asd} ({split_asd/split_total*100:.1f}%)")
        if split_asd > 0:
            logger.info(f"  Ratio: {split_control/split_asd:.2f}:1")
    
    # Per-site distribution (important for multi-site studies)
    logger.info("\n3. DISTRIBUTION BY SITE (Top 10):")
    logger.info("-"*70)
    
    if 'SITE_ID' in df.columns:
        site_stats = []
        for site in df['SITE_ID'].value_counts().head(10).index:
            site_df = df[df['SITE_ID'] == site]
            site_dx = site_df['DX_GROUP'].value_counts()
            site_control = site_dx.get(2, 0)
            site_asd = site_dx.get(1, 0)
            
            site_stats.append({
                'site': site,
                'total': len(site_df),
                'control': site_control,
                'asd': site_asd,
                'ratio': f"{site_control/site_asd:.2f}:1" if site_asd > 0 else "N/A"
            })
        
        for stat in site_stats:
            logger.info(f"{stat['site']:20} | Total: {stat['total']:4} | "
                      f"Control: {stat['control']:4} | ASD: {stat['asd']:4} | "
                      f"Ratio: {stat['ratio']}")
    
    # Check which subjects have graphs
    logger.info("\n4. SUBJECTS WITH CAUSAL GRAPHS:")
    logger.info("-"*70)
    
    if CAUSAL_GRAPHS_DIR.exists():
        try:
            graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
            graph_subjects = [f.stem.replace('_graph', '') for f in graph_files]
            
            # Find subjects with graphs
            df['has_graph'] = df['subject_id'].astype(str).isin(graph_subjects)
            
            graph_df = df[df['has_graph']]
            
            if len(graph_df) > 0:
                graph_dx = graph_df['DX_GROUP'].value_counts()
                graph_control = graph_dx.get(2, 0)
                graph_asd = graph_dx.get(1, 0)
                
                logger.info(f"Subjects with graphs: {len(graph_df)}/{len(df)}")
                logger.info(f"  Control: {graph_control} ({graph_control/len(graph_df)*100:.1f}%)")
                logger.info(f"  ASD: {graph_asd} ({graph_asd/len(graph_df)*100:.1f}%)")
                if graph_asd > 0:
                    logger.info(f"  Ratio: {graph_control/graph_asd:.2f}:1")
                
                # Check if imbalance worsened after filtering
                original_ratio = control_count / asd_count if asd_count > 0 else 0
                graph_ratio = graph_control / graph_asd if graph_asd > 0 else 0
                
                if graph_ratio > original_ratio * 1.1:
                    logger.warning(f"\n  ⚠️  WARNING: Imbalance WORSENED after graph filtering!")
                    logger.warning(f"     Original: {original_ratio:.2f}:1 → Graphs: {graph_ratio:.2f}:1")
            else:
                logger.warning("No subjects with graphs found")
        except Exception as e:
            logger.error(f"Failed to check graph files: {e}")
    
    # Recommendations
    logger.info("\n5. RECOMMENDATIONS:")
    logger.info("-"*70)
    
    if asd_count > 0:
        ratio = control_count / asd_count
        
        if ratio > 3.0:
            logger.error("❌ SEVERE imbalance (ratio > 3:1)")
            logger.error("   → MUST use: Focal Loss + Threshold Optimization")
            logger.error("   → CONSIDER: SMOTE oversampling or undersampling")
        elif ratio > 2.0:
            logger.warning("⚠️  MODERATE imbalance (ratio 2-3:1)")
            logger.warning("   → USE: Focal Loss + Threshold Optimization")
            logger.warning("   → Class weights may help")
        elif ratio > 1.5:
            logger.info("✓ MILD imbalance (ratio 1.5-2:1)")
            logger.info("   → USE: Threshold Optimization")
            logger.info("   → Focal Loss or class weights optional")
        else:
            logger.info("✓ BALANCED dataset (ratio < 1.5:1)")
            logger.info("   → Standard training should work")
    
    logger.info("="*70)


def generate_health_report(pheno_path=None, sample_png=20, sample_ts=10, run_deep_checks=False):
    """
    Generate comprehensive dataset health report.
    
    Args:
        pheno_path: Path to phenotype CSV (default: from config)
        sample_png: Number of PNG files to sample for deep validation
        sample_ts: Number of time series files to sample for deep validation
        run_deep_checks: If True, performs intensive file validation (slower)
    
    Reports:
    - Dataset overview (subjects, slices, completion)
    - Class balance (ASD vs Control)
    - Demographics (age, sex)
    - Site distribution
    - Data completeness (missing metadata/images)
    - Time series and feature file status
    - Graph construction status
    - Optional: Deep integrity checks (PNG/NPY validation)
    """
    pheno_path = pheno_path or (DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv")
    png_dir = DATA_ROOT / "images"
    
    # Load metadata
    if not pheno_path.exists():
        logger.error(f"Error: {pheno_path} not found.")
        return False
    
    df = pd.read_csv(pheno_path)
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    logger.info(f"Loaded metadata: {len(df)} records")
    
    # Load images
    if not png_dir.exists():
        logger.error(f"Error: Image folder {png_dir} not found.")
        return False
    
    downloaded_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]
    completed_subs = set([f.rsplit('_z', 1)[0] for f in downloaded_files])
    logger.info(f"Found {len(downloaded_files)} PNG slices from {len(completed_subs)} subjects")
    
    # Match metadata to images
    current_df = df[df['FILE_ID'].isin(completed_subs)].copy()
    
    if current_df.empty:
        logger.warning("[!] No matching metadata found for downloaded images.")
        return False
    
    logger.info(f"Matched {len(current_df)} subjects to metadata")
    
    # Generate report sections
    logger.info("\n" + "="*40)
    logger.info(f"{'DATASET HEALTH REPORT':^40}")
    logger.info("="*40)
    logger.info(f"Unique Subjects:   {len(completed_subs)}")
    logger.info(f"Total PNG Slices:  {len(downloaded_files)}")
    
    if len(completed_subs) > 0:
        avg_slices = len(downloaded_files) / len(completed_subs)
        logger.info(f"Avg Slices/Sub:    {avg_slices:.1f} (Target: 5.0)")
    
    # Class balance
    logger.info("-" * 40)
    logger.info("CLASS BALANCE")
    stats = current_df['DX_GROUP'].value_counts().to_dict()
    asd = stats.get(1, 0)
    tc = stats.get(2, 0)
    logger.info(f"  Autism (ASD):     {asd}")
    logger.info(f"  Controls (TC):    {tc}")
    if tc > 0:
        ratio = asd / tc
        logger.info(f"  Ratio (ASD/TC):   {ratio:.2f}")
    
    # Demographics
    logger.info("-" * 40)
    logger.info("DEMOGRAPHICS")
    if 'AGE_AT_SCAN' in current_df.columns:
        valid_age = current_df[current_df['AGE_AT_SCAN'] > 0]['AGE_AT_SCAN']
        if not valid_age.empty:
            logger.info(f"  Avg Age:          {valid_age.mean():.1f} years")
    
    if 'SEX' in current_df.columns:
        sex_stats = current_df['SEX'].value_counts().to_dict()
        males = sex_stats.get(1, 0)
        females = sex_stats.get(2, 0)
        logger.info(f"  Sex Ratio (M/F):  {males}/{females}")
    
    # Top sites
    logger.info("-" * 40)
    logger.info("TOP SITES")
    if 'SITE_ID' in current_df.columns:
        site_stats = current_df['SITE_ID'].value_counts().head(5)
        for site, count in site_stats.items():
            logger.info(f"  {str(site):<15}: {count} subjects")
    
    # Data completeness
    logger.info("-" * 40)
    logger.info("DATA COMPLETENESS")
    metadata_ids = set(df['FILE_ID'].unique())
    missing_metadata = completed_subs - metadata_ids
    
    if missing_metadata:
        logger.warning(f"  ⚠️  Subjects with images but NO metadata: {len(missing_metadata)}")
    else:
        logger.info("  ✅ All downloaded subjects have metadata")
    
    missing_images = metadata_ids - completed_subs
    if missing_images:
        logger.warning(f"  ⚠️  Subjects with metadata but NO images: {len(missing_images)}")
    else:
        logger.info("  ✅ All metadata subjects have images")
    
    # Slice distribution
    logger.info("-" * 40)
    logger.info("SLICE DISTRIBUTION")
    slice_counts = {}
    for filename in downloaded_files:
        subject_id = filename.rsplit('_z', 1)[0]
        slice_counts[subject_id] = slice_counts.get(subject_id, 0) + 1
    
    incomplete = {subid: count for subid, count in slice_counts.items() if count != 5}
    complete = sum(1 for count in slice_counts.values() if count == 5)
    logger.info(f"  Complete (5 slices): {complete}/{len(slice_counts)}")
    
    if incomplete:
        logger.warning(f"  ⚠️  Subjects with incomplete slices: {len(incomplete)}")
    else:
        logger.info("  ✅ All subjects have complete slice sets (5/5)")
    
    # Time series files
    logger.info("-" * 40)
    logger.info("TIME SERIES FILES")
    ts_dir = DATA_PROCESSED
    if ts_dir.exists():
        ts_files = list(ts_dir.glob("*_ts.npy"))
        logger.info(f"  Time series files:  {len(ts_files)}")
        ts_subjects = set([f.stem.replace('_ts', '') for f in ts_files])
        missing_ts = completed_subs - ts_subjects
        
        if missing_ts:
            logger.warning(f"  ⚠️  Downloaded subjects missing time series: {len(missing_ts)}")
        else:
            logger.info("  ✅ All downloaded subjects have time series")
    else:
        logger.warning(f"  ⚠️  Time series directory not found: {ts_dir}")
    
    # Feature files
    logger.info("-" * 40)
    logger.info("FEATURE EXTRACTION STATUS")
    feature_files = {
        'Spatial Features': DATA_METADATA / "node_features_3d.csv",
        'Temporal Features': DATA_METADATA / "node_attributes_temporal.csv",
        'Harmonized Features': DATA_METADATA / "node_attributes_harmonized.csv",
    }
    
    for feature_name, feature_path in feature_files.items():
        if feature_path.exists():
            try:
                feat_df = pd.read_csv(feature_path)
                logger.info(f"  ✅ {feature_name:<25}: {len(feat_df)} subjects")
            except Exception as e:
                logger.warning(f"  ⚠️  {feature_name:<25}: ERROR - {str(e)[:50]}")
        else:
            logger.warning(f"  ⚠️  {feature_name:<25}: NOT FOUND")
    
    # Graph files
    logger.info("-" * 40)
    logger.info("GRAPH CONSTRUCTION STATUS")
    graph_dir = DATA_PROCESSED / "causal_graphs"
    if graph_dir.exists():
        graph_files = list(graph_dir.glob("*_graph.pt"))
        logger.info(f"  Graph files:        {len(graph_files)}")
        if len(graph_files) > 0:
            logger.info("  Status:             ✅ Graphs constructed")
        else:
            logger.warning("  Status:             ⚠️  No graphs found")
    else:
        logger.warning(f"  ⚠️  Graph directory not found: {graph_dir}")
    
    # Deep integrity checks (optional, slower)
    if run_deep_checks:
        logger.info("\n" + "="*40)
        logger.info("DEEP INTEGRITY CHECKS")
        logger.info("="*40)
        
        # PNG validation
        logger.info("\nPNG FILE INTEGRITY (sampling)")
        corrupted_pngs = []
        wrong_size = []
        sample_files = random.sample(downloaded_files, min(sample_png, len(downloaded_files)))
        
        for png_file in sample_files:
            png_path = png_dir / png_file
            try:
                img = Image.open(png_path)
                if img.size != (640, 640):
                    wrong_size.append((png_file, img.size))
            except Exception as e:
                corrupted_pngs.append((png_file, str(e)[:40]))
        
        if not corrupted_pngs and not wrong_size:
            logger.info(f"  ✅ PNG files valid (sampled {len(sample_files)} files)")
        else:
            if corrupted_pngs:
                logger.warning(f"  ⚠️  Corrupted PNG files: {len(corrupted_pngs)}")
            if wrong_size:
                logger.warning(f"  ⚠️  Wrong dimensions: {len(wrong_size)}")
        
        # Time series validation
        if ts_dir.exists():
            logger.info("\nTIME SERIES VALIDATION (sampling)")
            ts_files = list(ts_dir.glob("*_ts.npy"))
            invalid_ts = []
            wrong_shape = []
            sample_ts_files = random.sample(ts_files, min(sample_ts, len(ts_files)))
            
            for ts_file in sample_ts_files:
                try:
                    data = np.load(ts_file)
                    if data.ndim != 2 or data.shape[1] != 170:
                        wrong_shape.append((ts_file.name, data.shape))
                    if np.isnan(data).any() or np.isinf(data).any():
                        invalid_ts.append((ts_file.name, "contains NaN/Inf"))
                except Exception as e:
                    invalid_ts.append((ts_file.name, str(e)[:40]))
            
            if not invalid_ts and not wrong_shape:
                logger.info(f"  ✅ Time series files valid (sampled {len(sample_ts_files)} files)")
            else:
                if invalid_ts:
                    logger.warning(f"  ⚠️  Invalid time series files: {len(invalid_ts)}")
                if wrong_shape:
                    logger.warning(f"  ⚠️  Wrong shape: {len(wrong_shape)}")
    
    logger.info("\n" + "="*40)
    logger.info("Health report complete.")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--distribution":
            check_distribution()
        elif sys.argv[1] == "--dataset":
            check_dataset_integrity()
        elif sys.argv[1] == "--class-analysis":
            analyze_class_distribution()
        elif sys.argv[1] == "--health":
            deep_checks = "--deep" in sys.argv
            generate_health_report(run_deep_checks=deep_checks)
        elif sys.argv[1] == "--help":
            print("Usage: python integrity.py [OPTION]")
            print("\nOptions:")
            print("  --dataset          Post-download integrity check")
            print("  --distribution     Pre-GNN dataset distribution check")
            print("  --class-analysis   Comprehensive class imbalance analysis")
            print("  --health           Dataset health report")
            print("  --health --deep    Health report with deep integrity checks")
            print("  (no args)          Run all checks")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Run with --help for usage information")
    else:
        # Run all checks by default
        check_dataset_integrity()
        print("\n")
        check_distribution()
        print("\n")
        analyze_class_distribution()
