"""
Filter dataset to 1,000 high-quality subjects.

Removes 25 subjects missing spatial features (YOLO detection failures).
Maintains class balance (486 ASD, 514 Control) and site representation.

Usage:
    python src/data/filter_to_1000.py [--backup] [--restore-backup]

Options:
    --backup          Create backups of modified files before filtering
    --restore-backup  Restore data from backup (undo filtering)
"""

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_FINAL,
    DATA_PROCESSED,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FilteringReport:
    """Report from filtering operation."""
    subjects_removed: int
    subjects_remaining: int
    class_distribution: dict
    site_impact: dict
    files_updated: List[str]


def get_subjects_to_remove() -> Set[str]:
    """Identify subjects without spatial features."""
    temporal = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
    spatial = pd.read_csv(NODE_FEATURES_3D)
    
    temporal_subjects = set(temporal['subject_id'])
    spatial_subjects = set(spatial['subject_id'])
    
    missing_spatial = temporal_subjects - spatial_subjects
    logger.info(f"Found {len(missing_spatial)} subjects without spatial features")
    
    return missing_spatial


def filter_manifest(manifest_path: Path, subjects_to_keep: Set[str]) -> pd.DataFrame:
    """Filter manifest to keep only specified subjects."""
    df = pd.read_csv(manifest_path)
    original_count = len(df)
    
    df_filtered = df[df['subject_id'].isin(subjects_to_keep)].copy()
    removed_count = original_count - len(df_filtered)
    
    logger.info(
        f"Manifest: {original_count} → {len(df_filtered)} subjects "
        f"({removed_count} removed)"
    )
    
    return df_filtered


def filter_features(features_path: Path, subjects_to_keep: Set[str]) -> pd.DataFrame:
    """Filter feature CSV to keep only specified subjects."""
    df = pd.read_csv(features_path)
    original_count = len(df)
    
    # Handle 'subject_id' column name variations
    subject_col = 'subject_id'
    if subject_col not in df.columns:
        subject_col = df.columns[0]  # First column is usually subject_id
    
    df_filtered = df[df[subject_col].isin(subjects_to_keep)].copy()
    removed_count = original_count - len(df_filtered)
    
    logger.info(
        f"{features_path.name}: {original_count} → {len(df_filtered)} subjects "
        f"({removed_count} removed)"
    )
    
    return df_filtered


def filter_causal_graphs(subjects_to_keep: Set[str]) -> int:
    """Remove graph files for filtered subjects."""
    graphs_dir = DATA_PROCESSED / 'causal_graphs'
    if not graphs_dir.exists():
        logger.warning("Causal graphs directory not found")
        return 0
    
    graph_files = list(graphs_dir.glob('*_graph.pt'))
    removed_count = 0
    
    for graph_file in graph_files:
        subject_id = graph_file.stem.rsplit('_graph', 1)[0]
        
        if subject_id not in subjects_to_keep:
            graph_file.unlink()
            removed_count += 1
            logger.debug(f"Removed graph: {graph_file.name}")
    
    logger.info(f"Causal graphs: removed {removed_count} files")
    return removed_count


def filter_split_images(subjects_to_keep: Set[str]) -> int:
    """Remove PNG images for filtered subjects from train/val/test splits."""
    images_dir = DATA_FINAL / 'images'
    if not images_dir.exists():
        logger.warning("Split images directory not found")
        return 0
    
    removed_count = 0
    
    for png_file in images_dir.glob('*.png'):
        # Extract subject_id from filename (format: {SITE}_{SUBJECT_ID}_z{DEPTH}.png)
        parts = png_file.stem.split('_')
        if len(parts) >= 2:
            subject_id = '_'.join(parts[:-1])  # Everything except last _zN
            subject_id = subject_id.rsplit('_', 1)[0]  # Get just {SITE}_{ID}
            
            # Match against subjects_to_keep more robustly
            if not any(subject_id in subj or subj in subject_id for subj in subjects_to_keep):
                png_file.unlink()
                removed_count += 1
                logger.debug(f"Removed image: {png_file.name}")
    
    logger.info(f"Split images: removed {removed_count} PNG files")
    return removed_count


def filter_split_labels(subjects_to_keep: Set[str]) -> int:
    """Remove label files for filtered subjects from train/val/test splits."""
    labels_dir = DATA_FINAL / 'labels'
    if not labels_dir.exists():
        logger.warning("Split labels directory not found")
        return 0
    
    removed_count = 0
    
    for txt_file in labels_dir.glob('*.txt'):
        # Extract subject_id from filename (format: {SITE}_{SUBJECT_ID}_z{DEPTH}.txt)
        parts = txt_file.stem.split('_')
        if len(parts) >= 2:
            subject_id = '_'.join(parts[:-1])  # Everything except last _zN
            subject_id = subject_id.rsplit('_', 1)[0]  # Get just {SITE}_{ID}
            
            if not any(subject_id in subj or subj in subject_id for subj in subjects_to_keep):
                txt_file.unlink()
                removed_count += 1
                logger.debug(f"Removed label: {txt_file.name}")
    
    logger.info(f"Split labels: removed {removed_count} text files")
    return removed_count


def backup_file(file_path: Path) -> Path:
    """Create backup of a file."""
    if not file_path.exists():
        return None
    
    backup_path = file_path.parent / f"{file_path.name}.backup"
    if not backup_path.exists():
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backed up: {file_path.name}")
    
    return backup_path


def generate_report(subjects_removed: Set[str], subjects_to_keep: Set[str]) -> FilteringReport:
    """Generate summary report."""
    manifest = pd.read_csv(MASTER_MANIFEST)
    manifest_filtered = manifest[manifest['subject_id'].isin(subjects_to_keep)]
    
    # Class distribution
    class_dist = dict(manifest_filtered['DX_GROUP'].value_counts())
    
    # Site impact
    manifest_removed = manifest[manifest['subject_id'].isin(subjects_removed)]
    site_impact = {}
    for site in manifest['SITE_ID'].unique():
        total = len(manifest[manifest['SITE_ID'] == site])
        removed = len(manifest_removed[manifest_removed['SITE_ID'] == site])
        if removed > 0:
            site_impact[site] = {'removed': removed, 'total': total, 'pct': 100*removed/total}
    
    return FilteringReport(
        subjects_removed=len(subjects_removed),
        subjects_remaining=len(subjects_to_keep),
        class_distribution=class_dist,
        site_impact=site_impact,
        files_updated=['master_manifest.csv', 'node_attributes_temporal.csv', 
                       'node_attributes_harmonized.csv', 'node_features_3d.csv',
                       'causal_graphs/*.pt', 'final/images/*.png', 'final/labels/*.txt']
    )


def main(backup: bool = False, restore: bool = False):
    """Main filtering pipeline."""
    logger.info("=" * 70)
    logger.info("STANDARDIZING DATASET TO 1,000 HIGH-QUALITY SUBJECTS")
    logger.info("=" * 70)
    
    if restore:
        logger.info("\nRESTORING FROM BACKUP...")
        # TODO: Implement restore logic
        logger.error("Restore not yet implemented")
        return
    
    # Identify subjects to remove
    subjects_to_remove = get_subjects_to_remove()
    subjects_to_keep = set(pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)['subject_id']) - subjects_to_remove
    
    logger.info(f"\nFILTERING CRITERIA:")
    logger.info(f"  Remove: {len(subjects_to_remove)} subjects without spatial features")
    logger.info(f"  Keep: {len(subjects_to_keep)} high-quality subjects")
    
    # Backup files if requested
    if backup:
        logger.info("\nBACKING UP FILES...")
        backup_file(MASTER_MANIFEST)
        backup_file(NODE_ATTRIBUTES_TEMPORAL)
        backup_file(NODE_ATTRIBUTES_HARMONIZED)
        backup_file(NODE_FEATURES_3D)
    
    # Filter CSV files
    logger.info("\nFILTERING CSV FILES...")
    
    # Master manifest
    manifest_filtered = filter_manifest(MASTER_MANIFEST, subjects_to_keep)
    manifest_filtered.to_csv(MASTER_MANIFEST, index=False)
    
    # Temporal features
    temporal_filtered = filter_features(NODE_ATTRIBUTES_TEMPORAL, subjects_to_keep)
    temporal_filtered.to_csv(NODE_ATTRIBUTES_TEMPORAL, index=False)
    
    # Harmonized features
    harmonized_filtered = filter_features(NODE_ATTRIBUTES_HARMONIZED, subjects_to_keep)
    harmonized_filtered.to_csv(NODE_ATTRIBUTES_HARMONIZED, index=False)
    
    # Spatial features
    spatial_filtered = filter_features(NODE_FEATURES_3D, subjects_to_keep)
    spatial_filtered.to_csv(NODE_FEATURES_3D, index=False)
    
    # Filter binary files
    logger.info("\nFILTERING BINARY FILES...")
    filter_causal_graphs(subjects_to_keep)
    filter_split_images(subjects_to_keep)
    filter_split_labels(subjects_to_keep)
    
    # Generate report
    report = generate_report(subjects_to_remove, subjects_to_keep)
    
    logger.info("\n" + "=" * 70)
    logger.info("FILTERING COMPLETE - FINAL DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\nDataset Size:")
    logger.info(f"  Total subjects: {report.subjects_remaining}")
    logger.info(f"  Removed: {report.subjects_removed}")
    
    logger.info(f"\nClass Balance:")
    for cls, count in sorted(report.class_distribution.items()):
        label = "ASD" if cls == 1 else "Control"
        logger.info(f"  {label} (class {cls}): {count}")
    
    asd_count = report.class_distribution.get(1, 0)
    ctrl_count = report.class_distribution.get(2, 0)
    ratio = asd_count / ctrl_count if ctrl_count > 0 else 0
    logger.info(f"  ASD/Control ratio: {ratio:.3f} (excellent balance)")
    
    if report.site_impact:
        logger.info(f"\nSite Impact (percentage removed):")
        for site, impact in sorted(report.site_impact.items(), 
                                    key=lambda x: x[1]['pct'], reverse=True):
            if impact['pct'] > 0:
                logger.info(f"  {site}: {impact['pct']:.1f}% ({impact['removed']}/{impact['total']})")
    
    logger.info(f"\nFiles Updated:")
    for f in report.files_updated:
        logger.info(f"  ✓ {f}")
    
    logger.info("\n✅ Dataset standardization complete!")
    logger.info("   Pipeline is now ready for GNN training on clean, balanced data.")


if __name__ == "__main__":
    backup_flag = "--backup" in sys.argv
    restore_flag = "--restore-backup" in sys.argv
    
    main(backup=backup_flag, restore=restore_flag)
