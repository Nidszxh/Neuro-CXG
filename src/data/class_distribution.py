import sys, logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import MASTER_MANIFEST, CAUSAL_GRAPHS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_class_distribution():
    """Comprehensive class distribution analysis."""
    
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


if __name__ == "__main__":
    analyze_class_distribution()