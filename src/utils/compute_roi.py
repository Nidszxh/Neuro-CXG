import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# SETUP LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR,
    NUM_TEMPORAL_FEATURES,
    ATLAS_PATH,
    ATLAS_METADATA,
)

# UTILITIES

def load_atlas_metadata(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Atlas metadata missing: {path}")
    with open(path) as f:
        meta = json.load(f)
    roi_ids = [m["roi_id"] for m in meta]
    return meta, roi_ids


def calculate_psd(ts: np.ndarray, tr: float) -> float:
    if len(ts) < 10:
        return 0.0
    ts = ts - np.mean(ts)
    freqs = np.fft.fftfreq(len(ts), d=tr)
    psd = np.abs(np.fft.fft(ts)) ** 2
    mask = (freqs > 0.01) & (freqs < 0.1)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


def extract_features(ts: np.ndarray, tr: float):
    """
    Fast + numerically safe feature extraction for a single ROI.
    """
    if not np.isfinite(ts).all():
        ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)

    std = np.std(ts)

    if std < 1e-6:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(skew(ts, bias=False))
        kurt_val = float(kurtosis(ts, bias=False))

    return {
        "mean": float(np.mean(ts)),
        "std": float(std),
        "skew": skew_val,
        "kurt": kurt_val,
        "psd": calculate_psd(ts, tr),
        "mssd": float(np.mean(np.diff(ts) ** 2)) if len(ts) > 1 else 0.0,
    }


def roi_is_valid(ts: np.ndarray, min_nonzero_ratio=0.1) -> bool:
    nonzero_ratio = np.count_nonzero(ts) / len(ts)
    return nonzero_ratio >= min_nonzero_ratio


# MAIN PIPELINE

def main():
    parser = argparse.ArgumentParser("ROI temporal feature extraction")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject subjects with any invalid ROI"
    )
    args = parser.parse_args()

    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest missing. Run manifest.py first.")
        return

    atlas_meta, atlas_roi_ids = load_atlas_metadata(ATLAS_METADATA)
    atlas_name = ATLAS_PATH.name
    atlas_roi_count = len(atlas_roi_ids)

    manifest = pd.read_csv(MASTER_MANIFEST)

    all_subjects = []
    roi_missing_counter = Counter()

    expected_n_rois = None
    processed = missing = failed = 0

    logger.info(f"Atlas: {atlas_name}")
    logger.info(f"Atlas ROIs: {atlas_roi_count}")
    logger.info(f"Subjects: {len(manifest)}")
    logger.info(f"Mode: {'STRICT' if args.strict else 'PRAGMATIC'}")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Subjects"):
        sub_id = row["subject_id"]
        split = row["split"]
        tr = row.get("TR", DEFAULT_TR)

        if pd.isna(tr) or tr <= 0:
            tr = DEFAULT_TR

        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        if not ts_path.exists():
            missing += 1
            continue

        try:
            ts_data = np.load(ts_path)
            if ts_data.ndim != 2:
                failed += 1
                continue

            n_rois = ts_data.shape[1]
            if expected_n_rois is None:
                expected_n_rois = n_rois
                logger.info(f"Detected ROI count: {expected_n_rois}")

            elif n_rois != expected_n_rois:
                logger.error(f"{sub_id}: ROI count mismatch ({n_rois})")
                failed += 1
                continue

            subject_row = {"subject_id": sub_id}
            bad_rois = 0

            for idx in range(n_rois):
                roi_ts = ts_data[:, idx]
                roi_id = atlas_roi_ids[idx] if idx < len(atlas_roi_ids) else idx + 1

                if not roi_is_valid(roi_ts):
                    roi_missing_counter[roi_id] += 1
                    bad_rois += 1

                    if args.strict:
                        break

                    # Pragmatic: fill NaNs
                    for k in ["mean", "std", "skew", "kurt", "psd", "mssd"]:
                        subject_row[f"roi{roi_id}_{k}"] = np.nan
                    continue

                feats = extract_features(roi_ts, tr)
                for k, v in feats.items():
                    subject_row[f"roi{roi_id}_{k}"] = v

            if args.strict and bad_rois > 0:
                failed += 1
                continue

            subject_row["missing_roi_fraction"] = bad_rois / n_rois
            all_subjects.append(subject_row)
            processed += 1

        except Exception as e:
            logger.exception(f"{sub_id}: failed ({e})")
            failed += 1

        # SAVE OUTPUT
    
    logger.info("=" * 60)
    logger.info(f"Processed: {processed}")
    logger.info(f"Missing TS: {missing}")
    logger.info(f"Failed:    {failed}")
    logger.info("=" * 60)

    if not all_subjects:
        logger.error("No valid subjects processed.")
        return

    df = pd.DataFrame(all_subjects)
    NODE_ATTRIBUTES_TEMPORAL.parent.mkdir(parents=True, exist_ok=True)

    with open(NODE_ATTRIBUTES_TEMPORAL, "w") as f:
        f.write(f"# atlas_name: {atlas_name}\n")
        f.write(f"# atlas_rois: {atlas_roi_count}\n")
        f.write(f"# features_per_roi: {NUM_TEMPORAL_FEATURES}\n")
        f.write(f"# strict_mode: {args.strict}\n")
        df.to_csv(f, index=False)

    # ROI coverage report
    coverage_report = {
        "atlas": atlas_name,
        "roi_missing_counts": dict(roi_missing_counter),
        "num_subjects": processed,
    }

    coverage_path = NODE_ATTRIBUTES_TEMPORAL.with_suffix(".roi_coverage.json")
    with open(coverage_path, "w") as f:
        json.dump(coverage_report, f, indent=2)

    logger.info("✅ Temporal ROI attributes saved")
    logger.info(f"CSV:  {NODE_ATTRIBUTES_TEMPORAL}")
    logger.info(f"ROI coverage: {coverage_path}")
    logger.info(f"Subjects kept: {processed}")


if __name__ == "__main__":
    main()