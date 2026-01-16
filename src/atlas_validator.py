"""
Emergency Atlas Validation and Download Module

Folder-aware, idempotent atlas handling:
- Scans /data/atlases for existing valid atlases
- Avoids unnecessary re-downloads
- Detects incomplete / corrupt atlas files
"""

import logging
import urllib.request
import ssl
import gzip
import shutil
import subprocess
from pathlib import Path
import nibabel as nib
import numpy as np

# Try to import requests (better SSL handling)
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logging.warning("requests library not found. Install with: pip install requests")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Atlas sources
AAL3_DOWNLOAD_URL = "https://www.gin.cnrs.fr/wp-content/uploads/aal3_for_SPM12.tar.gz"
BACKUP_AAL2_URL = "https://www.gin.cnrs.fr/wp-content/uploads/aal2.nii.gz"


# ============================================================
# DOWNLOAD UTILITIES
# ============================================================

def _download_with_requests(url: str, output_path: Path) -> bool:
    if not HAS_REQUESTS:
        return False
    try:
        logger.info("Attempting download with requests...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"requests failed: {e}")
        return False


def _download_with_curl(url: str, output_path: Path) -> bool:
    try:
        if subprocess.run(["which", "curl"], capture_output=True).returncode != 0:
            return False
        subprocess.run(
            ["curl", "-L", "--insecure", "-o", str(output_path), url, "--max-time", "300"],
            check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"curl failed: {e}")
        return False


def _download_with_wget(url: str, output_path: Path) -> bool:
    try:
        if subprocess.run(["which", "wget"], capture_output=True).returncode != 0:
            return False
        subprocess.run(
            ["wget", "-O", str(output_path), url, "--timeout=300"],
            check=True,
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"wget failed: {e}")
        return False


def _download_with_urllib_no_verify(url: str, output_path: Path) -> bool:
    try:
        logger.warning("Downloading with SSL verification DISABLED")
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=300) as r:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(r, f)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.warning(f"urllib failed: {e}")
        return False


def download_file_robust(url: str, output_path: Path) -> bool:
    return (
        _download_with_requests(url, output_path)
        or _download_with_curl(url, output_path)
        or _download_with_wget(url, output_path)
        or _download_with_urllib_no_verify(url, output_path)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_atlas(atlas_path: Path) -> bool:
    if not atlas_path.exists():
        return False

    try:
        img = nib.load(str(atlas_path))
        data = img.get_fdata()

        if data.ndim != 3:
            logger.error("Atlas is not 3D")
            return False

        if not np.isfinite(data).all():
            logger.error("Atlas contains NaN/Inf values")
            return False

        if np.count_nonzero(data) == 0:
            logger.error("Atlas is empty (all zeros)")
            return False

        labels = np.unique(data)
        num_rois = len(labels[labels > 0])

        if num_rois not in {116, 117, 120, 166, 170}:
            logger.warning(f"Unexpected ROI count: {num_rois}")
            return False

        logger.info(
            f"✓ Valid atlas | Shape={data.shape} | ROIs={num_rois}"
        )
        return True

    except Exception as e:
        logger.error(f"Atlas validation error: {e}")
        return False


# ============================================================
# FOLDER-AWARE DISCOVERY
# ============================================================

def find_existing_valid_atlas(atlas_dir: Path) -> Path | None:
    if not atlas_dir.exists():
        return None

    logger.info(f"Scanning atlas directory: {atlas_dir}")

    candidates = list(atlas_dir.glob("*.nii")) + list(atlas_dir.glob("*.nii.gz"))

    for atlas in candidates:
        logger.info(f"Checking {atlas.name}")
        if validate_atlas(atlas):
            logger.info(f"✓ Reusing existing atlas: {atlas}")
            return atlas

    return None


# ============================================================
# DOWNLOADERS
# ============================================================

def download_aal3_atlas(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "aal3_for_SPM12.tar.gz"

    if not download_file_robust(AAL3_DOWNLOAD_URL, tar_path):
        raise RuntimeError("Failed to download AAL3")

    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)

    candidates = list(output_dir.glob("**/AAL3*.nii*"))
    if not candidates:
        raise RuntimeError("AAL3 not found after extraction")

    atlas = candidates[0]
    if atlas.suffix == ".gz":
        final = atlas.with_suffix("")
        with gzip.open(atlas, "rb") as f_in, open(final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        atlas.unlink()
        atlas = final

    final_path = output_dir / "AAL3v1.nii"
    shutil.move(atlas, final_path)
    tar_path.unlink()

    return final_path


def download_aal2_fallback(output_dir: Path) -> Path:
    gz = output_dir / "AAL2.nii.gz"

    if not download_file_robust(BACKUP_AAL2_URL, gz):
        raise RuntimeError("Failed to download AAL2")

    atlas = output_dir / "AAL2.nii"
    with gzip.open(gz, "rb") as f_in, open(atlas, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz.unlink()

    return atlas


# ============================================================
# MAIN ENTRY
# ============================================================

def ensure_atlas(atlas_path: Path, auto_download: bool = True) -> bool:
    logger.info("=" * 60)
    logger.info("ATLAS SETUP")
    logger.info("=" * 60)

    atlas_dir = atlas_path.parent

    # STEP 1: Reuse existing atlas if valid
    existing = find_existing_valid_atlas(atlas_dir)
    if existing:
        if existing != atlas_path:
            shutil.copy(existing, atlas_path)
        return True

    if not auto_download:
        return False

    # STEP 2: Download AAL3
    try:
        atlas = download_aal3_atlas(atlas_dir)
        return validate_atlas(atlas)
    except Exception as e:
        logger.warning(f"AAL3 failed: {e}")

    # STEP 3: Fallback AAL2
    try:
        atlas = download_aal2_fallback(atlas_dir)
        return validate_atlas(atlas)
    except Exception as e:
        logger.error(f"AAL2 fallback failed: {e}")

    return False


# ============================================================
# METADATA
# ============================================================

def generate_atlas_metadata(atlas_path: Path, output_path: Path):
    import json

    img = nib.load(str(atlas_path))
    data = img.get_fdata()
    affine = img.affine

    labels = np.unique(data)
    labels = labels[labels > 0]

    metadata = []
    for label in labels:
        idx = np.argwhere(data == label)
        centroid_vox = idx.mean(axis=0)
        centroid_mni = affine @ np.append(centroid_vox, 1)

        metadata.append(
            dict(
                roi_id=int(label),
                x=float(centroid_mni[0]),
                y=float(centroid_mni[1]),
                z=float(centroid_mni[2]),
                num_voxels=len(idx),
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata generated: {output_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    # Add project root to path for module imports
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    try:
        from src.config import ATLAS_PATH, ATLAS_METADATA

        if ensure_atlas(ATLAS_PATH, auto_download=True):
            generate_atlas_metadata(ATLAS_PATH, ATLAS_METADATA)
            print("✅ Atlas ready")
        else:
            print("❌ Atlas setup failed")
            sys.exit(1)
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from project root or ensure config.py is available")
        sys.exit(1)
