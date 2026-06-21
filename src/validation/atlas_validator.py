import gzip
import logging
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# Atlas sources
AAL3_DOWNLOAD_URL = "https://www.gin.cnrs.fr/wp-content/uploads/aal3_for_SPM12.tar.gz"
BACKUP_AAL2_URL = "https://www.gin.cnrs.fr/wp-content/uploads/aal2.nii.gz"


def download_file(url: str, output_path: Path) -> bool:

    # Method 1: requests (preferred - better SSL handling)
    try:
        import requests

        logger.info(f"Downloading with requests: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✓ Download successful: {output_path.name}")
            return True

    except ImportError:
        logger.warning("requests library not found, falling back to curl")
    except Exception as e:
        logger.warning(f"requests failed: {e}, trying curl")

    # Method 2: curl (fallback - available on most systems)
    try:
        logger.info(f"Downloading with curl: {url}")
        subprocess.run(
            [
                "curl",
                "-L",
                "--insecure",
                "-o",
                str(output_path),
                url,
                "--max-time",
                "300",
            ],
            capture_output=True,
            check=True,
        )

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"✓ Download successful: {output_path.name}")
            return True

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"curl failed: {e}")

    logger.error("All download methods failed")
    return False


# VALIDATION


def validate_atlas(atlas_path: Path) -> bool:

    if not atlas_path.exists():
        return False

    try:
        img = nib.load(str(atlas_path))
        data = img.get_fdata()

        # Check dimensionality
        if data.ndim != 3:
            logger.error("Atlas is not 3D")
            return False

        # Check for invalid values
        if not np.isfinite(data).all():
            logger.error("Atlas contains NaN/Inf values")
            return False

        if np.count_nonzero(data) == 0:
            logger.error("Atlas is empty (all zeros)")
            return False

        # Check ROI count
        labels = np.unique(data)
        num_rois = len(labels[labels > 0])

        if num_rois not in {116, 117, 120, 164, 166, 170}:
            logger.warning(f"Unexpected ROI count: {num_rois}")
            # Don't fail - might be a valid variant

        logger.info(f"✓ Valid atlas | Shape={data.shape} | ROIs={num_rois}")
        return True

    except Exception as e:
        logger.error(f"Atlas validation error: {e}")
        return False


# ATLAS DISCOVERY & DOWNLOAD


def find_existing_atlas(atlas_dir: Path) -> Path | None:

    if not atlas_dir.exists():
        return None

    logger.info(f"Scanning atlas directory: {atlas_dir}")
    candidates = list(atlas_dir.glob("*.nii")) + list(atlas_dir.glob("*.nii.gz"))

    for atlas in candidates:
        logger.info(f"Checking {atlas.name}")
        if validate_atlas(atlas):
            logger.info(f"✓ Found valid atlas: {atlas}")
            return atlas

    return None


def download_aal3_atlas(output_dir: Path) -> Path:

    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "aal3_for_SPM12.tar.gz"

    if not download_file(AAL3_DOWNLOAD_URL, tar_path):
        raise RuntimeError("Failed to download AAL3")

    # Extract tarball
    import tarfile

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Find extracted atlas
    candidates = list(output_dir.glob("**/AAL3*.nii*"))
    if not candidates:
        raise RuntimeError("AAL3 not found after extraction")

    atlas = candidates[0]

    # Decompress if needed
    if atlas.suffix == ".gz":
        final = atlas.with_suffix("")
        with gzip.open(atlas, "rb") as f_in, open(final, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        atlas.unlink()
        atlas = final

    # Rename to standard name
    final_path = output_dir / "AAL3v1.nii"
    shutil.move(atlas, final_path)

    # Cleanup
    tar_path.unlink()

    return final_path


def download_aal2_fallback(output_dir: Path) -> Path:

    gz_path = output_dir / "AAL2.nii.gz"

    if not download_file(BACKUP_AAL2_URL, gz_path):
        raise RuntimeError("Failed to download AAL2")

    # Decompress
    atlas_path = output_dir / "AAL2.nii"
    with gzip.open(gz_path, "rb") as f_in, open(atlas_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()
    return atlas_path
