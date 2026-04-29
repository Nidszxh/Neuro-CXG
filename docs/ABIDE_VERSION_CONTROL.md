# ABIDE Dataset Version Control

**Status**: Documentation for reproducibility  
**Date**: April 29, 2026  

---

## ABIDE-I Dataset Versions

### Official Preprocessed Data

| Version | Source | Notes |
|---------|--------|-------|
| **CPAC 1.0** | INDI S3 Bucket | Current used by pipeline |
| CPAC 1.1 | INDI S3 Bucket | Updated preprocessing |

### Preprocessing Pipeline

The pipeline uses **CPAC filt_global** preprocessing:
- Bandpass filtering: 0.01–0.1 Hz
- Global signal regression: Yes
- Spatial normalization: MNI152

---

## Required Files & Checksums

### Phenotype CSV

| File | Version | MD5 Checksum | Status |
|------|---------|--------------|--------|
| `Phenotypic_V1_0b_preprocessed1.csv` | 1.0b | Required | Document this after first download |

### How to Generate Checksum

```bash
# After downloading phenotype CSV
md5sum data/raw/Phenotypic_V1_0b_preprocessed1.csv
```

### How to Verify (After Download)

```python
import hashlib

def verify_phenotype_checksum(csv_path: Path, expected_md5: str) -> bool:
    """Verify phenotype CSV matches expected checksum."""
    md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
    return md5 == expected_md5

# Example usage
csv_path = Path("data/raw/Phenotypic_V1_0b_preprocessed1.csv")
expected = "a1b2c3d4e5f6..."  # Replace with actual MD5
if verify_phenotype_checksum(csv_path, expected):
    print("✅ Checksum verified")
else:
    print("❌ Checksum mismatch - data may be corrupted or updated")
```

---

## Implementation: Add Checksum Validation

To add checksum validation to `abide_download.py`:

```python
# Add to abide_download.py
EXPECTED_PHENO_CHECKSUMS = {
    "Phenotypic_V1_0b_preprocessed1.csv": "a1b2c3d4e5f6...",  # Replace with actual
}

def verify_file_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file checksum matches expected."""
    import hashlib
    actual_md5 = hashlib.md5(file_path.read_bytes()).hexdigest()
    return actual_md5 == expected_md5

def download_phenotype_with_verification():
    """Download phenotype CSV with checksum verification."""
    # ... existing download code ...
    
    # Verify after download
    pheno_file = DATA_RAW / "Phenotypic_V1_0b_preprocessed1.csv"
    expected = EXPECTED_PHENO_CHECKSUMS.get(pheno_file.name)
    
    if expected and not verify_file_checksum(pheno_file, expected):
        raise ValueError(
            f"Phenotype file checksum mismatch! File may be updated. "
            f"Expected MD5: {expected}"
        )
    logger.info("Phenotype file checksum verified")
```

---

## Current Status

- [ ] **NOT IMPLEMENTED**: Checksum validation in abide_download.py
- [x] **DOCUMENTED**: Required phenotype version and checksum location
- [ ] **PENDING**: Generate MD5 checksum after first clean download

---

## Action Required

1. Run initial download with: `python src/run_pipeline.py --auto --skip-download`
2. Generate checksums: `md5sum data/raw/Phenotypic_V1_0b_preprocessed1.csv`
3. Update `EXPECTED_PHENO_CHECKSUMS` in abide_download.py
4. Re-run download to verify

---

## References

- ABIDE Preprocessing: https://fcp-indi.github.io/docs/user/preprocessing
- CPAC Pipeline: https://github.com/FCP-INDI/C-PAC
- INDI S3: s3://fcp-indi/data/Projects/ABIDE_Initiative/