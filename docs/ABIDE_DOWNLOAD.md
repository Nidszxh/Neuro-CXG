# ABIDE Data Acquisition

This document describes how to obtain ABIDE I data for Neuro-CXG.

## Primary Method: AWS S3 (Default)

Neuro-CXG downloads ABIDE data from the INDI AWS S3 bucket.

### Access Requirements

- **AWS Credentials**: None required (public bucket)
- **Region**: `us-east-1`
- **Bucket**: `indiana-public-data`
- **Prefix**: `ABIDE_Initiative/`

### What Gets Downloaded

From S3:
- Preprocessed fMRI images (CPAC pipeline)
- Phenotype data (CSV)

Total size: ~150GB compressed

### Fallback Methods

If S3 access fails, try these alternatives:

### 1. Pre-downloaded Dataset

If you already have ABIDE data from a previous run:
```bash
# Skip download stage
python src/run_pipeline.py --auto --skip-download
```

### 2. Manual Download from OpenNeuro

ABIDE I is available on OpenNeuro:

1. Go to: https://openneuro.org/datasets/ds000031
2. Download the dataset (~16GB zipped)
3. Extract to `data/abide/`
4. Update config in `src/core/config.py` to point to your data

### 3. Pre-processed Time Series Only

If you only need time-series data:

1. Request from: https://ABIDE-initiative.org/
2. Use the "PCA/FV" (principal components/feature vectors) extraction
3. Place in `data/processed/`

### 4. Alternative Storage (Future)

We plan to provide AWS S3 pre-signed URLs for direct download.

## Verification

After download:
```bash
python src/run_pipeline.py --skip-download --skip-split
```

This validates existing data.

## Troubleshooting

### S3 Access Denied

- Check your IP is not blocked
- Try from a different network
- Use VPN if behind corporate firewall

### Slow Downloads

- Use `--n-jobs` flag for parallel downloads
- Consider regional AWS endpoints

### Data Corruption

Re-download individual subjects:
```bash
python -m src.validation.pipeline_checks --dataset
```