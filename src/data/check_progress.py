import os
import logging
import sys
import random
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_PROCESSED, DATA_ROOT, DATA_METADATA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataHealthChecker:
    """Monitor dataset completion, balance, and quality."""
    
    def _validate_png_files(self):
        """Validate PNG file integrity (not corrupted, correct dimensions)."""
        logger.info("-" * 40)
        logger.info("PNG FILE INTEGRITY")
        
        corrupted = []
        wrong_size = []
        zero_bytes = []
        
        if not self.downloaded_files:
            logger.warning("  ⚠️  No PNG files to validate")
            return

        sample = list(self.downloaded_files)
        if len(sample) > self.sample_png:
            sample = random.sample(sample, self.sample_png)

        for png_file in sample:
            png_path = self.png_dir / png_file
            
            # Check file size
            file_size = png_path.stat().st_size
            if file_size == 0:
                zero_bytes.append(png_file)
                continue
            
            # Try to load and validate
            try:
                img = Image.open(png_path)
                # Expected size for brain slices: 640x640
                if img.size != (640, 640):
                    wrong_size.append((png_file, img.size))
            except Exception as e:
                corrupted.append((png_file, str(e)[:40]))
        
        if not corrupted and not zero_bytes and not wrong_size:
            logger.info(f"  ✅ PNG files valid (sampled {len(sample)} files)")
        else:
            if zero_bytes:
                logger.warning(f"  ⚠️  Zero-byte PNG files: {len(zero_bytes)}")
                for f in zero_bytes[:5]:
                    logger.warning(f"      • {f}")
            if corrupted:
                logger.warning(f"  ⚠️  Corrupted PNG files: {len(corrupted)}")
                for f, err in corrupted[:5]:
                    logger.warning(f"      • {f}: {err}")
            if wrong_size:
                logger.warning(f"  ⚠️  Wrong dimensions: {len(wrong_size)}")
                for f, size in wrong_size[:5]:
                    logger.warning(f"      • {f}: {size} (expected 640x640)")
        self.errors["corrupted_png"] = corrupted
        self.errors["wrong_size_png"] = wrong_size
        self.errors["zero_png"] = zero_bytes
    
    def _validate_timeseries_files(self):
        """Validate time series .npy files (loadable, correct shape)."""
        logger.info("-" * 40)
        logger.info("TIME SERIES VALIDATION")
        
        ts_dir = DATA_PROCESSED
        if not ts_dir.exists():
            logger.warning(f"  ⚠️  Time series directory not found: {ts_dir}")
            return
        
        ts_files = list(ts_dir.glob("*_ts.npy"))
        logger.info(f"  Total files:        {len(ts_files)}")
        
        invalid = []
        wrong_shape = []
        
        sample = ts_files
        if len(sample) > self.sample_ts:
            sample = random.sample(sample, self.sample_ts)

        for ts_file in sample:
            try:
                data = np.load(ts_file)
                # Expected: (timepoints, 170 ROIs)
                if data.ndim != 2 or data.shape[1] != 170:
                    wrong_shape.append((ts_file.name, data.shape))
                # Check for NaN/Inf
                if np.isnan(data).any() or np.isinf(data).any():
                    invalid.append((ts_file.name, "contains NaN/Inf"))
            except Exception as e:
                invalid.append((ts_file.name, str(e)[:40]))
        
        if not invalid and not wrong_shape:
            logger.info(f"  ✅ Time series files valid (sampled {len(sample)} files)")
        else:
            if invalid:
                logger.warning(f"  ⚠️  Invalid time series files: {len(invalid)}")
                for f, err in invalid[:5]:
                    logger.warning(f"      • {f}: {err}")
            if wrong_shape:
                logger.warning(f"  ⚠️  Wrong shape: {len(wrong_shape)}")
                for f, shape in wrong_shape[:5]:
                    logger.warning(f"      • {f}: {shape} (expected (T, 170))")
        self.errors["invalid_ts"] = invalid
        self.errors["wrong_shape_ts"] = wrong_shape
    
    def _validate_metadata_quality(self):
        """Validate metadata CSV for missing values, outliers, data quality."""
        logger.info("-" * 40)
        logger.info("METADATA QUALITY")
        
        issues = {}
        
        # Check for missing DX_GROUP
        if 'DX_GROUP' in self.df.columns:
            missing_dx = self.df['DX_GROUP'].isna().sum()
            if missing_dx > 0:
                issues['Missing DX_GROUP'] = missing_dx
        
        # Check for missing AGE_AT_SCAN
        if 'AGE_AT_SCAN' in self.df.columns:
            missing_age = self.df['AGE_AT_SCAN'].isna().sum()
            invalid_age = (self.df['AGE_AT_SCAN'] < 0).sum() + (self.df['AGE_AT_SCAN'] > 100).sum()
            if missing_age > 0:
                issues['Missing AGE_AT_SCAN'] = missing_age
            if invalid_age > 0:
                issues['Invalid AGE_AT_SCAN (outliers)'] = invalid_age
        
        # Check for missing SEX
        if 'SEX' in self.df.columns:
            missing_sex = self.df['SEX'].isna().sum()
            invalid_sex = ~self.df['SEX'].isin([1, 2]).sum()
            if missing_sex > 0:
                issues['Missing SEX'] = missing_sex
            if invalid_sex > 0:
                issues['Invalid SEX values'] = invalid_sex
        
        # Check for duplicate FILE_IDs
        duplicates = self.df['FILE_ID'].duplicated().sum()
        if duplicates > 0:
            issues['Duplicate FILE_ID'] = duplicates
        
        if not issues:
            logger.info(f"  ✅ Metadata quality OK (no missing/invalid values)")
            self.errors["metadata_issues"] = []
        else:
            for issue, count in issues.items():
                logger.warning(f"  ⚠️  {issue}: {count}")
            self.errors["metadata_issues"] = list(issues.items())
    
    def _validate_feature_csv_quality(self):
        """Validate feature CSVs for NaN, Inf, and correct dimensions."""
        logger.info("-" * 40)
        logger.info("FEATURE FILE QUALITY")
        
        feature_files = {
            'Spatial Features': DATA_METADATA / "node_features_3d.csv",
            'Temporal Features': DATA_METADATA / "node_attributes_temporal.csv",
            'Harmonized Features': DATA_METADATA / "node_attributes_harmonized.csv",
        }
        
        for feature_name, feature_path in feature_files.items():
            if not feature_path.exists():
                logger.warning(f"  ⚠️  {feature_name:<25}: NOT FOUND")
                continue
            
            try:
                df = pd.read_csv(feature_path)
                
                # Check for NaN/Inf
                nan_count = df.isna().sum().sum()
                inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
                
                if nan_count == 0 and inf_count == 0:
                    logger.info(f"  ✅ {feature_name:<25}: {len(df)} subjects, clean")
                else:
                    issues = []
                    if nan_count > 0:
                        issues.append(f"{nan_count} NaNs")
                    if inf_count > 0:
                        issues.append(f"{inf_count} Infs")
                    issue_text = ', '.join(issues)
                    logger.warning(f"  ⚠️  {feature_name:<25}: {issue_text}")
                    self.errors["feature_issues"].append((feature_name, issue_text))
            except Exception as e:
                logger.warning(f"  ⚠️  {feature_name:<25}: ERROR - {str(e)[:40]}")
                self.errors["feature_issues"].append((feature_name, str(e)[:40]))

    def __init__(self, pheno_path=None, png_dir=None, sample_png=20, sample_ts=10, run_deep_checks=False):
        """Initialize with custom paths or use defaults.

        run_deep_checks toggles heavy validations (PNG/NPY content) for one-off deep audits.
        sample_png / sample_ts control random sample sizes for deep checks.
        """
        self.pheno_path = pheno_path or (DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv")
        self.png_dir = png_dir or (DATA_ROOT / "images")
        self.sample_png = sample_png
        self.sample_ts = sample_ts
        self.run_deep_checks = run_deep_checks

        self.df = None
        self.completed_subs = set()
        self.downloaded_files = []
        self.current_df = None
        self.errors = {
            "missing_metadata": [],
            "missing_images": [],
            "corrupted_png": [],
            "wrong_size_png": [],
            "zero_png": [],
            "invalid_ts": [],
            "wrong_shape_ts": [],
            "feature_issues": [],
            "metadata_issues": []
        }
    
    def _load_metadata(self):
        """Load and validate phenotype CSV."""
        if not self.pheno_path.exists():
            logger.error(f"Error: {self.pheno_path} not found.")
            return False
        
        self.df = pd.read_csv(self.pheno_path)
        # Ensure FILE_ID is string and strip whitespace
        self.df['FILE_ID'] = self.df['FILE_ID'].astype(str).str.strip()
        logger.info(f"Loaded metadata: {len(self.df)} records")
        return True
    
    def _load_images(self):
        """Scan for downloaded PNG files and extract subject IDs."""
        if not self.png_dir.exists():
            logger.error(f"Error: Image folder {self.png_dir} not found.")
            return False
        
        self.downloaded_files = [f for f in os.listdir(self.png_dir) if f.endswith('.png')]
        # Extract subject ID by splitting from right on '_z' (handles underscores in FILE_ID)
        self.completed_subs = set([f.rsplit('_z', 1)[0] for f in self.downloaded_files])
        logger.info(f"Found {len(self.downloaded_files)} PNG slices from {len(self.completed_subs)} subjects")
        return True
    
    def _match_metadata(self):
        """Match metadata to downloaded images."""
        if self.df is None or not self.completed_subs:
            logger.warning("No metadata or images loaded.")
            return False
        
        self.current_df = self.df[self.df['FILE_ID'].isin(self.completed_subs)].copy()
        
        if self.current_df.empty:
            logger.warning("[!] No matching metadata found for downloaded images.")
            logger.warning("Check if FILE_ID in CSV matches the prefix of your PNG files.")
            return False
        
        logger.info(f"Matched {len(self.current_df)} subjects to metadata")
        return True
    
    def _report_overview(self):
        """Print dataset overview."""
        logger.info("\n" + "="*40)
        logger.info(f"{'DATASET HEALTH REPORT':^40}")
        logger.info("="*40)
        logger.info(f"Unique Subjects:   {len(self.completed_subs)}")
        logger.info(f"Total PNG Slices:  {len(self.downloaded_files)}")
        
        if len(self.completed_subs) > 0:
            avg_slices = len(self.downloaded_files) / len(self.completed_subs)
            logger.info(f"Avg Slices/Sub:    {avg_slices:.1f} (Target: 5.0)")
    
    def _report_class_balance(self):
        """Print class distribution (ASD vs Control)."""
        logger.info("-" * 40)
        logger.info(f"CLASS BALANCE")
        
        # DX_GROUP: 1 = ASD, 2 = Control
        stats = self.current_df['DX_GROUP'].value_counts().to_dict()
        asd = stats.get(1, 0)
        tc = stats.get(2, 0)
        
        logger.info(f"  Autism (ASD):     {asd}")
        logger.info(f"  Controls (TC):    {tc}")
        
        if tc > 0:
            ratio = asd / tc
            logger.info(f"  Ratio (ASD/TC):   {ratio:.2f}")
    
    def _report_demographics(self):
        """Print age and sex distribution."""
        logger.info("-" * 40)
        logger.info(f"DEMOGRAPHICS")
        
        if 'AGE_AT_SCAN' in self.current_df.columns:
            valid_age = self.current_df[self.current_df['AGE_AT_SCAN'] > 0]['AGE_AT_SCAN']
            if not valid_age.empty:
                logger.info(f"  Avg Age:          {valid_age.mean():.1f} years")
        
        if 'SEX' in self.current_df.columns:
            # 1 = Male, 2 = Female
            sex_stats = self.current_df['SEX'].value_counts().to_dict()
            males = sex_stats.get(1, 0)
            females = sex_stats.get(2, 0)
            logger.info(f"  Sex Ratio (M/F):  {males}/{females}")
    
    def _report_sites(self):
        """Print site distribution."""
        logger.info("-" * 40)
        logger.info("TOP SITES")
        
        if 'SITE_ID' in self.current_df.columns:
            site_stats = self.current_df['SITE_ID'].value_counts().head(5)
            for site, count in site_stats.items():
                logger.info(f"  {str(site):<15}: {count} subjects")
        
        logger.info("="*40 + "\n")
    
    def _report_missing_subjects(self):
        """Identify and report subjects with missing metadata or missing images."""
        logger.info("-" * 40)
        logger.info("DATA COMPLETENESS")
        
        # Subjects with images but no metadata
        metadata_ids = set(self.df['FILE_ID'].unique())
        missing_metadata = self.completed_subs - metadata_ids
        
        if missing_metadata:
            logger.warning(f"  ⚠️  Subjects with images but NO metadata: {len(missing_metadata)}")
            for subid in sorted(missing_metadata):
                logger.warning(f"      • {subid}")
        else:
            logger.info(f"  ✅ All downloaded subjects have metadata")
        
        # Subjects with metadata but no images
        missing_images = metadata_ids - self.completed_subs
        
        if missing_images:
            logger.warning(f"  ⚠️  Subjects with metadata but NO images: {len(missing_images)}")
            # Show first 10 if list is long
            missing_list = sorted(missing_images)
            for subid in missing_list[:10]:
                logger.warning(f"      • {subid}")
            if len(missing_list) > 10:
                logger.warning(f"      ... and {len(missing_list) - 10} more")
        else:
            logger.info(f"  ✅ All metadata subjects have images")
        
        logger.info("="*40 + "\n")
        
        self.errors["missing_metadata"] = sorted(missing_metadata)
        self.errors["missing_images"] = sorted(missing_images)

        return {
            'missing_metadata': missing_metadata,
            'missing_images': missing_images
        }
    
    def _report_slice_distribution(self):
        """Check if each subject has expected 5 slices."""
        logger.info("-" * 40)
        logger.info("SLICE DISTRIBUTION")
        
        # Count slices per subject
        slice_counts = {}
        for filename in self.downloaded_files:
            subject_id = filename.rsplit('_z', 1)[0]
            slice_counts[subject_id] = slice_counts.get(subject_id, 0) + 1
        
        # Find subjects with incomplete slices
        incomplete = {subid: count for subid, count in slice_counts.items() if count != 5}
        complete = sum(1 for count in slice_counts.values() if count == 5)
        
        logger.info(f"  Complete (5 slices): {complete}/{len(slice_counts)}")
        
        if incomplete:
            logger.warning(f"  ⚠️  Subjects with incomplete slices: {len(incomplete)}")
            for subid in sorted(incomplete.keys())[:10]:
                logger.warning(f"      • {subid}: {incomplete[subid]} slices")
            if len(incomplete) > 10:
                logger.warning(f"      ... and {len(incomplete) - 10} more")
        else:
            logger.info(f"  ✅ All subjects have complete slice sets (5/5)")
    
    def _report_timeseries_files(self):
        """Check for time series .npy files in data/processed/."""
        logger.info("-" * 40)
        logger.info("TIME SERIES FILES")
        
        ts_dir = DATA_PROCESSED
        if not ts_dir.exists():
            logger.warning(f"  ⚠️  Time series directory not found: {ts_dir}")
            return
        
        ts_files = list(ts_dir.glob("*_ts.npy"))
        logger.info(f"  Time series files:  {len(ts_files)}")
        
        # Check for matching subjects
        ts_subjects = set([f.stem.replace('_ts', '') for f in ts_files])
        missing_ts = self.completed_subs - ts_subjects
        
        if missing_ts:
            logger.warning(f"  ⚠️  Downloaded subjects missing time series: {len(missing_ts)}")
            for subid in sorted(missing_ts)[:5]:
                logger.warning(f"      • {subid}")
            if len(missing_ts) > 5:
                logger.warning(f"      ... and {len(missing_ts) - 5} more")
        else:
            logger.info(f"  ✅ All downloaded subjects have time series")
    
    def _report_feature_files(self):
        """Check for extracted feature files."""
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
                    df = pd.read_csv(feature_path)
                    logger.info(f"  ✅ {feature_name:<25}: {len(df)} subjects")
                except Exception as e:
                    logger.warning(f"  ⚠️  {feature_name:<25}: ERROR - {str(e)[:50]}")
            else:
                logger.warning(f"  ⚠️  {feature_name:<25}: NOT FOUND")
    
    def _report_graph_files(self):
        """Check for constructed causal graph files."""
        logger.info("-" * 40)
        logger.info("GRAPH CONSTRUCTION STATUS")
        
        graph_dir = DATA_PROCESSED / "causal_graphs"
        if not graph_dir.exists():
            logger.warning(f"  ⚠️  Graph directory not found: {graph_dir}")
            return
        
        graph_files = list(graph_dir.glob("*_graph.pt"))
        logger.info(f"  Graph files:        {len(graph_files)}")
        
        if len(graph_files) > 0:
            logger.info(f"  Status:             ✅ Graphs constructed")
        else:
            logger.warning(f"  Status:             ⚠️  No graphs found (construct_causal.py needs to run)")

    def _summary(self):
        """Print a concise summary of statuses and key issues."""
        logger.info("\n" + "*" * 32)
        logger.info(f"{'SUMMARY':^32}")
        logger.info("*" * 32)
        logger.info(f"Metadata:           {'✅' if not self.errors['metadata_issues'] else '⚠️'}")
        logger.info(f"PNG sample:         {'✅' if not (self.errors['corrupted_png'] or self.errors['zero_png'] or self.errors['wrong_size_png']) else '⚠️'}")
        logger.info(f"Time series sample: {'✅' if not (self.errors['invalid_ts'] or self.errors['wrong_shape_ts']) else '⚠️'}")
        logger.info(f"Features:           {'✅' if not self.errors['feature_issues'] else '⚠️'}")

        if self.errors['corrupted_png']:
            logger.info(f"Corrupted PNG examples: {[x[0] for x in self.errors['corrupted_png'][:3]]}")
        if self.errors['wrong_size_png']:
            logger.info(f"Wrong-size PNG examples: {[x[0] for x in self.errors['wrong_size_png'][:3]]}")
        if self.errors['invalid_ts']:
            logger.info(f"Invalid TS examples: {[x[0] for x in self.errors['invalid_ts'][:3]]}")
        if self.errors['wrong_shape_ts']:
            logger.info(f"Wrong-shape TS examples: {[x[0] for x in self.errors['wrong_shape_ts'][:3]]}")
        if self.errors['feature_issues']:
            logger.info(f"Feature issues: {self.errors['feature_issues'][:3]}")
    
    def generate_report(self):
        """Generate and print complete health report."""
        # Load and validate data
        if not self._load_metadata():
            return False
        
        if not self._load_images():
            return False
        
        if not self._match_metadata():
            return False
        
        # Generate report sections
        self._report_overview()
        self._report_class_balance()
        self._report_demographics()
        self._report_sites()
        self._report_missing_subjects()
        self._report_slice_distribution()
        self._report_timeseries_files()
        self._report_feature_files()
        self._report_graph_files()
        
        # DATA INTEGRITY VALIDATIONS (optional heavy checks)
        if self.run_deep_checks:
            self._validate_png_files()
            self._validate_timeseries_files()
            self._validate_metadata_quality()
            self._validate_feature_csv_quality()

        # Summary output
        self._summary()
        
        return True


if __name__ == "__main__":
    checker = DataHealthChecker()
    checker.generate_report()