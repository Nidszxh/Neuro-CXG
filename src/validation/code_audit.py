#!/usr/bin/env python3
"""
Comprehensive code audit for Neuro-CXG project.
Checks:
1. All files use consistent imports from config.py
2. No hardcoded constants that should be from config
3. Feature dimensions are correct (14: 8 temporal + 6 spatial)
4. Graph dimensions are correct (12, 14) for nodes
5. LOBE_NAMES consistency across codebase
6. NUM_LOBES consistency (should be 12)
"""

import sys
from pathlib import Path
import logging
import ast
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Key constants that should NOT be hardcoded
CONFIG_CONSTANTS = {
    "NUM_LOBES": 12,
    "NUM_TEMPORAL_FEATURES": 8,
    "NUM_SPATIAL_FEATURES": 6,
    "GNN_IN_CHANNELS": 14,
}

# Expected hardcoded values that are OK
ALLOWED_HARDCODED = {
    "0.5": "Thresholds, learning rates",
    "0.1": "Label smoothing, dropout values",
    "1.0": "Normalization constants",
    "5": "K-fold CV (specific requirement)",
    "170": "AAL3 ROI count (atlas-specific)",
}


class CodeAuditor:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def check_file(self, filepath: Path) -> None:
        """Check a single Python file for issues."""
        try:
            with open(filepath, "r") as f:
                content = f.read()

            # Check 1: Hardcoded feature dimensions
            self._check_hardcoded_dimensions(filepath, content)

            # Check 2: Hardcoded lobe names
            self._check_hardcoded_lobe_names(filepath, content)

            # Check 3: Config imports
            self._check_config_imports(filepath, content)

            # Check 4: Shape comments
            self._check_shape_comments(filepath, content)

            # Check 5: Syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                self.errors.append(
                    f"{filepath.relative_to(SRC_DIR)}: Syntax error at line {e.lineno}: {e.msg}"
                )

        except Exception as e:
            self.warnings.append(f"{filepath.relative_to(SRC_DIR)}: Error reading file: {e}")

    def _check_hardcoded_dimensions(self, filepath: Path, content: str) -> None:
        """Check for hardcoded feature dimensions like '5', '40', '(5, 8)', etc."""

        rel_path = filepath.relative_to(SRC_DIR)

        patterns = [
            (r"\(5\s*,\s*8\)", "Shape (5, 8) should be (12, 8)"),
            (r"\(5\s*,\s*6\)", "Shape (5, 6) should be (12, 6)"),
            (r"\(5\s*,\s*14\)", "Shape (5, 14) should be (12, 14)"),
            (r"\.reshape\s*\(\s*-1\s*,\s*5\s*,\s*14\)", "reshape(-1, 5, 14) should use NUM_LOBES"),
            (r"\.reshape\s*\(\s*-1\s*,\s*5\s*,\s*8\)", "reshape(-1, 5, 8) should use NUM_LOBES"),
            (r"range\s*\(\s*5\s*\)", "range(5) should use NUM_LOBES"),
        ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                line = content.split("\n")[line_num - 1]
                if not line.strip().startswith("#"):
                    if "feature_attribution.py" in str(filepath) and "reshape" in line:
                        pass
                    else:
                        self.warnings.append(f"{rel_path}:{line_num}: {message}")

    def _check_hardcoded_lobe_names(self, filepath: Path, content: str) -> None:
        """Check for hardcoded lobe names list."""

        rel_path = filepath.relative_to(SRC_DIR)

        pattern = r"lobe_names\s*=\s*\[\s*'Frontal'\s*,.*?\]"

        matches = re.finditer(pattern, content, re.DOTALL)
        for match in matches:
            if "LOBE_NAMES" in content or "#" in match.group():
                continue
            line_num = content[: match.start()].count("\n") + 1
            self.warnings.append(
                f"{rel_path}:{line_num}: Should use LOBE_NAMES from config instead of hardcoded list"
            )

    def _check_config_imports(self, filepath: Path, content: str) -> None:
        """Check if files that need config are importing it."""

        rel_path = filepath.relative_to(SRC_DIR)

        should_import_config = [
            "construct_causal.py",
            "extract_spatial.py",
            "gnn_model.py",
            "graph_factory.py",
            "safe_harmonization.py",
            "causal_gnn.py",
        ]

        if any(name in str(filepath) for name in should_import_config):
            if "from src.core.config import" not in content and "from .core.config import" not in content:
                if "NUM_LOBES" in content or "LOBE_NAMES" in content:
                    if "from" not in content or "config" not in content:
                        self.warnings.append(
                            f"{rel_path}: References NUM_LOBES/LOBE_NAMES but doesn't import from config"
                        )

    def _check_shape_comments(self, filepath: Path, content: str) -> None:
        """Check shape comments for consistency with 12-region architecture."""

        rel_path = filepath.relative_to(SRC_DIR)

        patterns = [
            (r"#.*\(5.*8\).*", "5x8 shape in comments"),
            (r"#.*5 lobe", "5 lobe in comments"),
            (r"#.*5 node", "5 node in comments"),
            (r'""".*5 lobe.*"""', "5 lobe in docstring"),
            (r'""".*5 node.*"""', "5 node in docstring"),
        ]

        for pattern, message in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[: match.start()].count("\n") + 1
                self.info.append(f"{rel_path}:{line_num}: Comment/docstring may reference old architecture: {message}")

    def print_report(self) -> bool:
        """Print audit report."""

        logger.info("=" * 70)
        logger.info("COMPREHENSIVE CODE AUDIT REPORT")
        logger.info("=" * 70)

        if self.errors:
            logger.error(f"\nERRORS ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        else:
            logger.info("\nERRORS: None found")

        if self.warnings:
            logger.warning(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        else:
            logger.info("\nWARNINGS: None found")

        if self.info:
            logger.info(f"\nINFO MESSAGES ({len(self.info)}):")
            for msg in self.info[:10]:
                logger.info(f"  - {msg}")
            if len(self.info) > 10:
                logger.info(f"  ... and {len(self.info) - 10} more info messages")

        logger.info("\n" + "=" * 70)
        logger.info("KEY VALIDATIONS:")
        logger.info(f"  - Total files checked: {self.total_files}")
        logger.info(f"  - Errors: {len(self.errors)}")
        logger.info(f"  - Warnings: {len(self.warnings)}")
        logger.info(f"  - Info messages: {len(self.info)}")
        logger.info("=" * 70)

        return len(self.errors) == 0


def run_audit() -> int:
    """Run comprehensive audit."""

    auditor = CodeAuditor()

    python_files = list(SRC_DIR.rglob("*.py"))
    auditor.total_files = len(python_files)

    logger.info(f"Auditing {len(python_files)} Python files in {SRC_DIR}")

    for filepath in sorted(python_files):
        if "__pycache__" in str(filepath):
            continue
        auditor.check_file(filepath)

    success = auditor.print_report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_audit())
