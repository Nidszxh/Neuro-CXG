from __future__ import annotations

import argparse
import shutil
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("NeuroCleaner")

# ── Configuration & Shields ──────────────────────────────────────────────────

# Files that identify the project root
ROOT_SENTINELS = {"requirements.txt", "pytest.ini", "README.md"}

# Global "Do Not Touch" patterns (Supports glob-style)
SHIELDED_DIRS = {
    "checkpoints*", "data", "raw", "atlases", ".git", "backup83"
}

SHIELDED_EXTS = {
    ".pt", ".pth", ".ckpt", ".nii", ".gz", ".npy", ".csv", 
    ".json", ".yaml", ".yml", ".py", ".md", ".bak"
}

# ── Logic Components ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleanCategory:
    label: str
    reason: str
    matcher: Callable[[Path], bool]

class ProjectContext:
    """Handles path resolution and global project state."""
    def __init__(self):
        self.root = self._find_root()

    def _find_root(self) -> Path:
        for parent in Path(__file__).resolve().parents:
            if any((parent / s).exists() for s in ROOT_SENTINELS):
                return parent
        return Path(__file__).resolve().parents[1]

    def is_shielded(self, path: Path) -> bool:
        # Check extensions
        if path.suffix.lower() in SHIELDED_EXTS:
            return True
        # Check path parts against shielded directory patterns
        parts = path.relative_to(self.root).parts if path.is_relative_to(self.root) else path.parts
        for part in parts:
            for pattern in SHIELDED_DIRS:
                if Path(part).match(pattern):
                    return True
        return False

# ── Specialized Matchers ─────────────────────────────────────────────────────

def get_stale_logs(root: Path, keep: int = 3) -> set[Path]:
    log_dir = root / "results"
    if not log_dir.exists():
        return set()
    logs = sorted(log_dir.glob("pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return set(logs[keep:])

# ── Category Registry ────────────────────────────────────────────────────────

def get_categories(ctx: ProjectContext) -> list[CleanCategory]:
    stale_logs = get_stale_logs(ctx.root)
    
    return [
        CleanCategory(
            "Python Bytecode", "Compiled artifacts re-generated at runtime.",
            lambda p: p.name == "__pycache__" or p.suffix in {".pyc", ".pyo", ".pyd"}
        ),
        CleanCategory(
            "Tool Caches", "Development and test metadata.",
            lambda p: p.name in {".pytest_cache", ".ipynb_checkpoints", ".mypy_cache", ".ruff_cache", ".coverage"}
        ),
        CleanCategory(
            "OS Junk", "System-generated metadata.",
            lambda p: p.name in {".DS_Store", "Thumbs.db", "desktop.ini"}
        ),
        CleanCategory(
            "Stale Logs", f"Old pipeline logs (kept {len(stale_logs)} newest).",
            lambda p: p in stale_logs
        ),
        CleanCategory(
            "YOLO Stub", "Incomplete training runs (v30) missing weights.",
            lambda p: p.name == "ROI_Detection_v30" and not (p / "weights" / "best.pt").exists()
        ),
        CleanCategory(
            "Pre-split Pool", "Redundant images/labels already moved to data/final/.",
            lambda p: (
                (p.suffix in {".png", ".txt"}) and 
                ("data/images" in p.as_posix() or "data/labels" in p.as_posix()) and
                "data/final" not in p.as_posix()
            )
        ),
    ]

# ── Core Engine ───────────────────────────────────────────────────────────────

@dataclass
class Finding:
    path: Path
    category: CleanCategory
    size: int

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()

def scan_project(ctx: ProjectContext) -> Iterator[Finding]:
    categories = get_categories(ctx)
    flagged_dirs: set[Path] = set()

    for path in sorted(ctx.root.rglob("*")):
        # Optimization: Skip children of already flagged directories
        if any(path.is_relative_to(d) for d in flagged_dirs):
            continue

        if ctx.is_shielded(path):
            continue

        for cat in categories:
            if cat.matcher(path):
                size = _get_size(path)
                yield Finding(path, cat, size)
                if path.is_dir():
                    flagged_dirs.add(path)
                break

def _get_size(path: Path) -> int:
    try:
        if path.is_file(): return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except (OSError, PermissionError):
        return 0

def format_size(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024: return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

# ── Execution ────────────────────────────────────────────────────────────────

def run_cleanup(findings: list[Finding], dry_run: bool = True):
    print(f"\n{'DRY RUN' if dry_run else 'EXECUTING'} MODE\n{'-'*40}")
    
    total_freed = 0
    for f in findings:
        status = "[SKIP]" if dry_run else "[DEL]"
        print(f"{status} {f.category.label:<15} | {f.path.relative_to(ROOT_CTX.root)} ({format_size(f.size)})")
        
        if not dry_run:
            try:
                if f.is_dir: shutil.rmtree(f.path)
                else: f.path.unlink()
                total_freed += f.size
            except Exception as e:
                logger.error(f"Failed to delete {f.path}: {e}")

    print(f"\nTotal space {'saved' if not dry_run else 'to be saved'}: {format_size(sum(f.size for f in findings))}")

# ── CLI ───────────────────────────────────────────────────────────────────────

ROOT_CTX = ProjectContext()

def main():
    parser = argparse.ArgumentParser(description="Neuro-CXG Project Cleaner")
    parser.add_argument("--execute", action="store_true", help="Delete files without confirmation")
    args = parser.parse_args()

    findings = list(scan_project(ROOT_CTX))
    
    if not findings:
        print("Project is already clean!")
        return

    if args.execute:
        run_cleanup(findings, dry_run=False)
    else:
        run_cleanup(findings, dry_run=True)
        confirm = input("\nProceed with deletion? (type 'yes'): ")
        if confirm.lower() == "yes":
            run_cleanup(findings, dry_run=False)

if __name__ == "__main__":
    main()