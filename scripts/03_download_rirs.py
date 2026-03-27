#!/usr/bin/env python3
"""
Step 3: Download SpatialScaper RIR datasets.

This is a wrapper that calls SpatialScaper's built-in prepare_rirs.py script.
Downloads METU, TAU (9 rooms), ARNI, MOTUS, RSOANU, DAGA RIR databases
and converts them to SOFA format for SpatialScaper.

Usage:
    python scripts/03_download_rirs.py

Note: This downloads ~20GB of data and may take 30-60 minutes.
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RIR_DIR


def main():
    spatialscaper_dir = PROJECT_ROOT / "SpatialScaper-main"
    prepare_script = spatialscaper_dir / "scripts" / "prepare_rirs.py"

    if not prepare_script.exists():
        print(f"ERROR: Cannot find {prepare_script}")
        sys.exit(1)

    print(f"{'=' * 60}")
    print(f"Downloading SpatialScaper RIR datasets")
    print(f"  Script: {prepare_script}")
    print(f"  Output: {RIR_DIR}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable,
        str(prepare_script),
        "--path", str(RIR_DIR),
        "--cleanup",
    ]

    subprocess.run(cmd, cwd=str(spatialscaper_dir / "scripts"), check=True)

    # Verify output
    sofa_dir = RIR_DIR / "spatialscaper_RIRs"
    if sofa_dir.exists():
        sofa_files = list(sofa_dir.glob("*.sofa"))
        print(f"\n{'=' * 60}")
        print(f"RIR download complete!")
        print(f"  SOFA files: {len(sofa_files)}")
        for f in sorted(sofa_files):
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"    {f.name}: {size_mb:.1f} MB")
        print(f"{'=' * 60}")
    else:
        print("WARNING: No SOFA output directory found!")


if __name__ == "__main__":
    main()
