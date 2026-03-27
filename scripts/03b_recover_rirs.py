#!/usr/bin/env python3
"""
Recovery script for RIR download.

The prepare_rirs.py script failed during TAU-SNoise download (connection broken).
The TAU-SRIR data is already downloaded. This script:
1. Combines & extracts TAU-SRIR (skip SNoise - we don't use ambient noise)
2. Converts TAU rooms to SOFA
3. Downloads & processes remaining databases (ARNI, MOTUS, RSOANU, DAGA)
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "SpatialScaper-main" / "scripts"))

from config import RIR_DIR

# Add SpatialScaper scripts to path for imports
SS_SCRIPTS = PROJECT_ROOT / "SpatialScaper-main" / "scripts"
sys.path.insert(0, str(SS_SCRIPTS))
os.chdir(str(SS_SCRIPTS))

from prepare_rirs import (
    download_and_extract_remotes,
    prepare_tau,
    prepare_arni,
    prepare_metu,
    prepare_motus,
    prepare_rsoanu,
    prepare_daga,
    ARNI_REMOTES, MOTUS_REMOTES, RSOANU_REMOTES, DAGA_REMOTES,
)
from utils import extract_zip
import shlex


def combine_multizip_safe(filename, destination):
    """combine_multizip with proper path quoting for spaces."""
    cmd = f"zip -s 0 {shlex.quote(str(filename))} --out {shlex.quote(str(destination))}"
    print(f"  Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    source_path = RIR_DIR / "source_data"
    sofa_path = RIR_DIR / "spatialscaper_RIRs"
    source_path.mkdir(parents=True, exist_ok=True)
    sofa_path.mkdir(parents=True, exist_ok=True)

    tau_path = source_path / "tau"

    # ── Step 1: Combine & extract TAU-SRIR ──
    tau_srir_extracted = tau_path / "TAU-SRIR_DB"
    if not tau_srir_extracted.exists():
        print("=" * 60)
        print("Step 1: Combining & extracting TAU-SRIR multi-part zip")
        print("=" * 60)

        srir_zip = tau_path / "TAU-SRIR_DB.zip"
        single_zip = tau_path / "single.zip"

        if srir_zip.exists():
            print("  Combining multi-part zip...")
            combine_multizip_safe(str(srir_zip), str(single_zip))
            print("  Extracting...")
            extract_zip(str(single_zip), str(tau_path))
            # Clean up combined zip
            if single_zip.exists():
                os.remove(str(single_zip))
            print("  Done!")
        else:
            print(f"  ERROR: {srir_zip} not found!")
            return
    else:
        print("TAU-SRIR already extracted, skipping.")

    # ── Step 2: Convert TAU to SOFA ──
    # Check if TAU SOFA files already exist
    tau_sofa_exists = any(sofa_path.glob("bomb_shelter_foa.sofa"))
    if not tau_sofa_exists:
        print("\n" + "=" * 60)
        print("Step 2: Converting TAU rooms to SOFA")
        print("=" * 60)
        prepare_tau(tau_path, sofa_path)
    else:
        print("TAU SOFA files already exist, skipping.")

    # ── Step 3: Download & process remaining databases ──
    remaining = [
        ("ARNI", ARNI_REMOTES, prepare_arni),
        ("MOTUS", MOTUS_REMOTES, prepare_motus),
        ("RSOANU", RSOANU_REMOTES, prepare_rsoanu),
        ("DAGA", DAGA_REMOTES, prepare_daga),
    ]

    for name, remotes, prepare_fn in remaining:
        db_path = source_path / remotes["database_name"]
        sofa_name = f"{remotes['database_name']}_mic.sofa"

        # Check if already done
        if (sofa_path / sofa_name).exists():
            print(f"\n{name} SOFA already exists, skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"Step 3: Downloading & processing {name}")
        print(f"{'=' * 60}")

        try:
            download_and_extract_remotes(remotes["remotes"], db_path, cleanup=True)

            if name == "ARNI":
                prepare_fn(db_path, sofa_path)
            elif name == "MOTUS":
                prepare_fn(db_path, sofa_path)
            elif name == "RSOANU":
                prepare_fn(db_path, sofa_path)
            elif name == "DAGA":
                prepare_fn(db_path, sofa_path)

        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("RIR Recovery Summary")
    print(f"{'=' * 60}")
    sofa_files = sorted(sofa_path.glob("*.sofa"))
    for f in sofa_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")

    # Check which rooms we need
    needed = ["bomb_shelter_foa.sofa", "gym_foa.sofa", "sc203_foa.sofa"]
    print(f"\nNeeded for scene generation:")
    for name in needed:
        exists = (sofa_path / name).exists()
        print(f"  {name}: {'✓' if exists else '✗ MISSING'}")


if __name__ == "__main__":
    main()
