#!/usr/bin/env python3
"""
Step 2: Download HRTF datasets (KU100, CIPIC, SADIE II) in SOFA format.

Downloads:
    - KU100 (TH Koln): Full 2-degree resolution HRIR
    - CIPIC Subject 003: Representative generic HRTF
    - SADIE II H3: Neumann KU100 dummy head

Output:
    data/hrtfs/ku100/HRIR_FULL2DEG.sofa
    data/hrtfs/cipic/subject_003.sofa
    data/hrtfs/sadie_ii/H3_HRIR_SOFA.sofa
"""

import sys
import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import HRTF_DIR, HRTF_FILES


# HRTF download sources
HRTF_DOWNLOADS = {
    "ku100": {
        "url": "https://zenodo.org/records/3928297/files/HRIR_FULL2DEG.sofa",
        "filename": "HRIR_FULL2DEG.sofa",
        "is_zip": False,
    },
    "cipic": {
        "url": "https://sofacoustics.org/data/database/cipic/subject_003.sofa",
        "filename": "subject_003.sofa",
        "is_zip": False,
    },
    "sadie_ii": {
        "url": "https://zenodo.org/records/12092466/files/H3_HRIR_SOFA.zip",
        "filename": "H3_HRIR_SOFA.zip",
        "is_zip": True,
        "extract_target": "H3_HRIR_SOFA.sofa",
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, dest_path):
    """Download a file with progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  Already exists: {dest_path}")
        return

    print(f"  Downloading: {url}")
    print(f"  Saving to: {dest_path}")

    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest_path.name) as t:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=t.update_to)


def download_and_verify_hrtf(name, info):
    """Download a single HRTF dataset and verify it."""
    print(f"\n--- {name.upper()} ---")

    dest_dir = HRTF_DIR / name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if info["is_zip"]:
        # Download zip, extract target file
        zip_path = dest_dir / info["filename"]
        target_path = HRTF_FILES[name]

        if target_path.exists():
            print(f"  Already exists: {target_path}")
        else:
            download_file(info["url"], zip_path)

            # Extract
            print(f"  Extracting...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                # List contents to find the target
                names = zf.namelist()
                print(f"  Zip contents: {names}")

                # Extract all to dest_dir
                zf.extractall(str(dest_dir))

            # Clean up zip
            if zip_path.exists():
                zip_path.unlink()
                print(f"  Cleaned up zip file")

            # Check if target exists (might be in a subdirectory)
            if not target_path.exists():
                # Search for the sofa file
                sofa_files = list(dest_dir.rglob("*.sofa"))
                if sofa_files:
                    # Move the first .sofa file to the expected location
                    sofa_files[0].rename(target_path)
                    print(f"  Moved to: {target_path}")
                else:
                    print(f"  WARNING: No .sofa file found after extraction!")
                    return False
    else:
        # Direct download
        target_path = HRTF_FILES[name]
        download_file(info["url"], target_path)

    # Verify the SOFA file
    return verify_sofa(name, target_path)


def verify_sofa(name, sofa_path):
    """Verify a SOFA file can be loaded and has expected data."""
    try:
        import pysofaconventions as pysofa

        sofa = pysofa.SOFAFile(str(sofa_path), "r")
        is_valid = sofa.isValid()

        # Get basic info
        ir_data = sofa.getDataIR()
        sr = sofa.getSamplingRate()
        source_pos = sofa.getVariableValue("SourcePosition")

        n_measurements = ir_data.shape[0]
        n_receivers = ir_data.shape[1]
        n_samples = ir_data.shape[2]

        print(f"  Valid SOFA: {is_valid}")
        print(f"  Measurements: {n_measurements}")
        print(f"  Receivers (ears): {n_receivers}")
        print(f"  IR length: {n_samples} samples")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Source positions shape: {source_pos.shape}")

        return is_valid

    except Exception as e:
        print(f"  ERROR verifying {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Downloading HRTF Datasets")
    print("=" * 60)

    results = {}
    for name, info in HRTF_DOWNLOADS.items():
        success = download_and_verify_hrtf(name, info)
        results[name] = success

    print("\n" + "=" * 60)
    print("HRTF Download Summary:")
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)


if __name__ == "__main__":
    main()
