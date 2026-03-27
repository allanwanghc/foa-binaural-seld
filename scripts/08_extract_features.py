#!/usr/bin/env python3
"""
Batch feature extraction for FOA and binaural audio + ADPIT label encoding.

Reads splits.json to get all scenes, then for each scene:
  - Extracts FOA features -> output/features/foa/{scene_id}.npy
  - Extracts binaural features -> output/features/binaural/{scene_id}.npy
  - Encodes ADPIT labels -> output/features/labels/{scene_id}.npy

Usage:
    python scripts/08_extract_features.py [--workers N]
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FEATURE_DIR,
    FMAX,
    FMIN,
    HOP_LENGTH,
    LABEL_FPS,
    N_FFT,
    N_MELS,
    NUM_CLASSES,
    OUTPUT_DIR,
    SCENES_DIR,
    TARGET_SR,
)
from src.features.binaural_features import extract_binaural_features
from src.features.foa_features import extract_foa_features
from src.features.label_encoder import seld_labels_to_adpit


def setup_dirs():
    """Create output directories."""
    for subdir in ["foa", "binaural", "labels"]:
        (FEATURE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def process_scene(scene_id):
    """
    Extract features and encode labels for a single scene.

    Args:
        scene_id: scene identifier string (e.g., "scene_00001")

    Returns:
        tuple (scene_id, success, message)
    """
    scene_dir = SCENES_DIR / scene_id
    foa_path = scene_dir / "foa_WXYZ.wav"
    binaural_path = scene_dir / "binaural_LR.wav"
    label_path = scene_dir / "labels.tsv"

    foa_out = FEATURE_DIR / "foa" / f"{scene_id}.npy"
    bin_out = FEATURE_DIR / "binaural" / f"{scene_id}.npy"
    label_out = FEATURE_DIR / "labels" / f"{scene_id}.npy"

    # Skip if all outputs exist
    if foa_out.exists() and bin_out.exists() and label_out.exists():
        return (scene_id, True, "skipped (already extracted)")

    # Validate inputs exist
    for path, name in [
        (foa_path, "foa_WXYZ.wav"),
        (binaural_path, "binaural_LR.wav"),
        (label_path, "labels.tsv"),
    ]:
        if not path.exists():
            return (scene_id, False, f"missing {name}")

    try:
        # Extract FOA features
        if not foa_out.exists():
            foa_feat = extract_foa_features(
                foa_path,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX,
            )
            np.save(foa_out, foa_feat)
        else:
            foa_feat = np.load(foa_out)

        # Extract binaural features
        if not bin_out.exists():
            bin_feat = extract_binaural_features(
                binaural_path,
                sr=TARGET_SR,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmin=FMIN,
                fmax=FMAX,
            )
            np.save(bin_out, bin_feat)

        # Encode ADPIT labels
        if not label_out.exists():
            # Use the number of FOA feature frames as reference
            num_frames = foa_feat.shape[0]
            label_mat = seld_labels_to_adpit(
                label_path, num_frames, num_classes=NUM_CLASSES
            )
            np.save(label_out, label_mat)

        return (scene_id, True, "done")

    except Exception as e:
        return (scene_id, False, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction for SELD"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    # Load splits to get all scene IDs
    splits_path = OUTPUT_DIR / "splits.json"
    if not splits_path.exists():
        print(f"ERROR: {splits_path} not found. Run 07_split_dataset.py first.")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    all_scenes = []
    for split_name in ["train", "val", "test"]:
        all_scenes.extend(splits.get(split_name, []))

    print(f"Total scenes to process: {len(all_scenes)}")
    setup_dirs()

    start_time = time.time()

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = pool.map(process_scene, all_scenes)
    else:
        results = [process_scene(sid) for sid in all_scenes]

    # Report results
    n_done = sum(1 for _, ok, msg in results if ok and msg == "done")
    n_skip = sum(1 for _, ok, msg in results if ok and msg != "done")
    n_fail = sum(1 for _, ok, _ in results if not ok)

    elapsed = time.time() - start_time
    print(f"\nFeature extraction complete in {elapsed:.1f}s")
    print(f"  Extracted: {n_done}")
    print(f"  Skipped:   {n_skip}")
    print(f"  Failed:    {n_fail}")

    if n_fail > 0:
        print("\nFailed scenes:")
        for sid, ok, msg in results:
            if not ok:
                print(f"  {sid}: {msg}")


if __name__ == "__main__":
    main()
