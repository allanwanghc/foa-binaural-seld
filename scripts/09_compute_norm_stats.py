#!/usr/bin/env python3
"""
Compute per-channel normalization statistics (mean, std) for training features.

Uses Welford online algorithm to compute statistics without loading all
features into memory at once.

Outputs:
  output/features/norm_stats/foa_norm_stats.npz
  output/features/norm_stats/binaural_norm_stats.npz

Usage:
    python scripts/09_compute_norm_stats.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BIN_NUM_CHANNELS,
    FEATURE_DIR,
    FOA_NUM_CHANNELS,
    N_MELS,
    OUTPUT_DIR,
)
from src.features.normalization import compute_norm_stats, save_norm_stats


def main():
    # Load splits to get training scenes
    splits_path = OUTPUT_DIR / "splits.json"
    if not splits_path.exists():
        print(f"ERROR: {splits_path} not found. Run 07_split_dataset.py first.")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    train_scenes = splits.get("train", [])
    if not train_scenes:
        print("ERROR: No training scenes found in splits.json")
        sys.exit(1)

    print(f"Computing normalization stats from {len(train_scenes)} training scenes")

    norm_stats_dir = FEATURE_DIR / "norm_stats"
    norm_stats_dir.mkdir(parents=True, exist_ok=True)

    # FOA normalization stats
    print("\nComputing FOA normalization stats...")
    foa_feat_dir = FEATURE_DIR / "foa"
    if foa_feat_dir.exists():
        foa_mean, foa_std = compute_norm_stats(
            foa_feat_dir, train_scenes, FOA_NUM_CHANNELS, N_MELS
        )
        save_norm_stats(
            foa_mean, foa_std, norm_stats_dir / "foa_norm_stats.npz"
        )
        print(f"  FOA mean range: [{foa_mean.min():.4f}, {foa_mean.max():.4f}]")
        print(f"  FOA std range:  [{foa_std.min():.4f}, {foa_std.max():.4f}]")
    else:
        print(f"  WARNING: {foa_feat_dir} not found, skipping FOA stats")

    # Binaural normalization stats
    print("\nComputing binaural normalization stats...")
    bin_feat_dir = FEATURE_DIR / "binaural"
    if bin_feat_dir.exists():
        bin_mean, bin_std = compute_norm_stats(
            bin_feat_dir, train_scenes, BIN_NUM_CHANNELS, N_MELS
        )
        save_norm_stats(
            bin_mean, bin_std, norm_stats_dir / "binaural_norm_stats.npz"
        )
        print(f"  Binaural mean range: [{bin_mean.min():.4f}, {bin_mean.max():.4f}]")
        print(f"  Binaural std range:  [{bin_std.min():.4f}, {bin_std.max():.4f}]")
    else:
        print(f"  WARNING: {bin_feat_dir} not found, skipping binaural stats")

    print("\nDone.")


if __name__ == "__main__":
    main()
