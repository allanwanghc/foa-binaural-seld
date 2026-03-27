"""
Per-channel z-score normalization using Welford online algorithm.

Computes running mean and variance across all training features
without loading everything into memory at once.
"""

import json
import numpy as np
from pathlib import Path


def compute_norm_stats(feature_dir, scene_list, num_channels, n_mels):
    """
    Compute per-channel mean and std using Welford online algorithm.

    Iterates over all .npy feature files in the given directory for scenes
    in scene_list, computing running statistics channel-by-channel.

    Args:
        feature_dir: Path to directory containing .npy feature files
        scene_list: list of scene IDs (e.g., ["scene_00001", ...])
        num_channels: number of feature channels (7 for FOA, 6 for binaural)
        n_mels: number of mel bins per channel

    Returns:
        tuple (mean, std) each of shape (num_channels, n_mels), dtype float32
    """
    feature_dir = Path(feature_dir)

    # Welford online algorithm state
    count = 0
    mean = np.zeros((num_channels, n_mels), dtype=np.float64)
    m2 = np.zeros((num_channels, n_mels), dtype=np.float64)

    for scene_id in scene_list:
        feat_path = feature_dir / f"{scene_id}.npy"
        if not feat_path.exists():
            continue

        feat = np.load(feat_path)  # (T, C, n_mels)
        assert feat.shape[1] == num_channels and feat.shape[2] == n_mels, (
            f"Feature shape mismatch: expected (T, {num_channels}, {n_mels}), "
            f"got {feat.shape}"
        )

        # Process each frame as an observation
        for t in range(feat.shape[0]):
            count += 1
            frame = feat[t].astype(np.float64)  # (C, n_mels)
            delta = frame - mean
            mean += delta / count
            delta2 = frame - mean
            m2 += delta * delta2

    if count < 2:
        raise ValueError(
            f"Need at least 2 frames to compute stats, got {count}"
        )

    variance = m2 / (count - 1)
    std = np.sqrt(variance)

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(features, mean, std, eps=1e-8):
    """
    Apply z-score normalization.

    Args:
        features: np.ndarray of shape (T, C, n_mels)
        mean: np.ndarray of shape (C, n_mels)
        std: np.ndarray of shape (C, n_mels)
        eps: small constant to avoid division by zero

    Returns:
        np.ndarray of same shape as features, dtype float32
    """
    return ((features - mean[np.newaxis]) / (std[np.newaxis] + eps)).astype(
        np.float32
    )


def save_norm_stats(mean, std, output_path):
    """
    Save normalization statistics to .npz file.

    Args:
        mean: np.ndarray of shape (C, n_mels)
        std: np.ndarray of shape (C, n_mels)
        output_path: Path to save .npz file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, mean=mean, std=std)
    print(f"Saved normalization stats to {output_path}")


def load_norm_stats(stats_path):
    """
    Load normalization statistics from .npz file.

    Args:
        stats_path: Path to .npz file

    Returns:
        tuple (mean, std) each of shape (C, n_mels)
    """
    data = np.load(stats_path)
    return data["mean"], data["std"]
