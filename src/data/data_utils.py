"""
Utility functions for SELD data loading.
"""

import os
import numpy as np
import torch


def build_chunk_indices(scene_list, label_dir, chunk_length, hop_length):
    """
    Pre-compute (scene_idx, start_frame, actual_length) for all chunks.

    For each scene, loads the label to determine total frames, then creates
    sliding window chunks with the given hop_length.

    Args:
        scene_list: list of scene filenames (e.g., ['scene_001.npy', ...])
        label_dir: path to directory containing label .npy files
        chunk_length: number of frames per chunk (e.g., 500)
        hop_length: hop between chunk start frames (e.g., 250 for 50% overlap)

    Returns:
        List of (scene_idx, start_frame, actual_length) tuples.
        actual_length may be < chunk_length for the last chunk of a scene.
    """
    chunk_indices = []
    for scene_idx, scene_name in enumerate(scene_list):
        label_path = os.path.join(label_dir, scene_name)
        # Use mmap to avoid loading full file just to get shape
        label = np.load(label_path, mmap_mode='r')
        total_frames = label.shape[0]

        start = 0
        while start < total_frames:
            actual_length = min(chunk_length, total_frames - start)
            # Only include chunks that have at least 10% of chunk_length
            if actual_length >= max(1, chunk_length // 10):
                chunk_indices.append((scene_idx, start, actual_length))
            start += hop_length

    return chunk_indices


def collate_fn(batch):
    """
    Custom collate function for SELDDataset.
    Pads features and labels to the maximum chunk length in the batch.

    Args:
        batch: list of (feat, label) tuples from SELDDataset.__getitem__

    Returns:
        feat_batch: (B, C, T_max, F) tensor
        label_batch: (B, T_max, 6, 4, num_classes) tensor
    """
    feats, labels = zip(*batch)

    # Find max time length in batch
    max_t = max(f.shape[1] for f in feats)  # feats are (C, T, F)

    # Check if padding is needed
    needs_padding = any(f.shape[1] < max_t for f in feats)

    if not needs_padding:
        feat_batch = torch.stack(feats, dim=0)
        label_batch = torch.stack(labels, dim=0)
    else:
        C, _, F = feats[0].shape
        num_tracks = labels[0].shape[1]  # 6
        num_axes = labels[0].shape[2]    # 4
        num_classes = labels[0].shape[3]  # 8

        feat_batch = torch.zeros(len(feats), C, max_t, F)
        label_batch = torch.zeros(len(labels), max_t, num_tracks, num_axes, num_classes)

        for i, (f, l) in enumerate(zip(feats, labels)):
            t = f.shape[1]
            feat_batch[i, :, :t, :] = f
            label_batch[i, :t, :, :, :] = l

    return feat_batch, label_batch


def compute_norm_stats(feature_dir, scene_list):
    """
    Compute per-channel mean and std across all scenes for normalization.

    Args:
        feature_dir: path to feature .npy files
        scene_list: list of scene filenames

    Returns:
        mean: (C, 1, F) array
        std: (C, 1, F) array
    """
    # First pass: accumulate sums
    n_total = 0
    running_sum = None
    running_sq_sum = None

    for scene_name in scene_list:
        feat_path = os.path.join(feature_dir, scene_name)
        feat = np.load(feat_path)  # (T, C*F) or (T, C, F)

        if feat.ndim == 2:
            # Assume flattened: (T, C*F) - need to know C to reshape
            # For now, accumulate as-is and reshape later
            pass
        elif feat.ndim == 3:
            # (T, C, F) format
            pass

        if running_sum is None:
            running_sum = feat.sum(axis=0).astype(np.float64)
            running_sq_sum = (feat.astype(np.float64) ** 2).sum(axis=0)
        else:
            running_sum += feat.sum(axis=0).astype(np.float64)
            running_sq_sum += (feat.astype(np.float64) ** 2).sum(axis=0)
        n_total += feat.shape[0]

    mean = running_sum / n_total
    std = np.sqrt(running_sq_sum / n_total - mean ** 2)
    std = np.maximum(std, 1e-8)  # Avoid division by zero

    return mean.astype(np.float32), std.astype(np.float32)
