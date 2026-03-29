"""
PyTorch Dataset for SELD training with Multi-ACCDOA ADPIT format.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.data_utils import build_chunk_indices


class SELDDataset(Dataset):
    """
    PyTorch Dataset for SELD.

    Args:
        feature_dir: Path to .npy feature files (foa/ or binaural/)
        label_dir: Path to .npy label files
        scene_list: list of scene filenames (e.g., ['scene_001.npy', ...])
        chunk_length: frames per chunk (default: 500, i.e. 5s at 100fps)
        hop_length: chunk hop for training overlap (default: 250 = 50% overlap)
        norm_stats: tuple of (mean, std) arrays, or None for no normalization
        mode: 'train' (random chunks with overlap) or 'eval' (sequential, no overlap)
    """

    def __init__(self, feature_dir, label_dir, scene_list, chunk_length=500,
                 hop_length=250, norm_stats=None, mode='train'):
        super().__init__()
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.scene_list = sorted(scene_list)
        self.chunk_length = chunk_length
        self.mode = mode
        self.norm_stats = norm_stats

        # Set hop: training uses overlap, eval uses no overlap
        if mode == 'train':
            self.hop_length = hop_length
        else:
            self.hop_length = chunk_length  # No overlap for eval

        # Pre-compute chunk indices
        self.chunk_indices = build_chunk_indices(
            self.scene_list, self.label_dir,
            self.chunk_length, self.hop_length
        )

    def __len__(self):
        return len(self.chunk_indices)

    def _load_feat(self, scene_idx):
        """Load feature file."""
        feat_path = os.path.join(self.feature_dir, self.scene_list[scene_idx])
        return np.load(feat_path)

    def _load_label(self, scene_idx):
        """Load label file."""
        label_path = os.path.join(self.label_dir, self.scene_list[scene_idx])
        return np.load(label_path)

    def __getitem__(self, idx):
        scene_idx, start_frame, actual_length = self.chunk_indices[idx]

        # Load feature and label via mmap
        feat_full = self._load_feat(scene_idx)
        label_full = self._load_label(scene_idx)

        # Extract chunk
        feat = np.array(feat_full[start_frame:start_frame + actual_length])  # Copy from mmap
        label = np.array(label_full[start_frame:start_frame + actual_length])

        # Features are stored as (T, C, F) - we need (C, T, F) for CNN input
        # If features are (T, C*F), reshape first
        if feat.ndim == 2:
            # Determine number of channels from feature dim
            # FOA: 7 channels, Binaural: 6 channels
            # With 64 mel bins: 7*64=448 or 6*64=384
            feat_dim = feat.shape[1]
            if feat_dim % 64 == 0:
                n_channels = feat_dim // 64
                feat = feat.reshape(actual_length, n_channels, 64)
            else:
                raise ValueError(f"Cannot determine channels from feature dim {feat_dim}")

        # feat shape: (T, C, F) -> (C, T, F)
        feat = feat.transpose(1, 0, 2)

        # Pad if shorter than chunk_length
        if actual_length < self.chunk_length:
            C, _, F = feat.shape
            pad_feat = np.zeros((C, self.chunk_length, F), dtype=feat.dtype)
            pad_feat[:, :actual_length, :] = feat
            feat = pad_feat

            # Label: (T, 6, 4, 8) or (T, 6, 5, 8)
            label_shape = list(label.shape)
            label_shape[0] = self.chunk_length
            pad_label = np.zeros(label_shape, dtype=label.dtype)
            pad_label[:actual_length] = label
            label = pad_label

        # Apply normalization
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            # mean, std can be (C, F) or (C*F,) shape
            if mean.ndim == 1:
                # Reshape to (C, 1, F) for broadcasting with (C, T, F)
                n_channels = feat.shape[0]
                n_freq = feat.shape[2]
                mean = mean.reshape(n_channels, n_freq)
                std = std.reshape(n_channels, n_freq)
            if mean.ndim == 2:
                mean = mean[:, np.newaxis, :]  # (C, 1, F)
                std = std[:, np.newaxis, :]
            feat = (feat - mean) / std

        # Convert to tensors
        feat = torch.from_numpy(feat.copy()).float()      # (C, T, F)
        label = torch.from_numpy(label.copy()).float()     # (T, 6, 4, 8)

        return feat, label
