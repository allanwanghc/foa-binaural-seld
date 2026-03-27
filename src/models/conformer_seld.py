"""
Conformer-based SELD model.

Uses Conformer blocks (FFN -> MHSA -> ConvModule -> FFN -> LayerNorm)
instead of GRU + self-attention. The CNN front-end reduces frequency
dimension while preserving time resolution.

Output: Multi-ACCDOA format for ADPIT loss.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ConvBlock2D, ConformerBlock, PositionalEncoding


class ConformerSELD(nn.Module):
    """
    Conformer-based SELD model.

    Architecture:
        CNN front-end: 2 x [Conv2d(128) -> BN -> ReLU -> MaxPool(1, 4)]
            -> (B, 128, T, F/16=4)
        Reshape: (B, T, 128*4=512)
        Linear projection: 512 -> d_model (256)
        Positional Encoding
        N x ConformerBlock:
            FFN(half) -> MHSA -> ConvModule(k=31) -> FFN(half) -> LayerNorm
        FC output: d_model -> num_tracks * 4 * num_classes

    Args:
        in_channels: number of input channels (7 for FOA, 6 for binaural)
        num_classes: number of sound event classes (default: 8)
        num_tracks: number of output tracks for Multi-ACCDOA (default: 3)
        cnn_filters: number of CNN filters (default: 128)
        d_model: Conformer model dimension (default: 256)
        n_heads: number of attention heads (default: 8)
        n_conformer_layers: number of Conformer blocks (default: 4)
        conv_kernel_size: ConvModule kernel size (default: 31)
        d_ff: feedforward dimension (default: 1024)
        dropout: dropout rate (default: 0.1)
        f_pool_sizes: frequency pooling sizes for CNN (default: [4, 4])
        n_freq_bins: number of frequency bins in input (default: 64)
    """

    def __init__(self, in_channels=7, num_classes=8, num_tracks=3,
                 cnn_filters=128, d_model=256, n_heads=8,
                 n_conformer_layers=4, conv_kernel_size=31,
                 d_ff=1024, dropout=0.1,
                 f_pool_sizes=None, n_freq_bins=64):
        super().__init__()

        if f_pool_sizes is None:
            f_pool_sizes = [4, 4]

        self.num_classes = num_classes
        self.num_tracks = num_tracks

        # CNN front-end (no time pooling)
        self.cnn = nn.ModuleList()
        for i, f_pool in enumerate(f_pool_sizes):
            in_ch = cnn_filters if i > 0 else in_channels
            self.cnn.append(ConvBlock2D(in_ch, cnn_filters))
            self.cnn.append(nn.MaxPool2d((1, f_pool)))
            self.cnn.append(nn.Dropout2d(p=dropout))

        # Compute reshape dimension
        freq_after_pool = int(np.floor(n_freq_bins / np.prod(f_pool_sizes)))
        cnn_out_dim = cnn_filters * freq_after_pool

        # Linear projection to d_model
        self.input_proj = nn.Linear(cnn_out_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_conformer_layers)
        ])

        # Output projection: 3 tracks * 3 axes (x, y, z) * num_classes
        self.output_layer = nn.Linear(d_model, num_tracks * 3 * num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, F) input features

        Returns:
            output: (B, T, 3*4*num_classes) Multi-ACCDOA output
        """
        # CNN front-end
        for layer in self.cnn:
            x = layer(x)

        # Reshape: (B, C_cnn, T, F_reduced) -> (B, T, C_cnn * F_reduced)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        # Linear projection
        x = self.input_proj(x)

        # Positional encoding
        x = self.pos_enc(x)

        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)

        # Output
        output = self.output_layer(x)  # (B, T, 3*3*num_classes)

        return output
