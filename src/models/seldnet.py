"""
SELDNet: CRNN architecture for Sound Event Localization and Detection.

Adapted from DCASE baseline seldnet_model.py for our pipeline:
- Feature fps = Label fps = 100fps, so NO time pooling in CNN
- Frequency pooling: [4, 4, 2] -> 64 / 32 = 2 bins remaining
- Output: Multi-ACCDOA format (B, T, 3, 4, num_classes) for ADPIT loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ConvBlock2D


class SELDNet(nn.Module):
    """
    CRNN for SELD with Multi-ACCDOA output.

    Architecture:
        CNN: 3 x [Conv2d(64) -> BN -> ReLU -> MaxPool2d -> Dropout]
             Frequency pooling: [4, 4, 2] (64 -> 16 -> 4 -> 2)
             Time pooling: [1, 1, 1] (no time pooling since feat_fps == label_fps)
        Reshape: (B, T, 64*2) = (B, T, 128)
        BiGRU: 2 layers, hidden=128 -> output dim = 128 (gated merge of bidirectional)
        Self-Attention: 2 layers, 8 heads
        FNN: Linear -> output
        Output: (B, T, 3*4*num_classes) for Multi-ACCDOA

    Args:
        in_channels: number of input channels (7 for FOA, 6 for binaural)
        num_classes: number of sound event classes (default: 8)
        num_tracks: number of output tracks for Multi-ACCDOA (default: 3)
        cnn_filters: number of CNN filters (default: 64)
        rnn_size: GRU hidden size (default: 128)
        rnn_layers: number of GRU layers (default: 2)
        attn_layers: number of self-attention layers (default: 2)
        attn_heads: number of attention heads (default: 8)
        fnn_size: feedforward layer size (default: 128)
        dropout: dropout rate (default: 0.05)
        f_pool_sizes: frequency pooling sizes (default: [4, 4, 2])
        t_pool_sizes: time pooling sizes (default: [1, 1, 1])
        n_freq_bins: number of frequency bins in input (default: 64)
    """

    def __init__(self, in_channels=7, num_classes=8, num_tracks=3,
                 cnn_filters=64, rnn_size=128, rnn_layers=2,
                 attn_layers=2, attn_heads=8, fnn_size=128,
                 dropout=0.05, f_pool_sizes=None, t_pool_sizes=None,
                 n_freq_bins=64):
        super().__init__()

        if f_pool_sizes is None:
            f_pool_sizes = [4, 4, 2]
        if t_pool_sizes is None:
            t_pool_sizes = [1, 1, 1]

        self.num_classes = num_classes
        self.num_tracks = num_tracks

        # CNN layers
        self.conv_blocks = nn.ModuleList()
        for i in range(len(f_pool_sizes)):
            in_ch = cnn_filters if i > 0 else in_channels
            self.conv_blocks.append(ConvBlock2D(in_ch, cnn_filters))
            self.conv_blocks.append(nn.MaxPool2d((t_pool_sizes[i], f_pool_sizes[i])))
            self.conv_blocks.append(nn.Dropout2d(p=dropout))

        # Compute GRU input dim after CNN frequency pooling
        freq_after_pool = int(np.floor(n_freq_bins / np.prod(f_pool_sizes)))
        self.gru_input_dim = cnn_filters * freq_after_pool

        # BiGRU
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Self-attention layers (following baseline: use rnn_size as embed_dim
        # after gated merge of bidirectional output)
        self.mhsa_blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(attn_layers):
            self.mhsa_blocks.append(
                nn.MultiheadAttention(
                    embed_dim=rnn_size, num_heads=attn_heads,
                    dropout=dropout, batch_first=True
                )
            )
            self.layer_norms.append(nn.LayerNorm(rnn_size))

        # Feedforward output
        self.fnn = nn.Linear(rnn_size, fnn_size)
        # Output: 3 tracks * 3 axes (x, y, z) * num_classes
        # In Multi-ACCDOA, activity is implied by norm(xyz) > 0.5
        # No distance output needed for our task
        self.output_layer = nn.Linear(fnn_size, num_tracks * 3 * num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, C, T, F) input features

        Returns:
            output: (B, T, 3*4*num_classes) Multi-ACCDOA output
        """
        # CNN
        for layer in self.conv_blocks:
            x = layer(x)

        # Reshape: (B, C_cnn, T, F_reduced) -> (B, T, C_cnn * F_reduced)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        # BiGRU with gated merge (following baseline)
        x, _ = self.gru(x)
        x = torch.tanh(x)
        # Gated merge of forward and backward: element-wise multiply
        x = x[:, :, x.shape[-1] // 2:] * x[:, :, :x.shape[-1] // 2]

        # Self-attention with residual connections
        for mhsa, ln in zip(self.mhsa_blocks, self.layer_norms):
            x_in = x
            x, _ = mhsa(x_in, x_in, x_in)
            x = x + x_in
            x = ln(x)

        # Feedforward
        x = self.fnn(x)

        # Output projection
        output = self.output_layer(x)  # (B, T, 3*3*num_classes)

        return output
