"""
CST-former: Channel-Spectro-Temporal Transformer for SELD.

Self-contained implementation adapted from the CST-former reference code.
Uses the CST-ULE (Unfolded Local Embedding) variant for channel attention.

Architecture:
    CNN front-end: 3 x [Conv2d(64) -> BN -> ReLU -> MaxPool -> Dropout]
    CST Attention (ULE variant):
        - Channel attention with Unfold/Fold local embedding
        - Spectral attention across frequency bins
        - Temporal attention across time frames
    FC output: Multi-ACCDOA format

Supports both FOA (7ch) and binaural (6ch) input.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.components import ConvBlock2D


class CSTAttentionULE(nn.Module):
    """
    Channel-Spectro-Temporal attention with Unfolded Local Embedding (ULE).

    Channel attention uses unfold/fold operations to create local patches
    as embeddings for cross-channel attention. Followed by spectral and
    temporal attention.

    Args:
        cnn_filters: number of CNN output channels (C dimension)
        freq_dim: frequency dimension after CNN pooling
        time_dim: time dimension (T)
        n_heads: number of attention heads
        dropout: dropout rate
        patch_size_t: temporal patch size for unfolding
        patch_size_f: frequency patch size for unfolding
        use_linear: whether to use linear projection after attention
        in_channels: number of input microphone channels
    """

    def __init__(self, cnn_filters=64, freq_dim=4, time_dim=500,
                 n_heads=8, dropout=0.1, patch_size_t=25, patch_size_f=4,
                 use_linear=True, in_channels=7):
        super().__init__()
        self.cnn_filters = cnn_filters
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.in_channels = in_channels
        self.use_linear = use_linear
        self.dropout_rate = dropout

        # Embedding dim: cnn_filters * freq_dim
        self.embed_dim = cnn_filters * freq_dim

        # Channel attention with ULE
        self.patch_size = (patch_size_t, patch_size_f)
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.fold = nn.Fold(
            output_size=(time_dim, freq_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        self.ch_attn_dim = patch_size_t * patch_size_f
        # Find valid number of heads for channel attention dim
        ch_n_heads = n_heads
        while self.ch_attn_dim % ch_n_heads != 0 and ch_n_heads > 1:
            ch_n_heads -= 1
        self.ch_mhsa = nn.MultiheadAttention(
            embed_dim=self.ch_attn_dim,
            num_heads=ch_n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ch_layer_norm = nn.LayerNorm(self.embed_dim)
        if use_linear:
            self.ch_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Spectral attention
        self.sp_mhsa = nn.MultiheadAttention(
            embed_dim=cnn_filters,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.sp_layer_norm = nn.LayerNorm(self.embed_dim)
        if use_linear:
            self.sp_linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Temporal attention
        self.temp_mhsa = nn.MultiheadAttention(
            embed_dim=cnn_filters,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.temp_layer_norm = nn.LayerNorm(self.embed_dim)
        if use_linear:
            self.temp_linear = nn.Linear(self.embed_dim, self.embed_dim)

        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, C, T, F):
        """
        Args:
            x: (B, T, C*F) where C=cnn_filters, F=freq_dim
            C: cnn filter dim
            T: time dim
            F: freq dim
        Returns:
            (B, T, C*F)
        """
        B = x.size(0)
        x_init = x.clone()

        # === Channel Attention with ULE ===
        # Reshape to (B, C, T, F) for unfold
        x_unfold_in = rearrange(x_init, 'b t (f c) -> b c t f', c=C, t=T, f=F).contiguous()
        x_unfold = self.unfold(x_unfold_in)  # (B, C*patch_t*patch_f, n_patches)
        x_unfold = rearrange(x_unfold, 'b (c u) tf -> (b tf) c u', c=C).contiguous()

        xc, _ = self.ch_mhsa(x_unfold, x_unfold, x_unfold)

        xc = rearrange(xc, '(b tf) c u -> b (c u) tf', b=B).contiguous()
        xc = self.fold(xc)  # (B, C, T, F)
        xc = rearrange(xc, 'b c t f -> b t (f c)').contiguous()

        if self.use_linear:
            xc = self.activation(self.ch_linear(xc))
        xc = xc + x_init
        if self.dropout_rate:
            xc = self.drop_out(xc)
        xc = self.ch_layer_norm(xc)

        # === Spectral Attention ===
        xs = rearrange(xc, 'b t (f c) -> (b t) f c', f=F).contiguous()
        xs, _ = self.sp_mhsa(xs, xs, xs)
        xs = rearrange(xs, '(b t) f c -> b t (f c)', t=T).contiguous()
        if self.use_linear:
            xs = self.activation(self.sp_linear(xs))
        xs = xs + xc
        if self.dropout_rate:
            xs = self.drop_out(xs)
        xs = self.sp_layer_norm(xs)

        # === Temporal Attention ===
        xt = rearrange(xs, 'b t (f c) -> (b f) t c', f=F).contiguous()
        xt, _ = self.temp_mhsa(xt, xt, xt)
        xt = rearrange(xt, '(b f) t c -> b t (f c)', f=F).contiguous()
        if self.use_linear:
            xt = self.activation(self.temp_linear(xt))
        xt = xt + xs
        if self.dropout_rate:
            xt = self.drop_out(xt)
        x = self.temp_layer_norm(xt)

        return x


class CSTFormer(nn.Module):
    """
    CST-former: Channel-Spectro-Temporal Transformer for SELD.

    Args:
        in_channels: number of input channels (7 for FOA, 6 for binaural)
        num_classes: number of sound event classes (default: 8)
        num_tracks: number of output tracks for Multi-ACCDOA (default: 3)
        cnn_filters: number of CNN filters (default: 64)
        n_attn_layers: number of CST attention layers (default: 2)
        n_heads: number of attention heads (default: 8)
        fnn_size: feedforward layer size (default: 256)
        dropout: dropout rate (default: 0.1)
        f_pool_sizes: frequency pooling sizes (default: [4, 4, 2])
        t_pool_sizes: time pooling sizes (default: [1, 1, 1])
        n_freq_bins: number of input frequency bins (default: 64)
        chunk_length: expected time frames (default: 500)
        patch_size_t: temporal patch size for ULE (default: 25)
        patch_size_f: frequency patch size for ULE (default: None, auto-set)
    """

    def __init__(self, in_channels=7, num_classes=8, num_tracks=3,
                 cnn_filters=64, n_attn_layers=2, n_heads=8,
                 fnn_size=256, dropout=0.1,
                 f_pool_sizes=None, t_pool_sizes=None,
                 n_freq_bins=64, chunk_length=500,
                 patch_size_t=25, patch_size_f=None):
        super().__init__()

        if f_pool_sizes is None:
            f_pool_sizes = [4, 4, 2]
        if t_pool_sizes is None:
            t_pool_sizes = [1, 1, 1]

        self.num_classes = num_classes
        self.num_tracks = num_tracks
        self.in_channels = in_channels

        # CNN front-end (no time pooling)
        self.conv_blocks = nn.ModuleList()
        for i in range(len(f_pool_sizes)):
            in_ch = cnn_filters if i > 0 else in_channels
            self.conv_blocks.append(ConvBlock2D(in_ch, cnn_filters))
            self.conv_blocks.append(nn.MaxPool2d((t_pool_sizes[i], f_pool_sizes[i])))
            self.conv_blocks.append(nn.Dropout2d(p=dropout))

        # Compute dimensions after CNN
        self.freq_dim = int(np.floor(n_freq_bins / np.prod(f_pool_sizes)))
        self.time_dim = int(np.floor(chunk_length / np.prod(t_pool_sizes)))
        self.embed_dim = cnn_filters * self.freq_dim

        # Auto-set patch_size_f to freq_dim if not specified
        if patch_size_f is None:
            patch_size_f = self.freq_dim

        # Adjust patch_size_t to evenly divide time_dim
        if self.time_dim % patch_size_t != 0:
            # Find nearest divisor
            for pt in range(patch_size_t, 0, -1):
                if self.time_dim % pt == 0:
                    patch_size_t = pt
                    break

        # CST attention layers
        self.cst_layers = nn.ModuleList([
            CSTAttentionULE(
                cnn_filters=cnn_filters,
                freq_dim=self.freq_dim,
                time_dim=self.time_dim,
                n_heads=n_heads,
                dropout=dropout,
                patch_size_t=patch_size_t,
                patch_size_f=patch_size_f,
                use_linear=True,
                in_channels=in_channels
            )
            for _ in range(n_attn_layers)
        ])

        # Feedforward output
        self.fnn = nn.ModuleList()
        self.fnn.append(nn.Linear(self.embed_dim, fnn_size))
        self.output_layer = nn.Linear(fnn_size, num_tracks * 3 * num_classes)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers, Kaiming for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, T, F) input features
                C_in = 7 (FOA) or 6 (binaural)

        Returns:
            output: (B, T, 3*4*num_classes) Multi-ACCDOA output
        """
        B, M, T, F_in = x.size()

        # CNN front-end
        for layer in self.conv_blocks:
            x = layer(x)
        # x: (B, cnn_filters, T, freq_dim)

        _, C, T_out, F_out = x.size()

        # Reshape to (B, T, C*F) for attention
        x = rearrange(x, 'b c t f -> b t (f c)').contiguous()

        # CST attention layers
        for cst_layer in self.cst_layers:
            x = cst_layer(x, C, T_out, F_out)

        # Feedforward output
        for fnn in self.fnn:
            x = fnn(x)

        output = self.output_layer(x)  # (B, T, 3*3*num_classes)

        return output
