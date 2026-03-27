"""
Shared model components for SELD architectures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer-based models.

    Args:
        d_model: embedding dimension
        max_len: maximum sequence length
        dropout: dropout rate
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for Conformer ConvModule.

    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size for depthwise conv
        padding: padding for depthwise conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvModule(nn.Module):
    """
    Conformer ConvModule: LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv
    -> BN -> Swish -> Pointwise Conv -> Dropout

    Args:
        d_model: model dimension
        kernel_size: kernel size for depthwise conv (default: 31)
        dropout: dropout rate
    """

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise expansion (x2 for GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish
        # Pointwise projection
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x


class FeedForwardModule(nn.Module):
    """
    Conformer Feed Forward Module: LayerNorm -> Linear -> Swish -> Dropout
    -> Linear -> Dropout

    Args:
        d_model: model dimension
        d_ff: feedforward dimension (default: 4 * d_model)
        dropout: dropout rate
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()  # Swish
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """
    Conformer Block: FFN(half) -> MHSA -> ConvModule -> FFN(half) -> LayerNorm

    The Macaron-net style with half-step residual connections for the
    feed-forward modules.

    Args:
        d_model: model dimension
        n_heads: number of attention heads
        conv_kernel_size: kernel size for ConvModule (default: 31)
        d_ff: feedforward dimension
        dropout: dropout rate
    """

    def __init__(self, d_model, n_heads=8, conv_kernel_size=31,
                 d_ff=None, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.mhsa_layer_norm = nn.LayerNorm(d_model)
        self.mhsa_dropout = nn.Dropout(dropout)
        self.conv_module = ConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        # First FFN (half-step)
        x = x + 0.5 * self.ffn1(x)

        # MHSA with residual
        residual = x
        x_norm = self.mhsa_layer_norm(x)
        x_attn, _ = self.mhsa(x_norm, x_norm, x_norm)
        x = residual + self.mhsa_dropout(x_attn)

        # ConvModule with residual
        x = x + self.conv_module(x)

        # Second FFN (half-step)
        x = x + 0.5 * self.ffn2(x)

        # Final LayerNorm
        x = self.final_layer_norm(x)

        return x


class ConvBlock2D(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU block.

    Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size (default: (3, 3))
        stride: stride (default: (1, 1))
        padding: padding (default: (1, 1))
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
