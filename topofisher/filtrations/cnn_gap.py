"""
CNN + Global Average Pooling filtration for 2D fields.

A deep strided CNN encoder that reduces large 2D inputs (e.g., 512×512 lensing
convergence maps) to a fixed-size feature vector via Global Average Pooling.
No fully-connected layers — purely convolutional.

Architecture (default):
  Input (512×512) → 6 strided conv layers → GAP → 64-dim vector

  Conv1: 1→64,   k=11, s=4  → 128×128
  Conv2: 64→128,  k=7,  s=4  → 32×32
  Conv3: 128→256, k=5,  s=2  → 16×16
  Conv4: 256→256, k=5,  s=2  → 8×8
  Conv5: 256→128, k=3,  s=2  → 4×4
  Conv6: 128→64,  k=3,  s=2  → 2×2
  GAP:   → 64

  Receptive field: 866 pixels (effectively global for 512×512)
  Parameters: ~3.2M

No BatchNorm — incompatible with Fisher information estimation
(removes mean structure that carries parameter information).

Designed for use with IdentityVectorization and MOPED compression.

Pipeline:
  Input (H×W) → CNN → GAP → output_dim vector
  Wraps output in identity filtration format: [[vec_0, vec_1, ...]]
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNGAPFiltration(nn.Module):
    """
    Deep strided CNN encoder with Global Average Pooling.

    Purely convolutional — no fully-connected layers. Uses large kernels
    at the top for wide receptive fields. Output dimension equals the
    last channel count.

    The output format is compatible with IdentityVectorization:
        [[vector_0, vector_1, ..., vector_n]] (single "homology dimension")

    Args:
        encoder_channels: List of output channels per layer.
        encoder_kernels: List of kernel sizes per layer.
        encoder_strides: List of strides per layer.

    Example:
        >>> filt = CNNGAPFiltration()
        >>> x = torch.randn(100, 512, 512)
        >>> out = filt(x)  # [[vec_0, ..., vec_99]], each vec is (64,)
    """

    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        circular_padding: bool = False,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 256, 128, 64]
        if encoder_kernels is None:
            encoder_kernels = [11, 7, 5, 5, 3, 3]
        if encoder_strides is None:
            encoder_strides = [4, 4, 2, 2, 2, 2]

        assert len(encoder_channels) == len(encoder_kernels) == len(encoder_strides), \
            "encoder_channels, encoder_kernels, and encoder_strides must have same length"

        pad_mode = 'circular' if circular_padding else 'zeros'

        layers = []
        in_ch = 1
        for out_ch, k, s in zip(encoder_channels, encoder_kernels, encoder_strides):
            pad = k // 2
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad,
                                    padding_mode=pad_mode))
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.encoder = nn.Sequential(*layers)
        self.output_dim = encoder_channels[-1]

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply CNN + GAP to produce feature vectors.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] in identity filtration format:
            [[vec_0, vec_1, ...]] where each vec has shape (output_dim,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, H, W)

        encoded = self.encoder(x)  # (batch, C, h, w)
        pooled = encoded.mean(dim=[2, 3])  # (batch, C) — Global Average Pooling

        vectors = [pooled[i] for i in range(pooled.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        channels = []
        for m in self.encoder:
            if isinstance(m, nn.Conv2d):
                channels.append(m.out_channels)
        return (f"CNNGAPFiltration(channels={channels}, "
                f"output_dim={self.output_dim}, "
                f"num_params={self.get_num_parameters()})")
