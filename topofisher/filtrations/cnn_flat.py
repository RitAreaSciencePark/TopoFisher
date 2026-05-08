"""
CNN backbone + Flatten — NN baseline counterpart to CNN+Persistence.

Uses the same trainable CNN encoder as CNNPersistenceFiltration, but replaces
persistence with a simple spatial flatten.  This provides a fair comparison:
same encoder, same parameter count up to the flatten/persistence split.

Architecture:
  Input (512×512) → CNN encoder (trainable) → feature maps (C × h × w)
  → flatten → C*h*w-dim vector

For slow_stride [8,16,16,16,8] k=[7,5,3,3,3] s=[4,2,2,2,2]:
  After last layer: 8 channels × 8×8 = 512-dim vector

Output format:
    [[vec_0, vec_1, ..., vec_n]]
    Compatible with 'identity' vectorization (single slot).
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNFlatFiltration(nn.Module):
    """
    Trainable CNN encoder + flatten (no pooling).

    Same encoder architecture as CNNPersistenceFiltration, but the spatial
    feature maps are flattened into a single vector instead of being fed
    through persistence.  This is the NN baseline for the TDA comparison.

    Args:
        encoder_channels: Channel dims for the CNN encoder.
        encoder_kernels: Kernel sizes for the CNN encoder.
        encoder_strides: Strides for the CNN encoder.
        circular_padding: Use circular padding in CNN convolutions.
    """

    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        circular_padding: bool = True,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [8, 16, 16, 16, 8]
        if encoder_kernels is None:
            encoder_kernels = [7, 5, 3, 3, 3]
        if encoder_strides is None:
            encoder_strides = [4, 2, 2, 2, 2]

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

        # Compute output dim from strides
        spatial = 512
        for s in encoder_strides:
            spatial = spatial // s
        self.output_dim = encoder_channels[-1] * spatial * spatial

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply CNN encoder + flatten.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]]: [[vec_0, vec_1, ...]]
                Each vec has shape (output_dim,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        encoded = self.encoder(x)       # (batch, C, h, w)
        flat = encoded.flatten(1)       # (batch, C*h*w)

        vectors = [flat[i] for i in range(flat.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        channels = []
        for m in self.encoder:
            if isinstance(m, nn.Conv2d):
                channels.append(m.out_channels)
        return (f"CNNFlatFiltration(channels={channels}, "
                f"output_dim={self.output_dim}, "
                f"num_params={self.get_num_parameters()})")
