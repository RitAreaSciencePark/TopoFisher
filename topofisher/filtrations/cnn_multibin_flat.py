"""
CNN + Flatten filtration for multi-bin 2D fields.

A deep strided CNN encoder that takes all tomographic bins as input channels
(e.g. 5 lensing bins → 5S input channels), reduces to an intermediate
multi-channel feature map via a 1×1 projection, and flattens to a fixed-size
feature vector. No Global Average Pooling — the full spatial structure at the
final resolution is preserved.

This serves as the NN baseline for comparison with CNN + 3D persistence.

Architecture (default, 5 bins):
  Input (B, 5, 512, 512)
  → Conv1: 5→8,   k=7, s=4, circular →  128×128
  → Conv2: 8→16,  k=5, s=2, circular →  64×64
  → Conv3: 16→16, k=3, s=2, circular →  32×32
  → Conv4: 16→16, k=3, s=2, circular →  16×16
  → Conv5: 16→8,  k=3, s=2, circular →  8×8
  → 1×1 projection: 8→5, no activation → 8×8
  → Flatten → 320-dim vector

Output format: [[vec_0, vec_1, ...]] (identity filtration format)
Designed for use with IdentityVectorization and MOPED compression.
Matched output dimension (~320) with CNN + 3D persistence for fair comparison.
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNMultiBinFlatFiltration(nn.Module):
    """
    Multi-bin CNN encoder with 1×1 projection and flatten.

    Takes all tomographic bins as input channels, processes through a shared
    CNN backbone, projects to `projection_channels` via a 1×1 conv (no
    activation), and flattens the spatial map to a feature vector.

    Args:
        encoder_channels: List of output channels per conv layer.
        encoder_kernels: List of kernel sizes per conv layer.
        encoder_strides: List of strides per conv layer.
        projection_channels: Number of output channels for 1×1 projection.
        circular_padding: Use circular padding for periodic fields.
        n_input_channels: Number of input channels (tomographic bins).
    """

    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        projection_channels: int = 5,
        circular_padding: bool = True,
        n_input_channels: int = 5,
        checkpoint_path: Optional[str] = None,
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

        self._encoder_channels = encoder_channels
        self._encoder_strides = encoder_strides
        self._projection_channels = projection_channels
        self._n_input_channels = n_input_channels

        pad_mode = 'circular' if circular_padding else 'zeros'

        # Build encoder
        layers = []
        in_ch = n_input_channels
        for out_ch, k, s in zip(encoder_channels, encoder_kernels, encoder_strides):
            pad = k // 2
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad,
                                    padding_mode=pad_mode))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.encoder = nn.Sequential(*layers)

        # 1×1 projection to projection_channels (no activation)
        self.projection = nn.Conv2d(encoder_channels[-1], projection_channels, 1)

        # Compute output spatial size for 512×512 input
        spatial = 512
        for s in encoder_strides:
            spatial = spatial // s
        self._output_spatial = spatial
        self.output_dim = projection_channels * spatial * spatial

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply multi-bin CNN + 1×1 projection + flatten.

        Args:
            x: Input tensor of shape (batch, n_bins, H, W)

        Returns:
            List[List[Tensor]] in identity filtration format:
            [[vec_0, vec_1, ...]] where each vec has shape (output_dim,)
        """
        if x.dim() == 2:
            # Single sample, single channel (H, W) → (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            # Batch of single-channel (B, H, W) → (B, 1, H, W)
            x = x.unsqueeze(1)

        # x is (B, n_bins, H, W) — n_bins acts as channels
        encoded = self.encoder(x)          # (B, C_last, h, w)
        projected = self.projection(encoded)  # (B, proj_ch, h, w)
        flat = projected.reshape(projected.shape[0], -1)  # (B, proj_ch * h * w)

        vectors = [flat[i] for i in range(flat.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        spatial = self._output_spatial
        return (f"CNNMultiBinFlatFiltration(\n"
                f"  encoder: {self._n_input_channels}→"
                f"{'→'.join(str(c) for c in self._encoder_channels)}, "
                f"strides={self._encoder_strides}\n"
                f"  projection: {self._encoder_channels[-1]}→"
                f"{self._projection_channels} (1×1, no activation)\n"
                f"  output: flatten({self._projection_channels}×{spatial}×{spatial}) "
                f"= {self.output_dim}-dim\n"
                f"  params: {self.get_num_parameters()}\n"
                f")")
