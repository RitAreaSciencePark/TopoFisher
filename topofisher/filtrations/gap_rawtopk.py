"""
CNN+GAP with precomputed TopK persistence concatenation.

Combines the slow_stride CNN+GAP encoder (trainable) with precomputed
TopK cubical persistence features (non-learnable). The CNN processes
the spatial field while TopK features are extracted from extra rows
packed into the input tensor by the pipeline's data loader.

Input format: (batch, H + n_extra_rows, W) where:
  - Rows [:H] contain the 2D field (passed to CNN+GAP)
  - Rows [H:] contain packed TopK features (flattened birth/death pairs)

Output format (IdentityVectorization compatible):
  [[vec_0, vec_1, ...]]  where each vec has shape (gap_dim + topk_dim,)

The TopK features are:
  - H0: top-k birth/death pairs, sorted by descending persistence
  - H1: top-k birth/death pairs, sorted by descending persistence
  - Flattened: [h0_b0, h0_d0, h0_b1, h0_d1, ..., h1_b0, h1_d0, ...]
  - Total dimension: topk_k * 2 * n_homology_dims (default: 50*2*2 = 200)
"""
from typing import List, Optional

import torch
import torch.nn as nn

from .cnn_gap import CNNGAPFiltration


class GAPRawTopKFiltration(nn.Module):
    """
    CNN+GAP encoder concatenated with precomputed TopK persistence features.

    During forward(), splits input into spatial field and packed TopK rows,
    runs CNN+GAP on the field, and concatenates the GAP vector with
    flattened TopK features.

    Args:
        topk_k: Number of TopK persistence features per homology dimension.
        n_homology_dims: Number of homology dimensions (default 2: H0 + H1).
        field_h: Height of the spatial field (to split from TopK rows).
        encoder_channels: CNN channel dimensions (passed to CNNGAPFiltration).
        encoder_kernels: CNN kernel sizes (passed to CNNGAPFiltration).
        encoder_strides: CNN strides (passed to CNNGAPFiltration).
        circular_padding: Use circular padding in CNN.
    """

    def __init__(
        self,
        topk_k: int = 50,
        n_homology_dims: int = 2,
        field_h: int = 512,
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

        self.gap = CNNGAPFiltration(
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            encoder_strides=encoder_strides,
            circular_padding=circular_padding,
        )

        self.topk_k = topk_k
        self.n_homology_dims = n_homology_dims
        self.field_h = field_h
        self.topk_dim = topk_k * 2 * n_homology_dims  # k × (birth,death) × dims
        self.output_dim = self.gap.output_dim + self.topk_dim

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply CNN+GAP on field and concatenate precomputed TopK features.

        Args:
            x: (batch, H + n_extra_rows, W) combined tensor

        Returns:
            [[vec_0, vec_1, ...]] where each vec is (gap_dim + topk_dim,)
        """
        # Split field from TopK rows
        field = x[:, :self.field_h, :]        # (batch, H, W)
        topk_rows = x[:, self.field_h:, :]    # (batch, n_extra_rows, W)

        # Flatten TopK rows and take the first topk_dim values
        topk_flat = topk_rows.reshape(x.shape[0], -1)[:, :self.topk_dim]  # (batch, topk_dim)

        # CNN+GAP on spatial field
        gap_out = self.gap(field)       # [[vec_0, vec_1, ...]] each (gap_dim,)
        gap_vectors = gap_out[0]        # list of (gap_dim,) tensors

        # Concatenate GAP + TopK per sample
        combined = []
        for i in range(len(gap_vectors)):
            combined.append(
                torch.cat([gap_vectors[i], topk_flat[i]], dim=0)
            )

        return [combined]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        gap_repr = repr(self.gap)
        return (
            f"GAPRawTopKFiltration(\n"
            f"  gap={gap_repr},\n"
            f"  topk_k={self.topk_k}, n_homology_dims={self.n_homology_dims},\n"
            f"  topk_dim={self.topk_dim}, output_dim={self.output_dim},\n"
            f"  field_h={self.field_h}\n"
            f")"
        )
