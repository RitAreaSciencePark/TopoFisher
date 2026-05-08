"""
Peak counts filtration for 2D convergence maps.

Counts local maxima (pixels strictly greater than all 8 neighbours) and bins
their heights into a histogram.  Non-learnable, deterministic.

Peak counts capture non-Gaussian information beyond the 2-point power
spectrum: over-dense structures (halos) produce high-κ peaks whose abundance
depends non-linearly on σ₈ and Ωₘ.

Pipeline:
  Input (H×W) → find local maxima → histogram of peak heights → n_bins vector

Designed for use with IdentityVectorization and MOPED compression.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakCountsFiltration(nn.Module):
    """
    Peak counts filtration for 2D convergence maps.

    A pixel is a local maximum if it is strictly greater than all 8 of its
    immediate neighbours.  Peak heights are binned into a fixed histogram.

    Args:
        n_bins: number of histogram bins (default 20)
        vmin:   lower edge of the first bin (default -0.05)
        vmax:   upper edge of the last bin  (default 0.20)
        log_counts: apply log1p to bin counts for Gaussianization (default True)
        border: number of border pixels to ignore on each side (default 1)

    Output dimension: n_bins

    Example::
        >>> filt = PeakCountsFiltration(n_bins=20, vmin=-0.05, vmax=0.20)
        >>> x = torch.randn(100, 512, 512)
        >>> out = filt(x)   # [[v0, v1, ..., v99]], each vi has shape (20,)
    """

    def __init__(
        self,
        n_bins: int = 20,
        vmin: float = -0.05,
        vmax: float = 0.20,
        log_counts: bool = True,
        border: int = 1,
    ):
        super().__init__()
        self.n_bins = int(n_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.log_counts = log_counts
        self.border = int(border)
        self.output_dim = self.n_bins

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute peak-count histograms.

        Args:
            x: (B, H, W) or (H, W) tensor of convergence maps

        Returns:
            List[List[Tensor]] in identity filtration format:
            [[h_0, h_1, ...]] where each h_i has shape (n_bins,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # --- find local maxima ---
        # unfold extracts 3×3 patches: shape (B, 9, H*W)
        patches = F.unfold(
            x.unsqueeze(1).float(),
            kernel_size=3,
            padding=1,
        )  # (B, 9, H*W)

        center    = patches[:, 4, :]                                        # (B, H*W)
        neighbors = torch.cat([patches[:, :4, :], patches[:, 5:, :]], dim=1)  # (B, 8, H*W)
        is_peak   = (center.unsqueeze(1) > neighbors).all(dim=1)            # (B, H*W)

        # Mask out the border pixels (reshape to B×H×W, zero border, flatten back)
        if self.border > 0:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
            b = self.border
            mask[:, b:H - b, b:W - b] = True
            is_peak = is_peak & mask.view(B, H * W)

        # --- histogram per sample ---
        results: List[torch.Tensor] = []
        for i in range(B):
            heights = center[i][is_peak[i]]
            hist = torch.histc(
                heights,
                bins=self.n_bins,
                min=self.vmin,
                max=self.vmax,
            ).to(dtype=dtype, device=device)
            if self.log_counts:
                hist = torch.log1p(hist)
            results.append(hist)

        return [results]

    def get_num_parameters(self) -> int:
        return 0

    def __repr__(self):
        return (
            f"PeakCountsFiltration(n_bins={self.n_bins}, "
            f"vmin={self.vmin}, vmax={self.vmax}, "
            f"log_counts={self.log_counts}, border={self.border})"
        )
