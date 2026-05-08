"""
Power Spectrum filtration for 2D fields.

Computes the radially-averaged power spectrum C_ell of 2D fields as a
summary statistic for Fisher information analysis.

Uses FFT-based computation with log-spaced radial bins. Non-learnable,
deterministic, and computationally cheap.

For convergence maps, C_ell captures all Gaussian 2-point information.
The feature vector is one power spectrum per sample.

Designed for use with IdentityVectorization and MOPED compression.

Pipeline:
  Input (H×W) → FFT → |FFT|^2 → radial binning → n_ell_bins vector
"""
from typing import List, Optional
import math
import torch
import torch.nn as nn
import numpy as np


class PowerSpectrumFiltration(nn.Module):
    """
    Power Spectrum as a topofisher filtration component.

    Non-learnable: computes radially-averaged auto-power spectrum from 2D fields.
    Uses precomputed bin index map for fast binning.

    Args:
        n_ell_bins: Number of log-spaced radial bins (default 80)
        k_min: Minimum wavenumber for binning (default 1.0)
        k_max: Maximum wavenumber, or None for N/2 (default None)
        field_size: Expected spatial size N for precomputing grids.
                    If None, computed from first input (default None).
        log_transform: If True, return log(C_ell) instead of C_ell.
                       This Gaussianizes the chi-squared distributed PS
                       estimates, which is required for valid Fisher analysis.
                       Bins with zero power are excluded (default True).

    Output dimension: n_ell_bins

    Example:
        >>> filt = PowerSpectrumFiltration(n_ell_bins=80, field_size=512)
        >>> x = torch.randn(100, 512, 512)
        >>> out = filt(x)  # [[vec_0, ..., vec_99]], each vec is (80,)
    """

    def __init__(
        self,
        n_ell_bins: int = 80,
        k_min: float = 1.0,
        k_max: Optional[float] = None,
        field_size: Optional[int] = None,
        log_transform: bool = True,
    ):
        super().__init__()
        self.n_ell_bins = n_ell_bins
        self.k_min = k_min
        self.k_max = k_max
        self.log_transform = log_transform
        self.output_dim = n_ell_bins

        # Pre-built grid (lazy initialization if field_size not given)
        self._grid_built = False
        if field_size is not None:
            self._build_grid(field_size)

    def _build_grid(self, N: int):
        """Build wavenumber grid and bin index map for an N×N FFT."""
        k_max = self.k_max if self.k_max is not None else N / 2

        # 2D wavenumber magnitude grid
        kx = torch.fft.fftfreq(N) * N
        ky = torch.fft.fftfreq(N) * N
        kx2d, ky2d = torch.meshgrid(kx, ky, indexing='ij')
        k_mag = torch.sqrt(kx2d**2 + ky2d**2)

        # Log-spaced bin edges
        k_edges = torch.logspace(
            math.log10(self.k_min), math.log10(k_max),
            self.n_ell_bins + 1
        )

        # Bin index map: assign each pixel to a bin (-1 = unassigned)
        bin_idx = torch.full((N, N), -1, dtype=torch.long)
        bin_counts = torch.zeros(self.n_ell_bins, dtype=torch.float64)
        for i in range(self.n_ell_bins):
            mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
            bin_idx[mask] = i
            bin_counts[i] = mask.sum().item()

        # Valid mask (pixels that belong to some bin)
        valid_mask = bin_idx >= 0

        # Register as buffers (move with model to GPU)
        self.register_buffer('_bin_idx', bin_idx)
        self.register_buffer('_bin_counts', bin_counts)
        self.register_buffer('_valid_mask', valid_mask)
        self.register_buffer('_k_edges', k_edges)
        self._N = N
        self._grid_built = True

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute power spectrum features.

        Args:
            x: (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] in identity filtration format:
            [[vec_0, vec_1, ...]] where each vec has shape (n_ell_bins,)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        N = x.shape[-1]
        if not self._grid_built or self._N != N:
            self._build_grid(N)
            # Move buffers to same device as input
            if x.is_cuda:
                self._bin_idx = self._bin_idx.to(x.device)
                self._bin_counts = self._bin_counts.to(x.device)
                self._valid_mask = self._valid_mask.to(x.device)
                self._k_edges = self._k_edges.to(x.device)

        batch = x.shape[0]

        # FFT → power spectrum
        # Normalization: /N^2 for unnormalized DFT (Parseval: Σ|X̃|² = N² Σ|x|²)
        # Gives C_ell values of order Var(field), typically ~1e-4 for κ maps
        fft = torch.fft.fft2(x)
        ps2d = (torch.abs(fft)**2) / (N**2)  # (batch, N, N)

        # Radial binning using precomputed index map
        ps2d_flat = ps2d.reshape(batch, -1)  # (batch, N*N)
        bin_idx_flat = self._bin_idx.reshape(-1)  # (N*N,)
        valid = self._valid_mask.reshape(-1)  # (N*N,)

        ps2d_valid = ps2d_flat[:, valid]  # (batch, n_valid)
        idx_valid = bin_idx_flat[valid]  # (n_valid,)

        # Scatter-add into bins
        ps1d = torch.zeros(batch, self.n_ell_bins, device=x.device, dtype=x.dtype)
        idx_expanded = idx_valid.unsqueeze(0).expand(batch, -1)
        ps1d.scatter_add_(1, idx_expanded, ps2d_valid.to(x.dtype))

        # Normalize by mode counts
        nonzero = self._bin_counts > 0
        ps1d[:, nonzero] /= self._bin_counts[nonzero].to(x.dtype)

        if self.log_transform:
            # log(C_ell) Gaussianizes the chi-squared distributed PS
            # Only keep bins with nonzero mode counts AND nonzero power
            valid = nonzero & (ps1d.mean(dim=0) > 0)
            ps1d_log = torch.log(ps1d[:, valid] + 1e-30)
            vectors = [ps1d_log[i] for i in range(batch)]
        else:
            vectors = [ps1d[i] for i in range(batch)]
        return [vectors]

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters (always 0)."""
        return 0

    def __repr__(self):
        return (f"PowerSpectrumFiltration(n_ell_bins={self.n_ell_bins}, "
                f"k_min={self.k_min}, k_max={self.k_max}, "
                f"log_transform={self.log_transform})")
