"""
Full-resolution CNN + differentiable pixel histogram — non-TDA baseline.

Same skip-connection backbone as cnn_fullres_persistence_v2 (identity init,
673 params), but replaces persistence with a soft histogram of pixel values.

The comparison is:
  - Persistence: topological features (multi-point spatial correlations)
  - Histogram: 1-point PDF (value distribution, no spatial information)

Both have the same backbone, same init, same number of trainable parameters.
The histogram adds zero extra learnable params.

Architecture:
  Input (B, H, W)
  ├─────────────────────────────────────────┐  Skip connection
  │                                         │
  │  Conv2d(1→C, k, s=1, circular) + ReLU  │
  │        × n_layers                       │
  │  Conv2d(C→1, k=1)  [no activation]     │
  │                                         │
  └──────────── + ─────────────────────────-┘
  │
  field = input + cnn(input)    # (B, H, W)
  │
  → per-sample standardize (μ=0, σ=1)
  → soft histogram (Gaussian kernel binning, n_bins bins)
  → n_bins features per sample
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNFullResHistogramFiltration(nn.Module):
    """
    Full-res CNN (skip connection, identity init) + differentiable histogram.

    Args:
        hidden_channels: Backbone intermediate channels (default 8).
        n_layers: Backbone conv layers before the 1×1 projection (default 2).
        kernel_size: Backbone kernel size (default 3).
        n_bins: Number of histogram bins (default 64).
        bin_range: Range of bin centers in standardized units (default 4.0,
                   bins span [-bin_range, +bin_range]).
        sigma: Gaussian kernel bandwidth (default: auto = bin_width).
        standardize: Per-sample standardize before histogram (default True).
        init_scale: Std of CNN weight initialization (default 0.1).
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        n_layers: int = 2,
        kernel_size: int = 3,
        n_bins: int = 64,
        bin_range: float = 4.0,
        sigma: Optional[float] = None,
        standardize: bool = True,
        init_scale: float = 0.1,
    ):
        super().__init__()

        self.standardize = standardize
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._init_scale = init_scale
        self._n_bins = n_bins
        self._bin_range = bin_range

        # Bin centers: evenly spaced in [-bin_range, +bin_range]
        centers = torch.linspace(-bin_range, bin_range, n_bins)
        self.register_buffer('bin_centers', centers)

        # Bandwidth: default to bin width
        bin_width = (2 * bin_range) / (n_bins - 1) if n_bins > 1 else 1.0
        self._sigma = sigma if sigma is not None else bin_width
        self._sigma_sq = self._sigma ** 2

        # === Backbone: stride-1 CNN with circular padding (identical to persistence v2) ===
        pad = kernel_size // 2
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers.append(nn.Conv2d(
                in_ch, hidden_channels, kernel_size,
                stride=1, padding=pad, padding_mode='circular',
            ))
            layers.append(nn.ReLU(inplace=True))
            in_ch = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, 1, 1))
        self.cnn = nn.Sequential(*layers)

        self.output_dim = n_bins

        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        total = sum(p.numel() for p in self.cnn.parameters())
        print(f"  Identity init (σ={scale}): CNN has {total} params, "
              f"histogram n_bins={self._n_bins}, σ_kernel={self._sigma:.4f}",
              flush=True)

    def _soft_histogram(self, values: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft histogram via Gaussian kernel binning.

        Uses gradient checkpointing: groups of bins are recomputed during backward
        instead of storing all intermediates. This reduces peak memory from
        O(n_bins × B × N) to O(chunk_size × B × N).

        Args:
            values: (B, N) flat pixel values

        Returns:
            hist: (B, n_bins) normalized histogram
        """
        from torch.utils.checkpoint import checkpoint

        B = values.shape[0]
        chunk_size = 8  # Recompute in groups of 8 bins

        def _compute_chunk(vals, start_idx):
            """Compute histogram for a chunk of bins."""
            end_idx = min(start_idx + chunk_size, self._n_bins)
            centers = self.bin_centers[start_idx:end_idx]  # (C,)
            # vals: (B, N), centers: (C,) → diff: (B, N, C)
            diff = vals.unsqueeze(-1) - centers.unsqueeze(0).unsqueeze(0)
            return torch.exp(-0.5 * diff * diff / self._sigma_sq).sum(dim=1)  # (B, C)

        chunks = []
        for start in range(0, self._n_bins, chunk_size):
            start_t = torch.tensor(start, device=values.device)
            chunk = checkpoint(
                _compute_chunk, values, start,
                use_reentrant=False,
            )
            chunks.append(chunk)

        hist = torch.cat(chunks, dim=1)  # (B, n_bins)

        # Normalize to get PDF
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-10)

        return hist

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Skip-connection CNN + soft histogram of pixel values.

        Args:
            x: (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]]: [[hist_0, hist_1, ...]]
                Each hist has shape (n_bins,).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B = x.shape[0]
        x_4d = x.unsqueeze(1)  # (B, 1, H, W)

        # Skip connection: field = input + cnn(input)
        residual = self.cnn(x_4d)
        field = x_4d + residual  # (B, 1, H, W)
        field = field.squeeze(1)  # (B, H, W)

        # Per-sample standardization
        if self.standardize:
            mean = field.mean(dim=(1, 2), keepdim=True)
            std = field.std(dim=(1, 2), keepdim=True)
            field = (field - mean) / (std + 1e-8)

        # Flatten spatial dims → soft histogram
        flat_pixels = field.reshape(B, -1)  # (B, H*W)
        hist = self._soft_histogram(flat_pixels)  # (B, n_bins)

        vectors = [hist[i] for i in range(B)]
        return [vectors]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = sum(p.numel() for p in self.cnn.parameters())
        return (f"CNNFullResHistogramFiltration(\n"
                f"  cnn: {self._n_layers}×Conv2d({self._hidden_channels}ch, "
                f"k={self._kernel_size}, stride=1, circular) + 1×1, "
                f"params={total}\n"
                f"  skip connection: field = input + cnn(input)\n"
                f"  init_scale={self._init_scale}\n"
                f"  histogram: {self._n_bins} bins, "
                f"range=[{-self._bin_range}, {self._bin_range}], "
                f"σ={self._sigma:.4f}\n"
                f"  standardize={self.standardize}\n"
                f")")
