"""
Full-resolution CNN + non-periodic GPU cubical persistence (Phase 6).

Identity-initialized skip connection ensures output ≈ input at epoch 1,
so persistence sees real field topology from the start. GPU non-periodic
persistence gives 204× speedup at 512² vs CPU periodic.

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
  output = input + cnn(input)    # (B, H, W)
  │
  → per-sample standardize
  → 2D non-periodic cubical persistence (GPU, batched)
  → H0 + H1 diagrams
  → DiffCurves vectorization
  → MOPED → Fisher

Justification for non-periodic on periodic fields:
  Circular padding encodes periodicity in CNN output VALUES. Non-periodic
  persistence only misses essential classes (topological invariants of T²,
  constant across all samples → zero Fisher information). All finite
  persistence pairs are preserved.
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNFullResPersistenceV2Filtration(nn.Module):
    """
    Full-res CNN (skip connection, identity init) + GPU non-periodic persistence.

    Args:
        hidden_channels: Number of intermediate CNN channels (default 8).
        n_layers: Number of conv layers before the 1×1 projection (default 2).
        kernel_size: Kernel size for conv layers (default 3).
        homology_dimensions: Homology dims for persistence (default [0, 1]).
        construction: 'T' (top cells) or 'V' (vertices). Default 'T'.
        sub_batch_size: Samples per GPU persistence call (default 100).
        standardize: Per-sample standardize before persistence (default True).
        init_scale: Std of CNN weight initialization (default 0.01).
        periodic: Use periodic boundary conditions in cubical persistence
            (default False — relies on circular padding to encode periodicity
            in CNN output values). Set True to match the gudhi-periodic
            backend used by non-CNN baselines (e.g. perslay_learnable_moped),
            so the diagram distribution at CNN ≈ identity is identical to
            the baseline — required for clean warm-starting.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        n_layers: int = 2,
        kernel_size: int = 3,
        homology_dimensions: Optional[List[int]] = None,
        construction: str = 'T',
        sub_batch_size: int = 100,
        standardize: bool = True,
        init_scale: float = 0.01,
        periodic: bool = False,
    ):
        super().__init__()

        if homology_dimensions is None:
            homology_dimensions = [0, 1]

        self.dimensions = homology_dimensions
        self.standardize = standardize
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._init_scale = init_scale
        self.periodic = periodic

        # Build stride-1 CNN with circular padding
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
        # Final 1×1 projection → single channel (no activation)
        layers.append(nn.Conv2d(hidden_channels, 1, 1))
        self.cnn = nn.Sequential(*layers)

        # GPU cubical persistence layer
        from .gudhi_gpu_cubical import GUDHIGPUCubicalLayer
        self.persistence = GUDHIGPUCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=[0.0] * len(homology_dimensions),
            superlevel=False,
            construction=construction,
            sub_batch_size=sub_batch_size,
            periodic=periodic,
        )

        # Near-zero init → cnn(input) ≈ 0 → output ≈ input
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        for m in self.cnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        total_params = sum(p.numel() for p in self.cnn.parameters())
        print(f"  Identity init (σ={scale}): CNN has {total_params} params, "
              f"output ≈ input at epoch 1", flush=True)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Skip-connection CNN + GPU 2D persistence.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]]: [[H0_diags], [H1_diags]]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # x: (B, H, W)
        x_4d = x.unsqueeze(1)  # (B, 1, H, W)

        # Skip connection: output = input + cnn(input)
        residual = self.cnn(x_4d)       # (B, 1, H, W)
        field = x_4d + residual         # (B, 1, H, W)
        field = field.squeeze(1)        # (B, H, W)

        # Per-sample standardization
        if self.standardize:
            mean = field.mean(dim=(1, 2), keepdim=True)
            std = field.std(dim=(1, 2), keepdim=True)
            field = (field - mean) / (std + 1e-8)

        # GPU persistence (handles batching, value-matching, differentiability)
        return self.persistence(field)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = sum(p.numel() for p in self.cnn.parameters())
        return (f"CNNFullResPersistenceV2Filtration(\n"
                f"  cnn: {self._n_layers}×Conv2d({self._hidden_channels}ch, "
                f"k={self._kernel_size}, stride=1, circular) + 1×1 proj, "
                f"params={total}\n"
                f"  skip connection: output = input + cnn(input)\n"
                f"  init_scale={self._init_scale}\n"
                f"  persistence: GPU {'periodic' if self.periodic else 'non-periodic'}, dims={self.dimensions}\n"
                f"  standardize={self.standardize}\n"
                f")")
