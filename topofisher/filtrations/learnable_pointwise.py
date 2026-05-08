"""
Learnable pointwise filtration — per-pixel affine transformation.

Instead of a CNN, each pixel has its own learnable scale and bias:
    output[i,j] = input[i,j] * weight[i,j] + bias[i,j]

This gives cubical persistence full control: gradients from the Fisher
loss flow directly to per-pixel parameters without any CNN inductive bias
(no spatial convolution, no weight sharing, no activation functions).

For a 64×64 GRF field: 2 × 64² = 8,192 parameters (scale + bias).
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class PointwiseTransform(nn.Module):
    """Per-pixel affine transformation: output = input * weight + bias."""

    def __init__(self, field_size: int = 64, standardize: bool = True):
        super().__init__()
        self.field_size = field_size
        self.standardize = standardize
        # Initialize weight=1, bias=0 (identity transform)
        self.weight = nn.Parameter(torch.ones(field_size, field_size))
        self.bias = nn.Parameter(torch.zeros(field_size, field_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, H, W) or (H, W)
        Returns:
            Transformed field, same shape as input.
        """
        single = x.ndim == 2
        if single:
            x = x.unsqueeze(0)

        # Per-pixel affine: no spatial coupling, no weight sharing
        out = x * self.weight + self.bias

        if self.standardize:
            # Per-sample standardization (needed for fixed-range PI grids)
            b = out.shape[0]
            flat = out.view(b, -1)
            mean = flat.mean(dim=1, keepdim=True).view(b, 1, 1)
            std = flat.std(dim=1, keepdim=True).view(b, 1, 1)
            out = (out - mean) / (std + 1e-8)

        if single:
            out = out.squeeze(0)

        return out


class LearnablePointwiseFiltration(nn.Module):
    """
    Pointwise learnable filtration: per-pixel transform + cubical persistence.

    Pipeline:
        Input (N×N) → PointwiseTransform → Cubical persistence → Diagrams

    Each pixel has its own learnable scale and bias (2N² parameters).
    Persistence gradients flow directly to per-pixel parameters.
    """

    def __init__(
        self,
        homology_dimensions: List[int] = [0, 1],
        field_size: int = 64,
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        skip_k: int = 1,
        standardize: bool = True,
        persistence_backend: str = 'gudhi',
        persistence_construction: str = 'T',
        periodic: bool = False,
    ):
        super().__init__()
        self.homology_dimensions = homology_dimensions
        self.superlevel = superlevel

        self.transform = PointwiseTransform(
            field_size=field_size,
            standardize=standardize,
        )

        self.cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=min_persistence,
            superlevel=superlevel,
            n_jobs=n_jobs,
            skip_k=skip_k,
            backend=persistence_backend,
            construction=persistence_construction,
            periodic=periodic,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        x_transformed = self.transform(x)
        diagrams = self.cubical(x_transformed)
        return diagrams
