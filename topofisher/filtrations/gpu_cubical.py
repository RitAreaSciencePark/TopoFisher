"""
GPU-accelerated cubical persistence backend using CMP (CUDA).

This module provides a drop-in replacement for the GUDHI-based persistence
computation in DifferentiableCubicalLayer, using the `cmp` package which
runs cubical persistence entirely on GPU via custom CUDA kernels.

Reference:
    Korkmaz et al., "CuMPerLay: Learning Cubical Multiparameter Persistence
    Vectorizations", ICCV 2025.
    https://github.com/circle-group/cmp

The key function `cubical_persistence_v_2d_full` from the `cmp` package:
  - Input:  (B, C, H, W) float tensor on CUDA
  - Output: (pairs, lengths) where:
      pairs:   (B*C, 2, F, 2) — birth/death values for H0 and H1
      lengths: (B*C, 2)       — number of valid pairs per dim

This module converts that output to TopoFisher's native format:
  List[List[Tensor]] — diagrams[hom_dim][sample] = (n_pairs, 2)

Performance: For a batch of 500 samples on 64×64 grids, CMP performs the
entire persistence computation in a single GPU kernel launch, vs. 500
sequential (or threaded) GUDHI calls on CPU. Expected 10-50× speedup
on the persistence bottleneck.

Installation:
    pip install --no-build-isolation "git+https://github.com/circle-group/cmp.git#egg=cmp"
    Requires: nvcc (CUDA toolkit), gcc-12/g++-12, PyTorch >= 2.0

Usage:
    This module is used as a backend in DifferentiableCubicalLayer when
    backend='cmp_gpu' is specified. It is NOT imported at module level
    to avoid hard dependency — if cmp is not installed, the gudhi backend
    is used instead.
"""
from typing import List, Tuple
import torch
import torch.nn as nn


def is_cmp_available() -> bool:
    """Check if the cmp CUDA persistence package is installed."""
    try:
        import cmp.cubical  # noqa: F401
        return True
    except ImportError:
        return False


class GPUCubicalLayer(nn.Module):
    """
    GPU-accelerated cubical persistence layer using CMP.

    Drop-in replacement for the GUDHI-based topology computation
    in DifferentiableCubicalLayer. Returns diagrams in the same
    List[List[Tensor]] format.

    Key differences from GUDHI backend:
    - Runs entirely on GPU (no CPU↔GPU transfer for persistence)
    - Batched: processes all samples in one kernel launch
    - Always computes both H0 and H1 (cmp's _full variant)
    - No skip-K caching needed (GPU is fast enough)
    - Gradients flow through the same indexing trick as GUDHI backend

    The CMP library uses V-construction for cubical complexes and
    computes persistence pairs via union-find on GPU. The birth/death
    values are gathered from the original input tensor, preserving
    the autograd graph.
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: List[float] = None,
        superlevel: bool = False,
        threshold: float = 1e10,
        buffer_size: int = 512,
    ):
        """
        Initialize GPU cubical persistence layer.

        Args:
            homology_dimensions: List of homology dimensions to compute.
                Must be subset of [0, 1] (CMP supports 2D only).
            min_persistence: Minimum persistence threshold per dimension.
            superlevel: If True, negate input for superlevel filtration.
            threshold: CMP internal threshold (default 1e10, effectively ∞).
            buffer_size: CMP internal buffer for persistence pairs.
        """
        super().__init__()

        # Validate: CMP only supports 2D cubical (H0 and H1)
        for d in homology_dimensions:
            if d not in (0, 1):
                raise ValueError(
                    f"CMP GPU backend only supports homology dimensions 0 and 1, got {d}. "
                    f"Use backend='gudhi' for higher dimensions."
                )

        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence or [0.0] * len(self.dimensions)
        self.superlevel = superlevel
        self.threshold = threshold
        self.buffer_size = buffer_size

        # Lazy import — validated at first forward() call
        self._cmp_cubical = None

    def _ensure_cmp(self):
        """Lazy-load cmp.cubical on first use."""
        if self._cmp_cubical is None:
            try:
                import cmp.cubical_v as cubical_v
                self._cmp_cubical = cubical_v
            except ImportError:
                try:
                    import cmp.cubical as cubical
                    self._cmp_cubical = cubical
                except ImportError:
                    raise ImportError(
                        "CMP GPU persistence package not found. Install with:\n"
                        "  pip install --no-build-isolation "
                        '"git+https://github.com/circle-group/cmp.git#egg=cmp"\n'
                        "Or use backend='gudhi' instead."
                    )

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute differentiable persistence diagrams on GPU.

        Args:
            X: Input tensor of shape (n_samples, H, W) on CUDA device.
                If (H, W), a batch dimension is added.

        Returns:
            List[List[Tensor]]: diagrams[hom_dim][sample] = (n_pairs, 2)
                Same format as DifferentiableCubicalLayer.forward()
        """
        self._ensure_cmp()

        # Handle single sample
        if X.ndim == 2:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # For superlevel filtration, negate values
        if self.superlevel:
            X = -X

        n_samples = X.shape[0]
        device = X.device

        # CMP expects (B, C, H, W) — add channel dim
        X_4d = X.unsqueeze(1)  # (n_samples, 1, H, W)

        # Run GPU persistence — returns (pairs, lengths)
        # pairs: (n_samples, 1, 2, F, 2) with birth/death values
        # lengths: (n_samples, 1, 2) with number of valid pairs per dim
        #
        # CMP's cubical_persistence_v_2d_full computes both H0 and H1
        # in a single call. The values are gathered from the input tensor
        # inside CMP, preserving the autograd graph.
        pairs, lengths = self._cmp_cubical.cubical_persistence_v_2d_full(
            X_4d, threshold=self.threshold, buffer_size=self.buffer_size
        )

        # Convert CMP output to TopoFisher diagram format
        # CMP returns with channel dimension preserved:
        #   pairs:   (B, C, 2, F, 2) — 5D tensor
        #   lengths: (B, C, 2)       — 3D tensor
        # We used C=1 channel, so squeeze the channel dimension to get:
        #   pairs:   (n_samples, 2, F, 2)
        #   lengths: (n_samples, 2)

        if pairs.ndim == 5:
            # (B, C, 2, F, 2) → (B, 2, F, 2) — take channel 0
            pairs = pairs[:, 0]
            lengths = lengths[:, 0]
        elif pairs.shape[0] != n_samples:
            # CMP merged batch*channel: (B*C, 2, F, 2) with C>1
            pairs = pairs.view(n_samples, -1, *pairs.shape[1:])[:, 0]
            lengths = lengths.view(n_samples, -1, *lengths.shape[1:])[:, 0]

        # Build output in TopoFisher format: List[List[Tensor]]
        all_diagrams = [[] for _ in self.dimensions]

        for dim_idx, dim in enumerate(self.dimensions):
            # CMP dimension index: 0=H0, 1=H1
            cmp_dim = dim  # Direct mapping for 2D cubical

            for i in range(n_samples):
                n_valid = int(lengths[i, cmp_dim].item())

                if n_valid > 0:
                    # Extract valid pairs: (n_valid, 2) with [birth, death]
                    dgm = pairs[i, cmp_dim, :n_valid, :]

                    # Apply minimum persistence threshold
                    min_pers = self.min_persistence[dim_idx]
                    if min_pers > 0:
                        persistence = torch.abs(dgm[:, 1] - dgm[:, 0])
                        dgm = dgm[persistence > min_pers]

                    all_diagrams[dim_idx].append(dgm)
                else:
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=X.dtype)
                    )

        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams
