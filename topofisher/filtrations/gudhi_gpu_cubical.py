"""
GPU-accelerated cubical persistence backend using GUDHI's CUDA extension.

This module provides a GPU-based cubical persistence layer using the GUDHI
fork with CUDA kernels (branch: gpu-cubical-personal-20260226). It serves
as an alternative to:
  - 'gudhi' backend: CPU-based, thread-parallel via GIL release
  - 'cmp_gpu' backend: GPU-based via CMP CUDA kernels

The GUDHI GPU extension computes persistence on CUDA and returns birth/death
values as numpy arrays. Differentiability is achieved by vectorized value-
matching: birth/death values are matched to input tensor elements in bulk
using argsort + searchsorted, then diagrams are reconstructed via torch.gather
to preserve the autograd graph.

Supports two cubical complex constructions:
  - T-construction (default): Input pixels = top-dimensional cells (squares).
    Uses _persistence_on_rectangles_from_top_cells_gpu_batched.
  - V-construction: Input pixels = vertices. The complex is built by
    embedding vertex values onto all cells via max of incident vertices,
    producing a (2h-1, 2w-1) complex. Uses
    _persistence_on_rectangles_from_vertices_gpu_batched.
    V-construction typically yields higher Fisher information for the same
    input because vertices participate in more persistence pairings.

Requirements:
    - GUDHI fork with GPU extension (gudhi._pers_cub_low_dim_gpu_ext.__backend__ == 'cuda')
    - CUDA device available via torch
    - Install: pip install from source with -DWITH_GUDHI_CUBICAL_GPU=ON

Usage:
    This module is used as a backend in DifferentiableCubicalLayer when
    backend='gudhi_gpu' is specified. Not imported at module level.
"""
from typing import List
import gc
import torch
import torch.nn as nn
import numpy as np


def is_gudhi_gpu_available() -> bool:
    """Check if GUDHI GPU extension is available and has CUDA backend."""
    try:
        from gudhi._pers_cub_low_dim_gpu_ext import __backend__
        return __backend__ == 'cuda'
    except ImportError:
        return False


class GUDHIGPUCubicalLayer(nn.Module):
    """
    GPU-accelerated cubical persistence layer using GUDHI CUDA extension.

    Uses the fork's batched GPU persistence function which processes all
    samples in a single call. Returns numpy birth/death arrays which are
    then matched back to input tensor values via batched GPU operations
    (torch.argsort + torch.searchsorted + torch.gather) for differentiable
    diagram construction — all matching stays on GPU.

    Supports two constructions:
    - construction='T' (default): T-construction. Input (B, H, W) treated
      as top-dimensional cells. Same semantics as GUDHI CPU default.
    - construction='V': V-construction. Input (B, H, W) treated as vertices.
      Produces diagrams equivalent to GUDHI CPU with vertices= parameter,
      and to CMP GPU (which always uses V-construction). Typically yields
      higher Fisher information (~11.3 vs ~10.4 for GRF B=0 at N=64).

    Differences from CMP backend:
    - Uses GUDHI's persistence algorithm (may differ in pairing details)
    - Batched GPU call via GUDHI CUDA extension
    - Value-matching for differentiability (batched GPU torch ops, not per-sample numpy)

    Differences from GUDHI CPU backend:
    - GPU-accelerated persistence computation
    - No cofaces_of_persistence_pairs() — uses value matching instead
    - No threading needed (GPU handles parallelism)
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: List[float] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        construction: str = 'T',
        sub_batch_size: int = 100,
        periodic: bool = False,
    ):
        super().__init__()

        max_dim = 2 if periodic else 1
        for d in homology_dimensions:
            if d not in range(max_dim + 1):
                allowed = list(range(max_dim + 1))
                raise ValueError(
                    f"GUDHI GPU backend supports homology dimensions {allowed}"
                    f" (periodic={periodic}), got {d}."
                )

        if construction not in ('T', 'V'):
            raise ValueError(
                f"construction must be 'T' or 'V', got '{construction}'"
            )

        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence or [0.0] * len(self.dimensions)
        self.superlevel = superlevel
        self.construction = construction
        self.sub_batch_size = sub_batch_size
        self.periodic = periodic
        self._gpu_func = None
        self._gpu_func_batched = None

    def _ensure_backend(self):
        """Lazy-load the GUDHI GPU extension."""
        if self._gpu_func is not None:
            return
        try:
            from gudhi._pers_cub_low_dim_gpu_ext import __backend__
            if __backend__ != 'cuda':
                raise RuntimeError(
                    f"GUDHI GPU extension loaded but backend is '{__backend__}', not 'cuda'. "
                    f"Rebuild GUDHI with -DWITH_GUDHI_CUBICAL_GPU=ON and CUDA toolkit."
                )
            if self.periodic:
                if self.construction == 'V':
                    from gudhi._pers_cub_low_dim_gpu_ext import (
                        _persistence_on_rectangle_periodic_from_vertices_gpu,
                        _persistence_on_rectangles_periodic_from_vertices_gpu_batched,
                    )
                    self._gpu_func = _persistence_on_rectangle_periodic_from_vertices_gpu
                    self._gpu_func_batched = _persistence_on_rectangles_periodic_from_vertices_gpu_batched
                else:
                    from gudhi._pers_cub_low_dim_gpu_ext import (
                        _persistence_on_rectangle_periodic_from_top_cells_gpu,
                        _persistence_on_rectangles_periodic_from_top_cells_gpu_batched,
                    )
                    self._gpu_func = _persistence_on_rectangle_periodic_from_top_cells_gpu
                    self._gpu_func_batched = _persistence_on_rectangles_periodic_from_top_cells_gpu_batched
            elif self.construction == 'V':
                from gudhi._pers_cub_low_dim_gpu_ext import (
                    _persistence_on_rectangle_from_vertices_gpu,
                    _persistence_on_rectangles_from_vertices_gpu_batched,
                )
                self._gpu_func = _persistence_on_rectangle_from_vertices_gpu
                self._gpu_func_batched = _persistence_on_rectangles_from_vertices_gpu_batched
            else:
                from gudhi._pers_cub_low_dim_gpu_ext import (
                    _persistence_on_rectangle_from_top_cells_gpu,
                    _persistence_on_rectangles_from_top_cells_gpu_batched,
                )
                self._gpu_func = _persistence_on_rectangle_from_top_cells_gpu
                self._gpu_func_batched = _persistence_on_rectangles_from_top_cells_gpu_batched
        except ImportError:
            raise ImportError(
                "GUDHI GPU extension not available. Install GUDHI fork with CUDA support:\n"
                "  CMAKE_ARGS='-DWITH_GUDHI_CUBICAL_GPU=ON' pip install .\n"
                "Or use backend='gudhi' (CPU) or backend='cmp_gpu' instead."
            )

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute differentiable persistence diagrams using GUDHI GPU.

        Processes the batch in sub-batches of size `self.sub_batch_size`
        to avoid CPU memory blowup from large intermediate arrays
        (float64 copies, argsort indices, sorted values). For a batch of
        B=5000 samples of 512×512, processing all at once would require
        ~55 GB of CPU RAM; sub-batching at 500 reduces this to ~6 GB.

        Args:
            X: Input tensor of shape (n_samples, H, W) or (H, W).

        Returns:
            List[List[Tensor]]: diagrams[hom_dim][sample] = (n_pairs, 2)
        """
        self._ensure_backend()

        if X.ndim == 2:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        if self.superlevel:
            X = -X

        B = X.shape[0]

        if B <= self.sub_batch_size:
            all_diagrams = self._process_batch(X)
        else:
            n_chunks = (B + self.sub_batch_size - 1) // self.sub_batch_size
            if not hasattr(self, '_sub_batch_logged'):
                print(f"    GPU persistence: sub-batching {B} samples into "
                      f"{n_chunks} chunks of {self.sub_batch_size}", flush=True)
                self._sub_batch_logged = True
            all_diagrams = [[] for _ in self.dimensions]
            for start in range(0, B, self.sub_batch_size):
                end = min(start + self.sub_batch_size, B)
                chunk = X[start:end]
                chunk_diagrams = self._process_batch(chunk)
                for dim_idx in range(len(self.dimensions)):
                    all_diagrams[dim_idx].extend(chunk_diagrams[dim_idx])
                del chunk_diagrams
                gc.collect()

        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams

    def _process_batch(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Process a single sub-batch through GPU persistence + differentiable
        value-matching.

        Strategy:
        1. Convert batch to numpy (B, H, W), call batched GPU function
        2. Batched argsort on GPU (single torch.argsort for entire batch)
        3. Pad diagrams in numpy, single transfer to GPU
        4. Batched torch.searchsorted on GPU for value-matching
        5. Batched torch.gather for differentiable diagram construction

        X must already have superlevel negation applied if needed.

        Args:
            X: Input tensor of shape (B, H, W), already preprocessed.

        Returns:
            List[List[Tensor]]: diagrams[hom_dim][sample] = (n_pairs, 2)
        """
        B, H, W = X.shape
        N = H * W
        device = X.device
        dtype = X.dtype

        # === Step 1: GUDHI GPU persistence (numpy required) ===
        X_np = X.detach().cpu().numpy().astype(np.float64)
        raw_diagrams = self._gpu_func_batched(X_np, 0.0)
        del X_np

        # === Step 2: Batched argsort on GPU ===
        X_flat = X.reshape(B, -1)  # (B, N) — stays on device with grad
        sort_indices = torch.argsort(X_flat.detach(), dim=1)  # (B, N) int64
        sorted_X = torch.gather(X_flat.detach().double(), 1, sort_indices)  # (B, N) float64

        # === Step 3: Per-dimension batched matching ===
        all_diagrams = [[] for _ in self.dimensions]

        for dim_idx, dim in enumerate(self.dimensions):
            min_pers = self.min_persistence[dim_idx]

            # Extract and filter diagrams (lightweight numpy loop)
            dgm_list = []
            n_pairs_list = []
            for i in range(B):
                dgm_np = raw_diagrams[i][dim]  # (K_i, 2) numpy float64
                if dgm_np.shape[0] > 0 and min_pers > 0:
                    pers = np.abs(dgm_np[:, 1] - dgm_np[:, 0])
                    dgm_np = dgm_np[pers > min_pers]
                dgm_list.append(dgm_np)
                n_pairs_list.append(dgm_np.shape[0])

            max_pairs = max(n_pairs_list) if n_pairs_list else 0

            if max_pairs == 0:
                for i in range(B):
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype)
                    )
                continue

            # Pad birth/death values in numpy (fast), then single GPU transfer
            birth_np = np.zeros((B, max_pairs), dtype=np.float64)
            death_np = np.zeros((B, max_pairs), dtype=np.float64)
            for i in range(B):
                k = n_pairs_list[i]
                if k > 0:
                    birth_np[i, :k] = dgm_list[i][:, 0]
                    death_np[i, :k] = dgm_list[i][:, 1]

            # Single transfer: (B, 2*max_pairs) float64 on GPU
            all_values = torch.from_numpy(
                np.concatenate([birth_np, death_np], axis=1)
            ).to(device)  # (B, 2*max_pairs)
            del birth_np, death_np

            # === Batched searchsorted on GPU ===
            insert_pos = torch.searchsorted(sorted_X, all_values)  # (B, 2*max_pairs)
            insert_pos = insert_pos.clamp(0, N - 1)
            right_pos = (insert_pos + 1).clamp(0, N - 1)

            # Compare left vs right neighbor (batched)
            left_vals = torch.gather(sorted_X, 1, insert_pos)
            right_vals = torch.gather(sorted_X, 1, right_pos)
            left_better = (left_vals - all_values).abs() <= (right_vals - all_values).abs()
            best_pos = torch.where(left_better, insert_pos, right_pos)
            del all_values, insert_pos, right_pos, left_vals, right_vals, left_better

            # Map sorted positions → original indices (batched)
            matched_indices = torch.gather(sort_indices, 1, best_pos)  # (B, 2*max_pairs)
            del best_pos

            b_indices = matched_indices[:, :max_pairs]   # (B, max_pairs)
            d_indices = matched_indices[:, max_pairs:]    # (B, max_pairs)
            del matched_indices

            # Differentiable gather from original tensor (batched)
            births = torch.gather(X_flat, 1, b_indices)  # (B, max_pairs) with grad
            deaths = torch.gather(X_flat, 1, d_indices)  # (B, max_pairs) with grad
            del b_indices, d_indices

            # Build (B, max_pairs, 2) diagram tensor in one op, then split by valid count
            dgm_batched = torch.stack([births, deaths], dim=2)  # (B, max_pairs, 2)
            del births, deaths

            # Unpad: split into per-sample diagrams using narrow (no GPU kernel per sample)
            for i in range(B):
                k = n_pairs_list[i]
                if k == 0:
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype)
                    )
                else:
                    all_diagrams[dim_idx].append(dgm_batched[i, :k].clone())  # clone to release dgm_batched

            del dgm_batched

        del raw_diagrams, sort_indices, sorted_X

        return all_diagrams
