"""
CNN + 3D cubical persistence for multi-bin 2D fields.

A deep strided CNN encoder takes all tomographic bins as input channels
(e.g. 5 lensing bins), reduces to an intermediate multi-channel feature
map via a 1×1 projection, then applies 3D cubical persistence where the
third axis is the bin (redshift) dimension.

This captures cross-bin topological structure: features that persist
across multiple bins correspond to structures present at multiple source
redshifts — cosmologically meaningful information.

Architecture (default, 5 bins):
  Input (B, 5, 512, 512)
  → Conv1: 5→8,   k=7, s=4, circular →  128×128
  → Conv2: 8→16,  k=5, s=2, circular →  64×64
  → Conv3: 16→16, k=3, s=2, circular →  32×32
  → Conv4: 16→16, k=3, s=2, circular →  16×16
  → Conv5: 16→8,  k=3, s=2, circular →  8×8
  → 1×1 projection: 8→5, no activation → (B, 5, 8, 8)
  → Per-sample standardize
  → GUDHI GPU batched 3D persistence on (B, 5, 8, 8) boxes
  → H0, H1, H2 diagrams per sample

3D non-periodic: GUDHI GPU extension treats (B, C, H, W) as batched 3D
boxes of shape (C, H, W). The bin axis C is physically non-periodic.
CNN circular padding handles spatial periodicity during downsampling.

Differentiability: Uses the GPU value-matching pattern (argsort +
searchsorted + gather) from CNNPersistenceFiltration. GPU persistence
returns numpy birth/death arrays; these are matched back to the input
tensor values on GPU for autograd compatibility.

Output format:
    [[H0_diag_0, ...], [H1_diag_0, ...], [H2_diag_0, ...]]
    Compatible with 'combined' vectorization (3 sub-vectorizers).
"""
from typing import List, Optional
import os
import torch
import torch.nn as nn
import numpy as np


class CNN3DPersistenceFiltration(nn.Module):
    """
    Multi-bin CNN encoder + 3D cubical persistence (GPU).

    Same CNN backbone as CNNMultiBinFlatFiltration, but instead of
    flattening the projected feature map, applies 3D cubical persistence
    where the third dimension is the bin/redshift axis.

    Uses GUDHI GPU ``_persistence_on_boxes_from_top_cells_gpu_batched``
    for the persistence computation (same as CNNPersistenceFiltration).

    Args:
        encoder_channels: List of output channels per conv layer.
        encoder_kernels: List of kernel sizes per conv layer.
        encoder_strides: List of strides per conv layer.
        projection_channels: Number of output channels for 1×1 projection
            (corresponds to the bin/redshift dimension in 3D persistence).
        circular_padding: Use circular padding for periodic fields.
        n_input_channels: Number of input channels (tomographic bins).
        homology_dimensions: Homology dimensions for persistence (default [0, 1, 2]).
        skip_k: Recompute topology every skip_k forward passes.
        standardize: Per-sample standardize before persistence.
        sub_batch_size: Max samples per GPU persistence call.
    """

    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        projection_channels: int = 5,
        circular_padding: bool = True,
        n_input_channels: int = 5,
        homology_dimensions: List[int] = [0, 1, 2],
        n_jobs: int = -1,
        skip_k: int = 1,
        standardize: bool = True,
        sub_batch_size: int = 200,
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
        self.dimensions = homology_dimensions
        self.min_persistence = [0.0] * len(homology_dimensions)
        self.skip_k = max(1, skip_k)
        self.standardize = standardize
        self.sub_batch_size = sub_batch_size

        pad_mode = 'circular' if circular_padding else 'zeros'

        # Build encoder (identical to CNNMultiBinFlatFiltration)
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

        # GPU persistence backend (lazy-loaded)
        self._gpu_func_batched = None

        # Skip-K topology caching
        self._call_count = 0
        self._cached_raw_diagrams = None
        self._cached_n_samples = 0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_gpu_backend(self):
        """Lazy-load GUDHI GPU 3D non-periodic persistence."""
        if self._gpu_func_batched is not None:
            return
        from gudhi._pers_cub_low_dim_gpu_ext import __backend__
        from gudhi._pers_cub_low_dim_gpu_ext import (
            _persistence_on_boxes_from_top_cells_gpu_batched,
        )
        self._gpu_func_batched = _persistence_on_boxes_from_top_cells_gpu_batched
        if not hasattr(self, '_gpu_backend_logged'):
            print(f"    3D persistence backend: {__backend__}", flush=True)
            self._gpu_backend_logged = True

    def _should_recompute(self, n_samples: int) -> bool:
        if self.skip_k <= 1 or not self.training:
            return True
        if self._cached_raw_diagrams is None or self._cached_n_samples != n_samples:
            return True
        return (self._call_count % self.skip_k == 0)

    def _process_gpu_batch(self, feat_3d, X_flat, N, device, dtype):
        """Process a sub-batch through GPU persistence + value-matching.

        Follows the same pattern as CNNPersistenceFiltration._process_gpu_batch.
        """
        B = feat_3d.shape[0]

        # 1) GPU persistence (non-differentiable, returns numpy)
        X_np = feat_3d.detach().cpu().numpy().astype(np.float64)
        raw_diagrams = self._gpu_func_batched(X_np, 0.0)
        del X_np

        # 2) Batched argsort on GPU for value-matching
        sort_indices = torch.argsort(X_flat.detach(), dim=1)  # (B, N)
        sorted_X = torch.gather(X_flat.detach().double(), 1, sort_indices)  # (B, N)

        # 3) Per-dimension batched matching
        all_diagrams = [[] for _ in self.dimensions]

        for dim_idx, dim in enumerate(self.dimensions):
            min_pers = self.min_persistence[dim_idx]

            dgm_list = []
            n_pairs_list = []
            for i in range(B):
                dgm_np = raw_diagrams[i][dim]  # (K_i, 2) float64
                if dgm_np.shape[0] > 0 and min_pers > 0:
                    pers = np.abs(dgm_np[:, 1] - dgm_np[:, 0])
                    dgm_np = dgm_np[pers > min_pers]
                dgm_list.append(dgm_np)
                n_pairs_list.append(dgm_np.shape[0])

            max_pairs = max(n_pairs_list) if n_pairs_list else 0
            if max_pairs == 0:
                for _ in range(B):
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype))
                continue

            # Pad birth/death values → single GPU transfer
            birth_np = np.zeros((B, max_pairs), dtype=np.float64)
            death_np = np.zeros((B, max_pairs), dtype=np.float64)
            for i in range(B):
                k = n_pairs_list[i]
                if k > 0:
                    birth_np[i, :k] = dgm_list[i][:, 0]
                    death_np[i, :k] = dgm_list[i][:, 1]

            all_values = torch.from_numpy(
                np.concatenate([birth_np, death_np], axis=1)
            ).to(device)  # (B, 2*max_pairs) float64
            del birth_np, death_np

            # Batched searchsorted on GPU
            insert_pos = torch.searchsorted(sorted_X, all_values).clamp(0, N - 1)
            right_pos = (insert_pos + 1).clamp(0, N - 1)
            left_vals = torch.gather(sorted_X, 1, insert_pos)
            right_vals = torch.gather(sorted_X, 1, right_pos)
            left_better = (left_vals - all_values).abs() <= (right_vals - all_values).abs()
            best_pos = torch.where(left_better, insert_pos, right_pos)
            del all_values, insert_pos, right_pos, left_vals, right_vals, left_better

            # Map sorted → original indices
            matched_indices = torch.gather(sort_indices, 1, best_pos)
            del best_pos
            b_indices = matched_indices[:, :max_pairs]
            d_indices = matched_indices[:, max_pairs:]
            del matched_indices

            # Differentiable gather (autograd flows back through X_flat)
            births = torch.gather(X_flat, 1, b_indices)
            deaths = torch.gather(X_flat, 1, d_indices)
            del b_indices, d_indices

            dgm_batched = torch.stack([births, deaths], dim=2)  # (B, max_pairs, 2)
            del births, deaths

            for i in range(B):
                k = n_pairs_list[i]
                if k == 0:
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype))
                else:
                    all_diagrams[dim_idx].append(dgm_batched[i, :k].clone())
            del dgm_batched

        del raw_diagrams, sort_indices, sorted_X
        return all_diagrams

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply multi-bin CNN + 1×1 projection + 3D cubical persistence (GPU).

        Args:
            x: Input tensor of shape (batch, n_bins, H, W)

        Returns:
            List[List[Tensor]]: [[H0_diags], [H1_diags], [H2_diags]]
                Each list has one diagram per sample, shape (n_pairs, 2).
        """
        self._ensure_gpu_backend()

        if x.dim() == 3:
            x = x.unsqueeze(0)

        # CNN encoder + 1×1 projection
        encoded = self.encoder(x)             # (B, C_last, h, w)
        projected = self.projection(encoded)  # (B, proj_ch, h, w)

        # Per-sample standardization (on the 3D field)
        if self.standardize:
            mean = projected.mean(dim=(1, 2, 3), keepdim=True)
            std = projected.std(dim=(1, 2, 3), keepdim=True)
            projected = (projected - mean) / (std + 1e-8)

        B = projected.shape[0]
        device = projected.device
        dtype = projected.dtype

        # GUDHI GPU treats (B, C, H, W) as batch of 3D (C, H, W) boxes
        # No permute needed — the function handles the memory layout
        feat_3d = projected  # (B, proj_ch, h, w)
        X_flat = projected.reshape(B, -1)  # (B, N) where N = proj_ch*h*w
        N = X_flat.shape[1]

        if B <= self.sub_batch_size:
            return self._process_gpu_batch(feat_3d, X_flat, N, device, dtype)

        # Sub-batching for large datasets
        all_diagrams = [[] for _ in self.dimensions]
        for start in range(0, B, self.sub_batch_size):
            end = min(start + self.sub_batch_size, B)
            chunk_3d = feat_3d[start:end]
            chunk_flat = X_flat[start:end]
            chunk_diags = self._process_gpu_batch(
                chunk_3d, chunk_flat, N, device, dtype
            )
            for dim_idx in range(len(self.dimensions)):
                all_diagrams[dim_idx].extend(chunk_diags[dim_idx])
            del chunk_diags
        return all_diagrams

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        spatial = self._output_spatial
        D = self._projection_channels
        N = spatial * spatial * D
        return (f"CNN3DPersistenceFiltration(\n"
                f"  encoder: {self._n_input_channels}→"
                f"{'→'.join(str(c) for c in self._encoder_channels)}, "
                f"strides={self._encoder_strides}\n"
                f"  projection: {self._encoder_channels[-1]}→"
                f"{D} (1×1, no activation)\n"
                f"  3D cubical (GPU): {spatial}×{spatial}×{D} = {N} cells, "
                f"non-periodic\n"
                f"  homology_dimensions={self.dimensions}, "
                f"standardize={self.standardize}, skip_k={self.skip_k}\n"
                f"  params: {self.get_num_parameters()}\n"
                f")")
