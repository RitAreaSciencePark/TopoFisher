"""
CNN backbone + per-channel 2D differentiable cubical persistence (replaces GAP).

Uses a trainable CNN encoder (warm-started from pre-trained checkpoint) and
replaces Global Average Pooling with differentiable cubical persistence applied
independently to each channel of the feature map.

Each CNN output channel (h × w) is treated as a separate 2D cubical complex
with periodic boundary conditions.  Per-channel persistence:
  - Respects channel independence (each channel learns complementary features)
  - Captures spatial topology (H0: components, H1: loops) of each feature map
  - Concatenates per-channel diagrams for each homology dimension

Persistence backend:
  - 'gudhi': Uses CPU PeriodicCubicalComplex(periodic_dimensions=[True,True])
    for 2D periodic fields.  Differentiable via cofaces_of_persistence_pairs.
  - 'gudhi_gpu': Uses GUDHI CUDA extension for batched 2D persistence.

Architecture:
  Input (512×512) → CNN encoder (trainable) → feature maps (C × h × w)
  → standardize → per-channel 2D cubical persistence (periodic in both dims)
  → concatenate diagrams across channels per homology dimension
  → [H0 diagrams, H1 diagrams]

For slow_stride [8,16,16,16,8] k=[7,5,3,3,3] s=[4,2,2,2,2]:
  After last layer: 8 channels × 8×8 = 64 cells each
  Per channel: H0 (components), H1 (loops on torus)
  Total diagrams: 8 × H0 + 8 × H1, concatenated per dimension per sample

Output format:
    [[H0_diag_0, ...], [H1_diag_0, ...]]
    Compatible with 'combined' vectorization (2 sub-vectorizers).
"""
from typing import List, Optional
import gc
import os
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from topofisher.filtrations.cnn_gap import CNNGAPFiltration


class CNNPersistenceFiltration(nn.Module):
    """
    Trainable CNN encoder + per-channel 2D differentiable cubical persistence.

    Each CNN output channel is treated as an independent 2D periodic cubical
    complex.  Persistence diagrams across channels are concatenated per
    homology dimension, giving one combined H0 diagram and one combined H1
    diagram per sample.

    Args:
        checkpoint_path: Path to pre-trained CNN+GAP checkpoint (.pt file).
        encoder_channels: Channel dims for the CNN encoder.
        encoder_kernels: Kernel sizes for the CNN encoder.
        encoder_strides: Strides for the CNN encoder.
        circular_padding: Use circular padding in CNN convolutions.
        homology_dimensions: Homology dimensions for persistence (e.g. [0, 1]).
        periodic_spatial: Periodic BCs in spatial dimensions.
        persistence_backend: 'gudhi' (per-channel 2D, periodic) or 'gudhi_gpu'.
        n_jobs: Parallel workers for CPU persistence (-1 = all cores).
        skip_k: Recompute topology every skip_k forward passes (CPU backend only).
        standardize: Per-sample standardize the feature tensor before persistence.
        sub_batch_size: GPU sub-batch size for value-matching.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        circular_padding: bool = True,
        homology_dimensions: List[int] = [0, 1, 2],
        periodic_spatial: bool = True,
        persistence_backend: str = 'gudhi_gpu',
        n_jobs: int = -1,
        skip_k: int = 5,
        standardize: bool = True,
        sub_batch_size: int = 200,
    ):
        super().__init__()

        self.standardize = standardize
        self.dimensions = homology_dimensions
        self.min_persistence = [0.0] * len(homology_dimensions)
        self.periodic_spatial = periodic_spatial
        self.persistence_backend = persistence_backend
        self.skip_k = max(1, skip_k)
        self.sub_batch_size = sub_batch_size

        if encoder_channels is None:
            encoder_channels = [8, 16, 16, 16, 8]
        if encoder_kernels is None:
            encoder_kernels = [7, 5, 3, 3, 3]
        if encoder_strides is None:
            encoder_strides = [4, 2, 2, 2, 2]

        # Build CNN encoder
        cnn = CNNGAPFiltration(
            encoder_channels=encoder_channels,
            encoder_kernels=encoder_kernels,
            encoder_strides=encoder_strides,
            circular_padding=circular_padding,
        )

        if checkpoint_path is not None:
            # Warm start from pre-trained checkpoint
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in ckpt and 'filtration' in ckpt['model_state_dict']:
                state_dict = ckpt['model_state_dict']['filtration']
            else:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} does not contain "
                    f"model_state_dict.filtration. Keys: {list(ckpt.keys())}"
                )
            cnn.load_state_dict(state_dict)
            print(f"    Loaded encoder weights from: {checkpoint_path}")
        else:
            print("    Random Xavier initialization (no checkpoint)")

        # Use the full CNN encoder (trainable — NOT frozen)
        self.encoder = cnn.encoder

        # Threading (for CPU backend)
        if n_jobs == -1:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            self.n_jobs = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
        else:
            self.n_jobs = max(1, n_jobs)

        # Skip-K topology caching (CPU backend only)
        self._call_count = 0
        self._cached_cof_pp = None
        self._cached_n_samples = 0

        # GPU persistence function (lazy-loaded)
        self._gpu_func_batched = None

        # Store info for repr
        self._n_channels = encoder_channels[-1]
        self._total_stride = 1
        for s in encoder_strides:
            self._total_stride *= s

    # ------------------------------------------------------------------
    # GPU backend: batched GUDHI CUDA + value-matching
    # ------------------------------------------------------------------

    def _ensure_gpu_backend(self):
        """Lazy-load GPU 3D persistence function (falls back to CPU threading)."""
        if self._gpu_func_batched is not None:
            return
        from gudhi._pers_cub_low_dim_gpu_ext import __backend__
        if self.periodic_spatial:
            from gudhi._pers_cub_low_dim_gpu_ext import (
                _persistence_on_boxes_periodic_from_top_cells_gpu_batched,
            )
            self._gpu_func_batched = _persistence_on_boxes_periodic_from_top_cells_gpu_batched
            self._n_hom_dims = 4  # H0, H1, H2, H3 for fully periodic 3D
        else:
            from gudhi._pers_cub_low_dim_gpu_ext import (
                _persistence_on_boxes_from_top_cells_gpu_batched,
            )
            self._gpu_func_batched = _persistence_on_boxes_from_top_cells_gpu_batched
            self._n_hom_dims = 3  # H0, H1, H2 for non-periodic 3D
        if not hasattr(self, '_gpu_backend_logged'):
            print(f"    3D persistence backend: {__backend__}", flush=True)
            self._gpu_backend_logged = True

    def _forward_gpu(self, feat: torch.Tensor) -> List[List[torch.Tensor]]:
        """GPU-batched persistence with value-matching for differentiability."""
        self._ensure_gpu_backend()

        B = feat.shape[0]
        device = feat.device
        dtype = feat.dtype

        # Reshape to (B, C, H, W) for GUDHI 3D batched function
        feat_3d = feat  # already (B, C, h, w) from encoder
        X_flat = feat.reshape(B, -1)   # (B, N) where N = C*h*w
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

    def _process_gpu_batch(self, feat_3d, X_flat, N, device, dtype):
        """Process a sub-batch through GPU persistence + value-matching."""
        B = feat_3d.shape[0]

        # 1) GPU persistence (non-differentiable)
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

            # Collect and filter diagrams
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

    # ------------------------------------------------------------------
    # CPU backend: GUDHI PeriodicCubicalComplex + cofaces
    # ------------------------------------------------------------------

    def _should_recompute(self, n_samples: int) -> bool:
        if self.skip_k <= 1 or not self.training:
            return True
        if self._cached_cof_pp is None or self._cached_n_samples != n_samples:
            return True
        return (self._call_count % self.skip_k == 0)

    def _topology_worker(self, sample_np: np.ndarray):
        """Compute 2D persistence for all channels of a single (C, h, w) sample.

        Returns a list of cofaces_of_persistence_pairs, one per channel.
        """
        C, H, W = sample_np.shape
        import gudhi
        results = []
        for c in range(C):
            cells = sample_np[c]  # (H, W)
            if not np.all(np.isfinite(cells)):
                cells = np.nan_to_num(cells, nan=0.0, posinf=1e6, neginf=-1e6)
            if self.periodic_spatial:
                cc = gudhi.PeriodicCubicalComplex(
                    top_dimensional_cells=cells,
                    periodic_dimensions=[True, True],
                )
            else:
                cc = gudhi.CubicalComplex(
                    dimensions=[H, W],
                    top_dimensional_cells=cells.flatten(),
                )
            cc.compute_persistence()
            results.append(cc.cofaces_of_persistence_pairs())
        return results

    def _forward_cpu(self, feat: torch.Tensor) -> List[List[torch.Tensor]]:
        """Per-channel 2D persistence with cofaces + batched gather.

        For each sample, runs 2D persistence on each channel independently,
        then concatenates the per-channel diagrams for each homology dimension.
        """
        B, C, H, W = feat.shape
        device = feat.device
        dtype = feat.dtype

        self._call_count += 1
        recompute = self._should_recompute(B)

        if recompute:
            feat_np = feat.detach().cpu().numpy()
            n_workers = min(self.n_jobs, B)
            if n_workers > 1 and B > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    all_cof_pp = list(executor.map(
                        self._topology_worker,
                        [feat_np[i] for i in range(B)]
                    ))
            else:
                all_cof_pp = [self._topology_worker(feat_np[i])
                              for i in range(B)]
            self._cached_cof_pp = all_cof_pp
            self._cached_n_samples = B
        else:
            all_cof_pp = self._cached_cof_pp

        # all_cof_pp[sample][channel] = cofaces_of_persistence_pairs result
        # Each channel has N_ch = H*W cells; cell indices are local to channel.
        N_ch = H * W
        all_diagrams = [[] for _ in self.dimensions]

        for dim_idx, dim in enumerate(self.dimensions):
            min_pers = self.min_persistence[dim_idx]

            for i in range(B):
                # Collect birth/death indices across all channels
                pairs_list = []
                for c in range(C):
                    cof = all_cof_pp[i][c]
                    if len(cof[0]) > dim and len(cof[0][dim]) > 0:
                        channel_pairs = cof[0][dim]  # (K, 2) local indices
                        # Offset indices to global flat index: c * N_ch + local_idx
                        offset = c * N_ch
                        pairs_list.append(channel_pairs + offset)

                if not pairs_list:
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype))
                    continue

                all_pairs = np.concatenate(pairs_list, axis=0)  # (K_total, 2)
                birth_idx = torch.as_tensor(
                    all_pairs[:, 0], dtype=torch.long, device=device)
                death_idx = torch.as_tensor(
                    all_pairs[:, 1], dtype=torch.long, device=device)

                # Differentiable gather from flattened feature tensor
                feat_flat_i = feat[i].reshape(-1)  # (C*H*W,)
                births = feat_flat_i[birth_idx]
                deaths = feat_flat_i[death_idx]
                dgm = torch.stack([births, deaths], dim=1)  # (K_total, 2)

                if min_pers > 0:
                    pers = torch.abs(dgm[:, 1] - dgm[:, 0])
                    dgm = dgm[pers > min_pers]

                all_diagrams[dim_idx].append(dgm)

        return all_diagrams

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply trainable CNN encoder + per-channel 2D cubical persistence.

        Args:
            x: Input tensor of shape (batch, H, W)

        Returns:
            List[List[Tensor]]: [[H0_diags], [H1_diags]]
                Each H_dim list has one diagram per sample (concatenated across channels).
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN encoder (trainable, gradients flow)
        feat = self.encoder(x)  # (batch, C, h, w)

        # Per-sample standardization (differentiable)
        if self.standardize:
            mean = feat.mean(dim=(1, 2, 3), keepdim=True)
            std = feat.std(dim=(1, 2, 3), keepdim=True)
            feat = (feat - mean) / (std + 1e-8)

        if self.persistence_backend == 'gudhi_gpu':
            return self._forward_gpu(feat)
        else:
            return self._forward_cpu(feat)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = sum(p.numel() for p in self.encoder.parameters())
        trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        spatial = 512 // self._total_stride
        periodic_str = f"[{self.periodic_spatial}, {self.periodic_spatial}]"
        return (f"CNNPersistenceFiltration(\n"
                f"  encoder: {self._n_channels}ch × {spatial}×{spatial}, "
                f"params={total} (trainable={trainable})\n"
                f"  per-channel 2D cubical: {self._n_channels} channels × "
                f"{spatial}×{spatial} = {spatial * spatial} cells/ch, "
                f"periodic={periodic_str}\n"
                f"  backend={self.persistence_backend}, "
                f"homology_dimensions={self.dimensions}, "
                f"standardize={self.standardize}\n"
                f")")
