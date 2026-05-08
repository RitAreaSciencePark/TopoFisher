"""
CNN + 2D periodic cubical persistence.

A CNN transforms the raw input (e.g. 512×512) into a single-channel scalar
field.  2D periodic cubical persistence is applied to the output.

Supports two modes:
  - Stride-1 (default): all convolutions use stride 1, output is full-res.
  - Strided: each layer can have a custom stride for downsampling.
    E.g. encoder_strides=[2,2,2] on a 512×512 input → 64×64 output.

Architecture:
  Input (B, H, W)
  → [Conv2d(in→C, k, s, circular_pad) + ReLU] × n_layers
  → Conv2d(C→1, k=1)  — linear projection, no activation
  → (B, H', W') scalar field
  → per-sample standardize
  → 2D PeriodicCubicalComplex(periodic_dimensions=[True, True])
  → [H0 diagrams, H1 diagrams]

Output format:
    [[H0_diag_0, ...], [H1_diag_0, ...]]
    Compatible with 'combined' vectorization (2 sub-vectorizers).
"""
from typing import List, Optional
import os
import torch
import torch.nn as nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class CNNFullResPersistenceFiltration(nn.Module):
    """
    CNN + 2D periodic cubical persistence.

    Args:
        hidden_channels: Number of intermediate CNN channels (default 8).
        n_layers: Number of conv layers before the 1×1 projection (default 2).
        kernel_size: Kernel size for conv layers (default 3).
        encoder_strides: Per-layer strides (list of ints). If None, all stride 1.
                         E.g. [2, 2, 2] on 512×512 → 64×64 output.
        output_size: Downsample CNN output to this spatial size before persistence.
                     None keeps CNN output resolution. E.g. 128 → (128, 128).
        homology_dimensions: Homology dims for persistence (default [0, 1]).
        periodic: Use periodic boundary conditions (default True).
        n_jobs: Parallel workers for CPU persistence (-1 = all cores).
        skip_k: Recompute topology every skip_k forward passes.
        standardize: Per-sample standardize before persistence.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        n_layers: int = 2,
        kernel_size: int = 3,
        encoder_strides: Optional[List[int]] = None,
        output_size: Optional[int] = None,
        homology_dimensions: List[int] = [0, 1],
        periodic: bool = True,
        n_jobs: int = -1,
        skip_k: int = 10,
        standardize: bool = True,
    ):
        super().__init__()

        self.dimensions = homology_dimensions
        self.min_persistence = [0.0] * len(homology_dimensions)
        self.periodic = periodic
        self.skip_k = max(1, skip_k)
        self.standardize = standardize
        self.output_size = output_size
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._kernel_size = kernel_size

        # Build CNN with circular padding (stride-1 or strided)
        if encoder_strides is None:
            encoder_strides = [1] * n_layers
        assert len(encoder_strides) == n_layers, \
            f"encoder_strides length ({len(encoder_strides)}) must match n_layers ({n_layers})"
        self._encoder_strides = encoder_strides

        pad = kernel_size // 2
        layers = []
        in_ch = 1
        for i in range(n_layers):
            layers.append(nn.Conv2d(
                in_ch, hidden_channels, kernel_size,
                stride=encoder_strides[i], padding=pad,
                padding_mode='circular',
            ))
            layers.append(nn.ReLU(inplace=True))
            in_ch = hidden_channels
        # Final 1×1 projection to single channel (no activation)
        layers.append(nn.Conv2d(hidden_channels, 1, 1))
        self.cnn = nn.Sequential(*layers)

        # Threading
        if n_jobs == -1:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            self.n_jobs = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
        else:
            self.n_jobs = max(1, n_jobs)

        # Skip-K caching
        self._call_count = 0
        self._cached_cof_pp = None
        self._cached_n_samples = 0

    def _should_recompute(self, n_samples: int) -> bool:
        if self.skip_k <= 1 or not self.training:
            return True
        if self._cached_cof_pp is None or self._cached_n_samples != n_samples:
            return True
        return (self._call_count % self.skip_k == 0)

    def _topology_worker(self, field_np: np.ndarray):
        """Compute 2D persistence for a single (H, W) field."""
        if not np.all(np.isfinite(field_np)):
            field_np = np.nan_to_num(field_np, nan=0.0, posinf=1e6, neginf=-1e6)

        import gudhi
        if self.periodic:
            cc = gudhi.PeriodicCubicalComplex(
                top_dimensional_cells=field_np,
                periodic_dimensions=[True, True],
            )
        else:
            H, W = field_np.shape
            cc = gudhi.CubicalComplex(
                dimensions=[H, W],
                top_dimensional_cells=field_np.flatten(),
            )
        cc.compute_persistence()
        return cc.cofaces_of_persistence_pairs()

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply stride-1 CNN + full-resolution 2D cubical persistence.

        Args:
            x: Input tensor of shape (batch, H, W)

        Returns:
            List[List[Tensor]]: [[H0_diags], [H1_diags]]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Stride-1 CNN → single channel
        field = self.cnn(x)          # (B, 1, H, W)

        # Optional downsampling before persistence
        if self.output_size is not None:
            field = nn.functional.adaptive_avg_pool2d(
                field, self.output_size)  # (B, 1, out, out)

        field = field.squeeze(1)     # (B, H, W)

        # Per-sample standardization
        if self.standardize:
            mean = field.mean(dim=(1, 2), keepdim=True)
            std = field.std(dim=(1, 2), keepdim=True)
            field = (field - mean) / (std + 1e-8)

        B, H, W = field.shape
        device = field.device
        dtype = field.dtype
        N = H * W

        self._call_count += 1
        recompute = self._should_recompute(B)

        if recompute:
            field_np = field.detach().cpu().numpy()
            n_workers = min(self.n_jobs, B)
            if n_workers > 1 and B > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    all_cof_pp = list(executor.map(
                        self._topology_worker,
                        [field_np[i] for i in range(B)]
                    ))
            else:
                all_cof_pp = [self._topology_worker(field_np[i])
                              for i in range(B)]
            self._cached_cof_pp = all_cof_pp
            self._cached_n_samples = B
        else:
            all_cof_pp = self._cached_cof_pp

        # Flatten field for gather
        field_flat = field.reshape(B, -1)  # (B, H*W)
        all_diagrams = [[] for _ in self.dimensions]

        for dim_idx, dim in enumerate(self.dimensions):
            min_pers = self.min_persistence[dim_idx]

            for i in range(B):
                cof = all_cof_pp[i]
                if len(cof[0]) > dim and len(cof[0][dim]) > 0:
                    pairs = cof[0][dim]  # (K, 2)
                    birth_idx = torch.as_tensor(
                        pairs[:, 0], dtype=torch.long, device=device)
                    death_idx = torch.as_tensor(
                        pairs[:, 1], dtype=torch.long, device=device)

                    births = field_flat[i][birth_idx]
                    deaths = field_flat[i][death_idx]
                    dgm = torch.stack([births, deaths], dim=1)  # (K, 2)

                    if min_pers > 0:
                        pers = torch.abs(dgm[:, 1] - dgm[:, 0])
                        dgm = dgm[pers > min_pers]

                    all_diagrams[dim_idx].append(dgm)
                else:
                    all_diagrams[dim_idx].append(
                        torch.empty((0, 2), device=device, dtype=dtype))

        return all_diagrams

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = sum(p.numel() for p in self.cnn.parameters())
        trainable = sum(p.numel() for p in self.cnn.parameters() if p.requires_grad)
        periodic_str = "[True, True]" if self.periodic else "[False, False]"
        total_stride = 1
        for s in self._encoder_strides:
            total_stride *= s
        stride_str = f", strides={self._encoder_strides}" if total_stride > 1 else ""
        return (f"CNNFullResPersistenceFiltration(\n"
                f"  cnn: {self._n_layers}×Conv2d({self._hidden_channels}ch, "
                f"k={self._kernel_size}{stride_str}) + 1×1 proj, "
                f"params={total} (trainable={trainable})\n"
                f"  2D cubical: output_size={self.output_size or 'full-res'}, "
                f"periodic={periodic_str}\n"
                f"  homology_dimensions={self.dimensions}, "
                f"standardize={self.standardize}, "
                f"skip_k={self.skip_k}\n"
                f")")
