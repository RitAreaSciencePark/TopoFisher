"""
Frozen CNN+GAP + nonlearnable TDA diagnostic filtration.

Loads a pre-trained CNN+GAP checkpoint (frozen) and runs nonlearnable cubical
persistence on the raw input. Outputs both GAP vectors and persistence diagrams
for downstream concatenation via HybridVectorization.

This is a diagnostic tool to test whether TDA features are complementary to
the learned CNN+GAP representation. No training required — inference only.

Output format (same as CNNGAPTDAFiltration):
    [[gap_vec_0, gap_vec_1, ...], [H0_diag_0, H0_diag_1, ...], [H1_diag_0, ...]]
    Slot 0: GAP vectors (from frozen checkpoint)
    Slots 1+: persistence diagrams (from nonlearnable cubical persistence)
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from topofisher.filtrations.cnn_gap import CNNGAPFiltration
from topofisher.filtrations.cubical import CubicalLayer


class FrozenGAPPlusTDAFiltration(nn.Module):
    """
    Diagnostic filtration: frozen CNN+GAP + nonlearnable cubical persistence.

    Loads a pre-trained CNN+GAP model from checkpoint and freezes all weights.
    Simultaneously runs nonlearnable cubical persistence on the raw input field.
    The two outputs are combined into the hybrid format compatible with
    HybridVectorization.

    Args:
        checkpoint_path: Path to the pre-trained CNN+GAP checkpoint (.pt file).
        gap_channels: Channel dims matching the checkpoint architecture.
        gap_kernels: Kernel sizes matching the checkpoint architecture.
        gap_strides: Strides matching the checkpoint architecture.
        circular_padding: Circular padding matching the checkpoint architecture.
        homology_dimensions: Homology dimensions for cubical persistence.
        persistence_backend: Backend for cubical persistence.
        construction: Cubical complex construction ('T' or 'V').
        periodic: Periodic boundary conditions for cubical persistence.
        n_jobs: Parallel workers for CPU persistence computation.
        tda_downsample_size: If set, avg-pool the input to this resolution
            before computing persistence. E.g. 128 for 512→128. None = no
            downsampling (compute on raw field).
    """

    def __init__(
        self,
        checkpoint_path: str,
        gap_channels: Optional[List[int]] = None,
        gap_kernels: Optional[List[int]] = None,
        gap_strides: Optional[List[int]] = None,
        circular_padding: bool = True,
        homology_dimensions: List[int] = [0, 1],
        persistence_backend: str = 'gudhi',
        construction: str = 'V',
        periodic: bool = True,
        n_jobs: int = -1,
        gap_batch_size: int = 200,
        tda_downsample_size: Optional[int] = None,
    ):
        super().__init__()

        self.gap_batch_size = gap_batch_size
        self.tda_downsample_size = tda_downsample_size
        # Defaults match slow_stride architecture
        if gap_channels is None:
            gap_channels = [8, 16, 16, 16, 8]
        if gap_kernels is None:
            gap_kernels = [7, 5, 3, 3, 3]
        if gap_strides is None:
            gap_strides = [4, 2, 2, 2, 2]

        # Build GAP branch with same architecture as checkpoint
        self.gap_branch = CNNGAPFiltration(
            encoder_channels=gap_channels,
            encoder_kernels=gap_kernels,
            encoder_strides=gap_strides,
            circular_padding=circular_padding,
        )

        # Load checkpoint and extract filtration weights
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in ckpt and 'filtration' in ckpt['model_state_dict']:
            state_dict = ckpt['model_state_dict']['filtration']
        else:
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not contain "
                f"model_state_dict.filtration. Keys: {list(ckpt.keys())}"
            )
        self.gap_branch.load_state_dict(state_dict)

        # Freeze all GAP branch parameters
        for p in self.gap_branch.parameters():
            p.requires_grad = False
        self.gap_branch.eval()

        # Nonlearnable cubical persistence on raw input
        self.cubical = CubicalLayer(
            homology_dimensions=homology_dimensions,
            backend=persistence_backend,
            construction=construction,
            periodic=periodic,
            n_jobs=n_jobs,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply frozen GAP + nonlearnable cubical persistence.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] with structure:
                [0]: GAP vectors [vec_0, vec_1, ...] (frozen, from checkpoint)
                [1]: H0 diagrams [diag_0, diag_1, ...]
                [2]: H1 diagrams [diag_0, diag_1, ...]
        """
        import time as _time
        n = x.shape[0]
        input_shape = 'x'.join(str(s) for s in x.shape[1:])

        # Branch 1: Frozen GAP vectors (batched to avoid OOM on large datasets)
        print(f"    [FrozenGAP] Branch 1: CNN+GAP on {n} × {input_shape}...",
              flush=True)
        t0 = _time.time()
        gap_vectors = []
        with torch.no_grad():
            for i in range(0, n, self.gap_batch_size):
                chunk = x[i:i + self.gap_batch_size]
                gap_output = self.gap_branch(chunk)  # [[vec_0, vec_1, ...]]
                gap_vectors.extend(gap_output[0])
                if (i // self.gap_batch_size) % 20 == 0:
                    print(f"      GAP: {i + len(chunk)}/{n}", flush=True)
        print(f"    [FrozenGAP] Branch 1 done: {_time.time() - t0:.1f}s",
              flush=True)

        # Branch 2: Nonlearnable cubical persistence (CPU)
        x_tda = x.cpu() if x.is_cuda else x
        if self.tda_downsample_size is not None:
            tda_shape_before = 'x'.join(str(s) for s in x_tda.shape[1:])
            x_tda = F.adaptive_avg_pool2d(
                x_tda.unsqueeze(1), self.tda_downsample_size
            ).squeeze(1)
            tda_shape_after = 'x'.join(str(s) for s in x_tda.shape[1:])
            print(f"    [FrozenGAP] Branch 2: TDA on {n} × {tda_shape_after} "
                  f"(downsampled from {tda_shape_before})", flush=True)
        else:
            print(f"    [FrozenGAP] Branch 2: TDA on {n} × {input_shape}",
                  flush=True)
        t1 = _time.time()
        tda_diagrams = self.cubical(x_tda)  # [[H0_diags], [H1_diags]]
        print(f"    [FrozenGAP] Branch 2 done: {_time.time() - t1:.1f}s",
              flush=True)

        # Combine: slot 0 = GAP vectors, slots 1+ = persistence diagrams
        return [gap_vectors] + tda_diagrams

    def train(self, mode: bool = True):
        """Override train to keep GAP branch always in eval mode."""
        super().train(mode)
        self.gap_branch.eval()
        return self

    def get_num_parameters(self) -> int:
        """Return total trainable parameters (should be 0)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"FrozenGAPPlusTDAFiltration(\n"
                f"  gap_branch: output_dim={self.gap_branch.output_dim}, "
                f"frozen_params={sum(p.numel() for p in self.gap_branch.parameters())}\n"
                f"  cubical: {self.cubical}\n"
                f"  trainable_params={self.get_num_parameters()}\n"
                f")")
