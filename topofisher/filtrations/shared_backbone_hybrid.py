"""
Shared-backbone hybrid filtration: single CNN with TDA tap.

A single CNN backbone (e.g. slow_stride) processes the input. At a configurable
intermediate layer, the feature map is "tapped" — channel-averaged to produce
a 2D field which is fed to cubical persistence. The remaining layers continue
to GAP as usual.

This forces the CNN to learn representations that are simultaneously useful for
global statistics (GAP) AND topology (persistence), injecting topological
invariance into the learned features.

Output format (same as CNNGAPTDAFiltration):
    [[gap_vec_0, gap_vec_1, ...], [H0_diag_0, H0_diag_1, ...], [H1_diag_0, ...]]
    Slot 0: GAP vectors (from full backbone)
    Slots 1+: persistence diagrams (from intermediate feature map)
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class SharedBackboneHybridFiltration(nn.Module):
    """
    Shared CNN backbone with TDA tap at an intermediate layer.

    The backbone is split at `tap_after_layer`:
      - early_layers (layers 0..tap_after_layer): shared computation
      - late_layers (layers tap_after_layer+1..end): continue to GAP

    At the tap point, the multi-channel feature map is channel-averaged to
    produce a single 2D field, which is fed through cubical persistence.

    Spatial resolution at tap points for slow_stride [8,16,16,16,8]:
      Layer 0 (s=4): 128×128
      Layer 1 (s=2): 64×64   ← recommended tap point
      Layer 2 (s=2): 32×32
      Layer 3 (s=2): 16×16

    Args:
        encoder_channels: Channel dims for backbone CNN.
        encoder_kernels: Kernel sizes for backbone CNN.
        encoder_strides: Strides for backbone CNN.
        circular_padding: Use circular padding (for periodic fields).
        tap_after_layer: Layer index after which to tap for TDA (0-indexed).
            Default 1 → tap after Conv2, giving 64×64 for slow_stride.
        homology_dimensions: Homology dimensions for persistence.
        persistence_backend: Backend for cubical persistence.
        persistence_construction: Cubical complex construction ('T' or 'V').
        periodic: Periodic boundary conditions for persistence.
        skip_k: Recompute topology every skip_k steps.
    """

    def __init__(
        self,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        circular_padding: bool = True,
        tap_after_layer: int = 1,
        homology_dimensions: List[int] = [0, 1],
        persistence_backend: str = 'gudhi_gpu',
        persistence_construction: str = 'V',
        periodic: bool = True,
        skip_k: int = 5,
    ):
        super().__init__()

        # Defaults match slow_stride
        if encoder_channels is None:
            encoder_channels = [8, 16, 16, 16, 8]
        if encoder_kernels is None:
            encoder_kernels = [7, 5, 3, 3, 3]
        if encoder_strides is None:
            encoder_strides = [4, 2, 2, 2, 2]

        assert len(encoder_channels) == len(encoder_kernels) == len(encoder_strides)
        n_layers = len(encoder_channels)
        assert 0 <= tap_after_layer < n_layers - 1, \
            f"tap_after_layer must be in [0, {n_layers - 2}], got {tap_after_layer}"

        self.tap_after_layer = tap_after_layer
        self.output_dim = encoder_channels[-1]

        pad_mode = 'circular' if circular_padding else 'zeros'

        # Build early layers (up to and including tap layer)
        early = []
        in_ch = 1
        for i in range(tap_after_layer + 1):
            out_ch = encoder_channels[i]
            k = encoder_kernels[i]
            s = encoder_strides[i]
            early.append(nn.Conv2d(in_ch, out_ch, k, stride=s,
                                   padding=k // 2, padding_mode=pad_mode))
            early.append(nn.ReLU())
            in_ch = out_ch
        self.early_layers = nn.Sequential(*early)

        # Build late layers (after tap layer → GAP)
        late = []
        for i in range(tap_after_layer + 1, n_layers):
            out_ch = encoder_channels[i]
            k = encoder_kernels[i]
            s = encoder_strides[i]
            late.append(nn.Conv2d(in_ch, out_ch, k, stride=s,
                                  padding=k // 2, padding_mode=pad_mode))
            late.append(nn.ReLU())
            in_ch = out_ch
        self.late_layers = nn.Sequential(*late)

        # Cubical persistence on the tapped feature map
        self.tda_cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            superlevel=False,
            n_jobs=1,
            skip_k=skip_k,
            backend=persistence_backend,
            construction=persistence_construction,
            periodic=periodic,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply shared backbone with TDA tap.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] with structure:
                [0]: GAP vectors [vec_0, vec_1, ...] (shape (output_dim,))
                [1]: H0 diagrams [diag_0, diag_1, ...] (shape (n_pairs, 2))
                [2]: H1 diagrams [diag_0, diag_1, ...]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, H, W)

        # Early layers → shared feature map
        feat = self.early_layers(x)  # (batch, C_tap, h_tap, w_tap)

        # TDA tap: channel-average → 2D field → persistence
        tapped = feat.mean(dim=1)  # (batch, h_tap, w_tap)

        # Per-sample standardize for stable persistence
        mean = tapped.mean(dim=(-2, -1), keepdim=True)
        std = tapped.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        tapped = (tapped - mean) / std

        tda_diagrams = self.tda_cubical(tapped)  # [[H0_diags], [H1_diags]]

        # Late layers → GAP
        encoded = self.late_layers(feat)  # (batch, C_out, h_out, w_out)
        pooled = encoded.mean(dim=[2, 3])  # (batch, C_out) — GAP
        gap_vectors = [pooled[i] for i in range(pooled.shape[0])]

        # Combine: slot 0 = GAP vectors, slots 1+ = persistence diagrams
        return [gap_vectors] + tda_diagrams

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def invalidate_topology_cache(self):
        self.tda_cubical.invalidate_cache()

    def get_topology_cache_stats(self) -> dict:
        return self.tda_cubical.get_cache_stats()

    def __repr__(self):
        early_ch = [m.out_channels for m in self.early_layers if isinstance(m, nn.Conv2d)]
        late_ch = [m.out_channels for m in self.late_layers if isinstance(m, nn.Conv2d)]
        return (f"SharedBackboneHybridFiltration(\n"
                f"  early_layers: {early_ch}\n"
                f"  late_layers: {late_ch}\n"
                f"  tap_after_layer={self.tap_after_layer}\n"
                f"  output_dim={self.output_dim}\n"
                f"  total_params={self.get_num_parameters()}\n"
                f")")
