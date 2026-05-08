"""
Hybrid CNN+GAP + TDA filtration for 2D fields.

Two parallel branches from the same input:
  1) CNN+GAP branch: strided CNN → Global Average Pooling → fixed-dim vector
  2) TDA branch: size-preserving CNN → periodic cubical persistence → diagrams

The outputs are concatenated at the vectorization stage via HybridVectorization,
which applies IdentityVectorization to the GAP vectors and DiffCurves to the
persistence diagrams, then concatenates everything.

Output format:
    [[gap_vec_0, gap_vec_1, ...], [H0_diag_0, H0_diag_1, ...], [H1_diag_0, ...]]
    Slot 0: GAP vectors (one per sample)
    Slots 1+: persistence diagrams (one list per homology dimension)
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.cnn_gap import CNNGAPFiltration
from topofisher.filtrations.learnable import PreFiltrationCNN
from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class CNNGAPTDAFiltration(nn.Module):
    """
    Hybrid filtration: CNN+GAP (global features) + CNN+TDA (topological features).

    Two independent CNNs so each branch can specialize:
    - GAP branch learns to extract global summary statistics
    - TDA branch learns to enhance topological features for persistence

    Args:
        gap_channels: Channel dims for the GAP branch strided CNN.
        gap_kernels: Kernel sizes for the GAP branch.
        gap_strides: Strides for the GAP branch.
        tda_channels: Channel dims for the TDA branch CNN.
        tda_kernel_size: Kernel size for the TDA branch (must be odd).
        tda_type: TDA branch architecture type:
            'preserve': Size-preserving CNN (PreFiltrationCNN) — default.
            'downsample': Downsampling CNN (DownsamplingCNN) → target_size.
                         Much faster persistence on high-res inputs (e.g. 512→64).
        tda_target_size: Target resolution for downsample TDA branch (default 64).
        tda_input_size: Input resolution for downsample TDA branch (default 512).
        tda_standardize: Per-sample standardize CNN output before persistence.
        homology_dimensions: Homology dimensions for persistence.
        circular_padding: Use circular padding (for periodic fields).
        residual: Use skip connection in TDA branch CNN.
        persistence_backend: Backend for cubical persistence.
        persistence_construction: Cubical complex construction ('T' or 'V').
        periodic: Periodic boundary conditions for persistence.
        skip_k: Recompute topology every skip_k steps (for speed).
        activation: Activation function for both CNNs.
    """

    def __init__(
        self,
        gap_channels: Optional[List[int]] = None,
        gap_kernels: Optional[List[int]] = None,
        gap_strides: Optional[List[int]] = None,
        tda_channels: Optional[List[int]] = None,
        tda_kernel_size: int = 3,
        tda_type: str = 'preserve',
        tda_target_size: int = 64,
        tda_input_size: int = 512,
        tda_standardize: bool = True,
        homology_dimensions: List[int] = [0, 1],
        circular_padding: bool = True,
        residual: bool = False,
        persistence_backend: str = 'gudhi_gpu',
        persistence_construction: str = 'V',
        periodic: bool = True,
        skip_k: int = 1,
        activation: str = 'relu',
    ):
        super().__init__()
        self.tda_type = tda_type
        self.tda_standardize = tda_standardize

        # Defaults for tiny architecture
        if gap_channels is None:
            gap_channels = [8, 16, 8]
        if gap_kernels is None:
            gap_kernels = [3, 3, 3]
        if gap_strides is None:
            gap_strides = [2, 2, 2]
        if tda_channels is None:
            tda_channels = [16, 16]

        # Branch 1: CNN + GAP → fixed-dim vector
        self.gap_branch = CNNGAPFiltration(
            encoder_channels=gap_channels,
            encoder_kernels=gap_kernels,
            encoder_strides=gap_strides,
            circular_padding=circular_padding,
        )

        # Branch 2: CNN → cubical persistence → diagrams
        if tda_type == 'preserve':
            # Size-preserving CNN (original behavior)
            self.tda_cnn = PreFiltrationCNN(
                hidden_channels=tda_channels,
                kernel_size=tda_kernel_size,
                activation=activation,
                residual=residual,
                standardize=True,
                circular_padding=circular_padding,
            )
        elif tda_type == 'downsample':
            # Downsampling CNN: 512→64 (much faster persistence)
            from topofisher.filtrations.learnable_downsample import DownsamplingCNN
            self.tda_cnn = DownsamplingCNN(
                target_size=tda_target_size,
                hidden_channels=tda_channels,
                kernel_size=tda_kernel_size,
                activation=activation,
                input_size=tda_input_size,
                circular_padding=circular_padding,
                residual=residual,
            )
        else:
            raise ValueError(f"Unknown tda_type: {tda_type!r}. Use 'preserve' or 'downsample'.")

        self.tda_cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            superlevel=False,
            n_jobs=1,
            skip_k=skip_k,
            backend=persistence_backend,
            construction=persistence_construction,
            periodic=periodic,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply both branches to input fields.

        Args:
            x: Input tensor of shape (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] with structure:
                [0]: GAP vectors [[vec_0, vec_1, ...]]  (each vec has shape (gap_dim,))
                [1]: H0 diagrams [[diag_0, diag_1, ...]] (each diag has shape (n_pairs, 2))
                [2]: H1 diagrams [[diag_0, diag_1, ...]]
        """
        # Branch 1: GAP vectors
        gap_output = self.gap_branch(x)  # [[vec_0, vec_1, ...]]
        gap_vectors = gap_output[0]  # [vec_0, vec_1, ...]

        # Branch 2: TDA diagrams
        x_transformed = self.tda_cnn(x)

        # For downsample branch: optionally standardize after CNN
        if self.tda_type == 'downsample' and self.tda_standardize:
            if x_transformed.ndim == 3:
                mean = x_transformed.mean(dim=(-2, -1), keepdim=True)
                std = x_transformed.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                x_transformed = (x_transformed - mean) / std
            elif x_transformed.ndim == 2:
                mean = x_transformed.mean()
                std = x_transformed.std().clamp(min=1e-6)
                x_transformed = (x_transformed - mean) / std

        tda_diagrams = self.tda_cubical(x_transformed)  # [[H0_diags], [H1_diags]]

        # Combine: slot 0 = GAP vectors, slots 1+ = persistence diagrams
        result = [gap_vectors] + tda_diagrams
        return result

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_gap_parameters(self) -> int:
        """Return number of GAP branch parameters."""
        return sum(p.numel() for p in self.gap_branch.parameters() if p.requires_grad)

    def get_tda_parameters(self) -> int:
        """Return number of TDA branch parameters."""
        tda_params = sum(p.numel() for p in self.tda_cnn.parameters() if p.requires_grad)
        return tda_params

    def invalidate_topology_cache(self):
        """Clear cached topology indices in the cubical layer."""
        self.tda_cubical.invalidate_cache()

    def get_topology_cache_stats(self) -> dict:
        """Return topology cache statistics from the cubical layer."""
        return self.tda_cubical.get_cache_stats()

    def __repr__(self):
        gap_ch = []
        for m in self.gap_branch.encoder:
            if isinstance(m, nn.Conv2d):
                gap_ch.append(m.out_channels)
        return (f"CNNGAPTDAFiltration(\n"
                f"  gap_branch: channels={gap_ch}, gap_dim={self.gap_branch.output_dim}, "
                f"params={self.get_gap_parameters()}\n"
                f"  tda_branch: params={self.get_tda_parameters()}\n"
                f"  total_params={self.get_num_parameters()}\n"
                f")")
