"""
Learnable downsampling filtration: CNN → downsample → cubical persistence.

This module implements a learnable filtration that downsamples high-resolution
input fields to a smaller target resolution before computing persistence diagrams.
This dramatically speeds up the CPU-bound cubical persistence step:
    - Cubical persistence scales as O(N^2 log N^2) with map resolution
    - N=64 → N=16 gives ~16× speedup in persistence (6.1ms → 0.4ms per map)

Architecture:
    Input (N×N) → Stride-2 Conv layers → (target_size×target_size) → Cubical persistence → Diagrams

The number of stride-2 downsampling stages is computed automatically from the
input resolution and target_size. Between downsampling stages, additional
size-preserving convolutions with learned features process the representation.

Training Strategy:
    - CNN learns to compress spatial information optimally for persistence
    - Gradient flows through CNN via torch.gather in cubical layer
    - Fisher loss: minimize -log|F|
    - Data regenerated each epoch since filtration changes
"""
from typing import List, Optional
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class DownsamplingCNN(nn.Module):
    """
    CNN that downsamples input fields from N×N to target_size×target_size.

    Architecture:
        For each downsampling stage (stride-2):
            Conv2d(stride=2, kernel_size=k, padding=k//2)  →  halves spatial dims
            Activation

        Final:
            Refinement conv layers (stride=1) → single channel output
            AdaptiveAvgPool2d for exact target_size guarantee

    The number of stride-2 stages is log2(input_size / target_size).
    If input_size is not a power-of-2 multiple of target_size, an adaptive
    average pooling layer handles the final resize.

    Example:
        N=64, target_size=16 → 2 stride-2 stages (64→32→16)
        N=128, target_size=16 → 3 stride-2 stages (128→64→32→16)
        N=512, target_size=32 → 4 stride-2 stages (512→256→128→64→32)
        N=512, target_size=16 → 5 stride-2 stages (512→256→128→64→32→16)
    """

    def __init__(
        self,
        target_size: int = 16,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 5,
        activation: str = 'relu',
        input_size: int = 512,
        gradient_checkpointing: bool = False,
        circular_padding: bool = False,
        residual: bool = False,
    ):
        """
        Initialize downsampling CNN.

        Args:
            target_size: Target spatial resolution after downsampling
            hidden_channels: Hidden channel dimensions for each downsampling stage.
                            Length determines number of conv layers per stage.
            kernel_size: Convolution kernel size (must be odd)
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
            input_size: Expected input spatial resolution. Used to compute
                       the number of stride-2 downsampling blocks:
                       n_blocks = ceil(log2(input_size / target_size)).
                       Default 512 for lensing maps.
            gradient_checkpointing: If True, use gradient checkpointing on
                       downsample_layers to trade compute for memory. Saves
                       ~60% CNN activation memory with ~20% slower backward.
                       Essential for wide architectures (e.g. [64,64,64]) on
                       high-resolution inputs (512×512).
            circular_padding: If True, use padding_mode='circular' on all Conv2d
                       layers. Should be True when the input field is periodic
                       (e.g., FFT-generated GRFs, flat-sky lensing maps).
            residual: If True, add a skip connection: output = AvgPool(input) + CNN(input).
                       The CNN learns additive corrections to the average-pooled
                       raw field, preserving the native topology as a baseline.
        """
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.residual = residual

        self.target_size = target_size

        # Build activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        padding = kernel_size // 2
        pad_mode = 'circular' if circular_padding else 'zeros'

        # We'll build the network dynamically in the first forward pass
        # because we need to know the input spatial size to determine
        # the number of downsampling stages.
        # Instead, we build a flexible architecture that works for any input
        # size by using stride-2 conv layers + adaptive pooling at the end.

        layers = []
        in_ch = 1  # single-channel input

        # Use hidden_channels to define the architecture.
        # Each hidden channel defines one conv block.
        # We distribute stride-2 layers across the blocks:
        # first few blocks use stride=2 (downsampling), rest use stride=1.
        # The number of stride-2 blocks is determined at forward time
        # by using adaptive pooling as the final spatial adjustment.

        # Strategy: all conv layers use stride=1 with same-padding.
        # A separate adaptive pool at the end handles the spatial resize.
        # This is simpler and works for any input size.

        # BUT: stride-2 convs are better than pooling because they're learnable.
        # Let's use a fixed number of stride-2 layers based on common use cases.
        # For generality, we use ceil(log2(max_expected_input / target_size))
        # stride-2 layers. Extra spatial reduction is handled by adaptive pool.

        # Build: [Downsample blocks] → [Refinement blocks] → [Final conv]
        # We allocate the first min(n_hidden, needed_stages) channels to
        # stride-2 downsampling, and the rest to size-preserving refinement.

        # For now, use a simple two-phase approach:
        # Phase 1: Stride-2 downsampling blocks (as many as needed)
        # Phase 2: Stride-1 refinement with hidden channels
        # Phase 3: Final 1×1 → single channel

        # Since we don't know input_size at init, we'll build a generic
        # architecture and use adaptive_avg_pool2d to hit target_size.

        # Number of stride-2 blocks computed from input/target size ratio:
        # 512→32: 4 blocks, 512→16: 5 blocks, 64→16: 2 blocks
        n_downsample_blocks = max(1, math.ceil(math.log2(input_size / target_size)))

        for i in range(n_downsample_blocks):
            out_ch = hidden_channels[min(i, len(hidden_channels) - 1)]
            # Stride-2 conv for downsampling
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2,
                                    padding=padding, padding_mode=pad_mode))
            layers.append(act_fn)
            in_ch = out_ch

        self.downsample_layers = nn.Sequential(*layers)

        # Refinement: size-preserving conv layers on the downsampled feature map
        refine_layers = []
        for ch in hidden_channels:
            refine_layers.append(nn.Conv2d(in_ch, ch, kernel_size, stride=1,
                                          padding=padding, padding_mode=pad_mode))
            refine_layers.append(act_fn)
            in_ch = ch

        self.refine_layers = nn.Sequential(*refine_layers)

        # Final conv: reduce to single channel
        self.final_conv = nn.Conv2d(in_ch, 1, kernel_size, stride=1,
                                    padding=padding, padding_mode=pad_mode)

        # Adaptive pool to guarantee target_size output.
        # When circular_padding=True and input_size / target_size is a power of 2
        # (e.g. 512/64 = 8 = 2^3), the stride-2 blocks already reach target_size
        # exactly, so adaptive pool is a no-op. We replace with Identity to make
        # the periodic pipeline fully consistent. For non-power-of-2 ratios, we
        # keep AdaptiveAvgPool2d as approximate spatial adjustment.
        expected_size_after_downsample = input_size // (2 ** n_downsample_blocks)
        if circular_padding and expected_size_after_downsample == target_size:
            self.adaptive_pool = nn.Identity()
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(target_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input field to target resolution.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            Downsampled tensor of shape (batch_size, target_size, target_size) or
            (target_size, target_size)
        """
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)

        # (B, H, W) → (B, 1, H, W)
        x = x.unsqueeze(1)

        # For residual: compute average-pooled raw field at target resolution
        if self.residual:
            x_pool = nn.functional.adaptive_avg_pool2d(x, self.target_size).squeeze(1)

        # Stride-2 downsampling (with gradient checkpointing to save memory)
        if self.training and self.gradient_checkpointing:
            x = grad_checkpoint(self.downsample_layers, x, use_reentrant=False)
        else:
            x = self.downsample_layers(x)

        # Adaptive pool to exact target size
        x = self.adaptive_pool(x)

        # Size-preserving refinement (also checkpoint if enabled)
        if self.training and self.gradient_checkpointing:
            x = grad_checkpoint(self.refine_layers, x, use_reentrant=False)
        else:
            x = self.refine_layers(x)

        # Final conv to single channel
        x = self.final_conv(x)

        # (B, 1, target_size, target_size) → (B, target_size, target_size)
        x = x.squeeze(1)

        if self.residual:
            # Skip connection: add average-pooled raw input to CNN output.
            # Preserves native field topology as baseline; CNN learns corrections.
            x = x + x_pool

        if single_sample:
            x = x.squeeze(0)

        return x


class LearnableDownsampleFiltration(nn.Module):
    """
    Learnable downsampling filtration: CNN downsample → cubical persistence.

    This filtration solves the computational bottleneck of learnable TDA on
    high-resolution fields. Instead of running O(N^2)-expensive cubical
    persistence on the full-resolution field, a learnable CNN first
    downsamples to a small target resolution, then persistence is computed
    on the compressed representation.

    Pipeline:
        Input (N×N) → DownsamplingCNN → (K×K) → Cubical persistence → Diagrams

    where K = target_size (default 16).

    Speedup example (N=64, K=16):
        - Original: 6.1 ms/map cubical persistence → 330 min per run
        - Downsampled: 0.4 ms/map cubical persistence → ~22 min per run
        - Speedup: ~15×

    The CNN learns to compress spatial information in a way that preserves
    the topological features most relevant to Fisher information.

    Example:
        >>> filtration = LearnableDownsampleFiltration(
        ...     target_size=16,
        ...     homology_dimensions=[0, 1],
        ...     hidden_channels=[32, 64, 32]
        ... )
        >>> field = torch.randn(10, 64, 64, requires_grad=True)
        >>> diagrams = filtration(field)  # List[List[Tensor]]
        >>> # Persistence computed on 16×16, not 64×64!
    """

    def __init__(
        self,
        target_size: int = 16,
        homology_dimensions: List[int] = [0, 1],
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 5,
        activation: str = 'relu',
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        skip_k: int = 1,
        persistence_backend: str = 'gudhi',
        persistence_construction: str = 'T',
        standardize: bool = True,
        input_size: int = 512,
        gpu_sub_batch_size: int = 100,
        gradient_checkpointing: bool = False,
        circular_padding: bool = False,
        periodic: bool = False,
        residual: bool = False,
    ):
        """
        Initialize learnable downsampling filtration.

        Args:
            target_size: Target spatial resolution for persistence computation
            homology_dimensions: Homology dimensions to compute (e.g. [0, 1])
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function
            min_persistence: Minimum persistence threshold per dimension
            superlevel: If True, compute superlevel filtration
            n_jobs: Number of parallel threads for CPU GUDHI backend.
            skip_k: Recompute topology every skip_k forward passes (CPU backend).
            persistence_backend: Persistence backend.
                   'gudhi' (default): CPU GUDHI with optional threading.
                   'gudhi_gpu': GPU GUDHI CUDA extension (recommended for lensing).
                   'cmp_gpu': GPU CMP CUDA kernels.
            persistence_construction: Cubical complex construction ('T' or 'V').
                   'V' recommended: higher Fisher info, vertices → (2K-1)×(2K-1) complex.
                   Only used with gudhi_gpu backend.
            standardize: If True, per-sample standardize CNN output (mean=0, std=1).
                   Preserves pixel ordering (and thus persistence topology) while
                   keeping values in a bounded range for stable PI grids.
            input_size: Expected input spatial resolution (for computing
                       number of stride-2 downsampling blocks).
            gradient_checkpointing: If True, use gradient checkpointing on the
                   CNN to reduce GPU memory by ~60% at ~20% compute cost.
                   Essential for wide architectures on high-resolution inputs.
            circular_padding: If True, use padding_mode='circular' on all CNN
                   Conv2d layers. Should be True for periodic input fields.
            periodic: If True, use periodic boundary conditions in cubical
                   persistence computation. Should be True when input fields
                   are periodic (FFT-generated).
            residual: If True, add skip connection in CNN: output = AvgPool(input) + CNN(input).
                   Preserves the raw field's topology as baseline; CNN learns corrections.
        """
        super().__init__()

        self.target_size = target_size
        self.homology_dimensions = homology_dimensions
        self.superlevel = superlevel
        self.standardize = standardize
        # Sub-batch CNN during eval to avoid OOM on large validation sets.
        # Training uses batch_size (e.g. 200) which fits in memory.
        # Validation/test may pass 1250+ samples at once.
        self.cnn_eval_batch_size = 200

        # Downsampling CNN: N×N → target_size×target_size
        self.cnn = DownsamplingCNN(
            target_size=target_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            input_size=input_size,
            gradient_checkpointing=gradient_checkpointing,
            circular_padding=circular_padding,
            residual=residual,
        )

        # Differentiable cubical persistence on downsampled field
        self.cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=min_persistence,
            superlevel=superlevel,
            n_jobs=n_jobs,
            skip_k=skip_k,
            backend=persistence_backend,
            construction=persistence_construction,
            gpu_sub_batch_size=gpu_sub_batch_size,
            periodic=periodic,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply downsampling + cubical persistence to input fields.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2)
        """
        # Downsample: (B, N, N) → (B, target_size, target_size)
        # During eval, sub-batch the CNN to avoid OOM on large validation sets
        # (e.g., 1250 × 512×512 with 64 channels → 19.5 GiB per layer output).
        # Training always uses batch_size (e.g. 200), so no sub-batching needed.
        if not self.training and x.ndim == 3 and x.shape[0] > self.cnn_eval_batch_size:
            chunks = []
            for i in range(0, x.shape[0], self.cnn_eval_batch_size):
                chunks.append(self.cnn(x[i:i + self.cnn_eval_batch_size]))
            x_down = torch.cat(chunks, dim=0)
        else:
            x_down = self.cnn(x)

        if self.standardize:
            # Per-sample standardization: keep CNN output in bounded range
            # for stable PI grids. Preserves pixel ordering (persistence
            # topology) since standardization is a monotone transform per sample.
            if x_down.ndim == 3:
                mean = x_down.mean(dim=(-2, -1), keepdim=True)
                std = x_down.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                x_down = (x_down - mean) / std
            elif x_down.ndim == 2:
                mean = x_down.mean()
                std = x_down.std().clamp(min=1e-6)
                x_down = (x_down - mean) / std

        # Cubical persistence on downsampled field
        diagrams = self.cubical(x_down)

        return diagrams

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
