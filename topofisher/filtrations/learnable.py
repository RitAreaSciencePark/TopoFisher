"""
Learnable filtration layer with CNN-based field transformation.

This module implements a learnable filtration that transforms input fields
using a CNN before computing persistence diagrams. The CNN is trained to
maximize Fisher information by transforming the input field in a way that
enhances topological features relevant to parameter inference.

Training Strategy:
    - CNN transforms field: N×N → N×N (learnable, size-preserving)
    - Differentiable cubical persistence on transformed field
    - Gradient flows through CNN via Fisher loss: minimize -log|F|
    - Data regenerated each epoch since filtration changes

Pipeline:
    Input (N×N) → CNN → Transformed (N×N) → Persistence → Diagrams

The CNN learns to enhance topologically discriminative features.
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class PreFiltrationCNN(nn.Module):
    """
    CNN that processes input fields before persistence filtration.

    Architecture:
        Input (N×N) → Conv layers → Output (N×N)

    The network learns to transform the input field in a way that enhances
    topological features relevant to Fisher information maximization.
    Fully convolutional - works with any input size and preserves dimensions.
    """

    def __init__(
        self,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        residual: bool = False,
        standardize: bool = True,
        circular_padding: bool = False,
    ):
        """
        Initialize CNN transformer.

        Args:
            hidden_channels: List of hidden channel dimensions
            kernel_size: Convolution kernel size (must be odd)
            activation: Activation function ('relu', 'leaky_relu', 'tanh')
            residual: If True, use skip connection: output = input + CNN(input).
                This preserves the raw field's topology as a baseline, and the
                CNN learns additive corrections. Essential for spectra where
                the CNN's effective low-pass filtering destroys high-frequency
                topological features (e.g., steep red spectra like B=2 where
                P(k) ∝ k^{-2}).
            standardize: If True, apply per-sample standardization (mean=0, std=1)
                to the CNN output. Needed for Learnable TDA pipelines where
                persistence image grids have a fixed range. Should be False for
                CNN-only pipelines (no persistence) where standardization removes
                useful scale information and destabilizes training.
            circular_padding: If True, use padding_mode='circular' on all Conv2d
                layers. Should be True when the input field is periodic (e.g.,
                FFT-generated GRFs, flat-sky lensing maps). Default False for
                backward compatibility.
        """
        super().__init__()
        self.residual = residual
        self.standardize = standardize
        self.circular_padding = circular_padding

        # Build activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Ensure kernel size is odd for proper padding
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        pad_mode = 'circular' if circular_padding else 'zeros'

        # Build convolutional layers
        layers = []
        in_channels = 1  # Single-channel input field

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                    padding=padding, padding_mode=pad_mode))
            layers.append(act_fn)
            in_channels = out_channels

        # Final conv to single channel
        layers.append(nn.Conv2d(in_channels, 1, kernel_size,
                                padding=padding, padding_mode=pad_mode))

        self.conv_layers = nn.Sequential(*layers)
        
        # Apply Xavier/Glorot initialization for better training stability
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
        Transform input field (size-preserving).

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            Transformed tensor of shape (batch_size, H, W) or (H, W)
        """
        # Handle single sample
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        # Add channel dimension: (batch, H, W) → (batch, 1, H, W)
        x = x.unsqueeze(1)

        # Apply convolutions (size-preserving due to padding)
        x_cnn = self.conv_layers(x)

        # Remove channel dimension: (batch, 1, H, W) → (batch, H, W)
        x_cnn = x_cnn.squeeze(1)

        if self.residual:
            # Skip connection: preserve the raw field and add learned correction.
            # x still has channel dim here, so squeeze it first.
            x = x.squeeze(1) + x_cnn
        else:
            x = x_cnn

        if self.standardize:
            # Per-sample standardization: stabilize CNN output scale.
            #
            # Without this, the CNN output magnitude grows unbounded during
            # training (0.07 → 4 → 25 → 100+ → NaN within ~70 epochs),
            # causing persistence diagram points to spread beyond the fixed
            # PI grid → dead features → degenerate Fisher → NaN divergence.
            #
            # Cubical persistence depends ONLY on the relative ordering of
            # pixel values, not their absolute scale. Standardizing to
            # mean=0, std=1 per sample preserves the ordering (and thus
            # the persistence diagram topology) while keeping outputs in a
            # bounded range that the PI grid can consistently capture.
            #
            # Note: this is NOT BatchNorm (which normalizes across the batch
            # and removes inter-sample mean differences that Fisher info
            # needs). This is per-sample standardization — each sample keeps
            # its own relative structure, just rescaled to unit variance.
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            x = (x - mean) / std

        # Remove batch dimension if single sample
        if single_sample:
            x = x.squeeze(0)

        return x


class LearnableFiltration(nn.Module):
    """
    Learnable filtration layer combining CNN transformation and cubical persistence.

    Pipeline:
        Input field (N×N) → CNN → Transformed field (N×N) → Cubical persistence → Diagrams

    The CNN is trained end-to-end to maximize Fisher information by transforming
    the input field in a way that enhances topologically discriminative features.

    Training:
        - Generate fresh samples each epoch (since filtration changes)
        - Compute persistence on CNN-transformed fields
        - Maximize Fisher determinant: minimize -log|F|
        - Gradient flows through CNN and gather operations

    Example:
        >>> # Create learnable filtration
        >>> filtration = LearnableFiltration(
        ...     homology_dimensions=[0, 1],
        ...     hidden_channels=[32, 64, 32]
        ... )
        >>>
        >>> # Forward pass
        >>> field = torch.randn(10, 16, 16, requires_grad=True)
        >>> diagrams = filtration(field)  # List[List[Tensor]]
        >>>
        >>> # diagrams[0] = H0 diagrams (10 diagrams)
        >>> # diagrams[1] = H1 diagrams (10 diagrams)
        >>>
        >>> # Compute loss and train
        >>> loss = some_fisher_loss(diagrams)
        >>> loss.backward()  # Gradients flow to CNN!
    """

    def __init__(
        self,
        homology_dimensions: List[int] = [0, 1],
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        skip_k: int = 1,
        residual: bool = False,
        standardize: bool = True,
        persistence_backend: str = 'gudhi',
        persistence_construction: str = 'T',
        circular_padding: bool = False,
        periodic: bool = False,
    ):
        """
        Initialize learnable filtration.

        Args:
            homology_dimensions: List of homology dimensions to compute
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function
            min_persistence: Minimum persistence threshold for each dimension
            superlevel: If True, compute superlevel filtration (negate values).
                       Use for fields where "valleys" are features of interest,
                       e.g., GRFs with negative spectral index B.
            n_jobs: Number of parallel threads for GUDHI topology computation.
                   1 = serial (default), -1 = use all available CPU cores,
                   >1 = use that many threads. GUDHI releases the GIL so
                   threads give true parallelism for the C++ compute.
            skip_k: Recompute GUDHI topology every skip_k forward passes
                   during training. 1 = always recompute (default/exact).
                   Values 5-10 give significant speedup with minimal accuracy
                   loss. Cache is always bypassed during eval mode.
            residual: If True, use skip connection in CNN: output = input + CNN(input).
                     Preserves the raw field's topology, CNN learns additive corrections.
                     Essential for steep spectra (|B|≥2) where CNN low-pass filtering
                     destroys topological content.
            standardize: If True, apply per-sample standardization in CNN output.
                     Needed for TDA pipelines (PI grid has fixed range). Should be
                     False for CNN-only pipelines where it hurts training stability.
            persistence_backend: Persistence computation backend.
                   'gudhi' (default): CPU-based GUDHI with optional threading.
                   'cmp_gpu': GPU-accelerated CMP CUDA kernels (requires cmp package).
            persistence_construction: Cubical complex construction type ('T' or 'V').
                   'T' (default): T-construction — input pixels are top-dimensional cells.
                   'V': V-construction — input pixels are vertices. Produces diagrams
                        equivalent to CMP GPU. Typically yields higher Fisher information.
                   Only used with gudhi_gpu backend.

        Note:
            BatchNorm is NOT used as it's incompatible with Fisher information
            estimation (removes mean structure which Fisher info measures).
        """
        super().__init__()

        self.homology_dimensions = homology_dimensions
        self.superlevel = superlevel

        # Pre-filtration CNN (learnable) - fully convolutional, works with any input size
        self.cnn = PreFiltrationCNN(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            residual=residual,
            standardize=standardize,
            circular_padding=circular_padding,
        )

        # Differentiable cubical persistence layer (with optional threading + skip-K)
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
        """
        Apply learnable filtration to input fields.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2)
        """
        # Transform field with CNN (differentiable)
        x_transformed = self.cnn(x)

        # Compute persistence on transformed field (differentiable via torch.gather)
        diagrams = self.cubical(x_transformed)

        return diagrams

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_topology_cache_stats(self) -> dict:
        """Return topology cache statistics from the cubical layer."""
        return self.cubical.get_cache_stats()

    def invalidate_topology_cache(self):
        """Clear cached topology indices in the cubical layer."""
        self.cubical.invalidate_cache()
