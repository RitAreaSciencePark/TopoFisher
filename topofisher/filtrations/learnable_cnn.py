"""
Learnable CNN filtration layer WITHOUT topological analysis.

This module implements a learnable CNN that transforms input fields
and flattens the output directly, bypassing any TDA (Topological Data Analysis).
The CNN is trained to maximize Fisher information.

Training Strategy:
    - CNN transforms field: N×N → N×N (learnable, size-preserving)
    - Output is flattened to N^2 vector
    - MOPED compression on the flattened vector
    - Gradient flows through CNN via Fisher loss: minimize -log|F|
    - Data regenerated each epoch since CNN changes

Pipeline:
    Input (N×N) → CNN → Transformed (N×N) → Flatten → N^2 vector

The CNN learns to extract features relevant for parameter inference,
without relying on topological features.
"""
from typing import List, Optional
import torch
import torch.nn as nn

from .learnable import PreFiltrationCNN


class LearnableCNNFiltration(nn.Module):
    """
    Learnable CNN filtration that bypasses TDA.

    Pipeline:
        Input field (N×N) → CNN → Transformed field (N×N) → Flatten → Vector (N^2)

    The CNN is trained end-to-end to maximize Fisher information by learning
    feature representations directly, without computing persistence diagrams.

    The output format is compatible with IdentityVectorization, which expects:
        [[vector_0, vector_1, ..., vector_n]] (single "homology dimension")

    Training:
        - Generate fresh samples each epoch (since CNN changes)
        - Flatten CNN output to N^2 vector
        - Apply MOPED compression
        - Maximize Fisher determinant: minimize -log|F|
        - Gradient flows through CNN

    Example:
        >>> # Create learnable CNN filtration
        >>> filtration = LearnableCNNFiltration(
        ...     hidden_channels=[32, 64, 32]
        ... )
        >>>
        >>> # Forward pass
        >>> field = torch.randn(10, 16, 16, requires_grad=True)
        >>> vectors = filtration(field)  # List[List[Tensor]]
        >>>
        >>> # vectors[0] = list of 10 flattened vectors
        >>> # Each vector has shape (256,) for 16x16 input
        >>>
        >>> # Use with IdentityVectorization to get (10, 256) tensor
        >>> from topofisher.vectorizations import IdentityVectorization
        >>> vec = IdentityVectorization()
        >>> summary = vec(vectors)  # Shape: (10, 256)
    """

    def __init__(
        self,
        hidden_channels: List[int] = [32, 64, 32],
        kernel_size: int = 3,
        activation: str = 'relu',
        standardize: bool = False,
        circular_padding: bool = False,
    ):
        """
        Initialize learnable CNN filtration.

        Args:
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function ('relu', 'leaky_relu', 'tanh')
            standardize: If True, apply per-sample standardization in CNN.
                Defaults to False for CNN-only pipelines (no persistence
                diagrams), since standardization removes useful scale
                information and destabilizes training.
            circular_padding: If True, use padding_mode='circular' on all Conv2d
                layers. Should be True for periodic input fields.

        Note:
            BatchNorm is NOT used as it's incompatible with Fisher information
            estimation (removes mean structure which Fisher info measures).
        """
        super().__init__()

        # Pre-filtration CNN (learnable) - fully convolutional, works with any input size
        self.cnn = PreFiltrationCNN(
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            standardize=standardize,
            circular_padding=circular_padding,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply CNN transformation and flatten output.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of flattened vectors for pipeline compatibility.
            Format: [[vector_0, vector_1, ..., vector_n]]
            Each vector has shape (H * W,)
        """
        # Handle single sample
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        batch_size = x.shape[0]

        # Transform field with CNN (differentiable)
        x_transformed = self.cnn(x)  # Shape: (batch_size, H, W)

        # Flatten each sample: (batch_size, H, W) → list of (H*W,) vectors
        vectors = []
        for i in range(batch_size):
            flattened = x_transformed[i].flatten()  # Shape: (H * W,)
            vectors.append(flattened)

        # Return in format compatible with IdentityVectorization
        # Single "homology dimension" containing all vectors
        return [vectors]

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        """String representation showing configuration."""
        return (f"LearnableCNNFiltration("
                f"hidden_channels={list(self.cnn.conv_layers)}, "
                f"num_params={self.get_num_parameters()})")
