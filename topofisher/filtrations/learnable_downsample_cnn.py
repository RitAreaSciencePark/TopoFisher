"""
Learnable downsampling CNN filtration WITHOUT topological analysis.

This module implements the same DownsamplingCNN architecture as
LearnableDownsampleFiltration, but flattens the CNN output directly
instead of computing persistence diagrams. This serves as an ablation
baseline to isolate the contribution of the TDA bottleneck.

Pipeline:
    Input (N×N) → DownsamplingCNN → (K×K) → Flatten → K² vector

The CNN learns to compress spatial information for Fisher information
maximization. Without topological constraints, the network must extract
all useful information through direct learned features alone.

Ablation purpose:
    Compare with LearnableDownsampleFiltration (same CNN + persistence)
    to show that the topological bottleneck provides an important
    inductive bias beyond what the CNN alone can learn.
"""
from typing import List, Optional
import torch
import torch.nn as nn

from .learnable_downsample import DownsamplingCNN


class LearnableDownsampleCNNFiltration(nn.Module):
    """
    Learnable downsampling CNN filtration that bypasses TDA.

    Uses the same DownsamplingCNN architecture as LearnableDownsampleFiltration
    but flattens the output to a vector instead of computing cubical persistence.
    This is the CNN-only ablation for high-resolution (e.g. 512×512) inputs.

    Pipeline:
        Input (N×N) → DownsamplingCNN → (K×K) → Flatten → K² vector

    The output format is compatible with IdentityVectorization, which expects:
        [[vector_0, vector_1, ..., vector_n]] (single "homology dimension")

    Example:
        >>> filtration = LearnableDownsampleCNNFiltration(
        ...     target_size=64, hidden_channels=[32, 32, 32],
        ...     kernel_size=5, input_size=512
        ... )
        >>> field = torch.randn(10, 512, 512, requires_grad=True)
        >>> vectors = filtration(field)  # List[List[Tensor]]
        >>> # vectors[0] = list of 10 flattened vectors, each shape (4096,)
    """

    def __init__(
        self,
        target_size: int = 64,
        hidden_channels: List[int] = [32, 32, 32],
        kernel_size: int = 5,
        activation: str = 'leaky_relu',
        standardize: bool = False,
        input_size: int = 512,
        gradient_checkpointing: bool = True,
        circular_padding: bool = False,
        # Accept and ignore persistence-specific params for config compatibility
        homology_dimensions: Optional[List[int]] = None,
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        skip_k: int = 1,
        persistence_backend: str = 'gudhi',
        persistence_construction: str = 'T',
        gpu_sub_batch_size: int = 100,
    ):
        """
        Initialize learnable downsampling CNN filtration (no TDA).

        Args:
            target_size: Target spatial resolution after downsampling
            hidden_channels: CNN hidden channel dimensions
            kernel_size: CNN convolution kernel size (must be odd)
            activation: CNN activation function ('relu', 'leaky_relu', 'tanh')
            standardize: If True, per-sample standardize CNN output
            input_size: Expected input spatial resolution
            gradient_checkpointing: If True, use gradient checkpointing on CNN
            circular_padding: If True, use padding_mode='circular' on all Conv2d
                layers. Should be True for periodic input fields.

        Note:
            The persistence-specific parameters (homology_dimensions,
            min_persistence, superlevel, n_jobs, skip_k, persistence_backend,
            persistence_construction, gpu_sub_batch_size) are accepted but
            ignored. This allows using this filtration as a drop-in replacement
            for LearnableDownsampleFiltration in existing configs by only
            changing the filtration type.
        """
        super().__init__()

        self.target_size = target_size
        self.standardize = standardize
        self.cnn_eval_batch_size = 200

        # Same DownsamplingCNN as LearnableDownsampleFiltration
        self.cnn = DownsamplingCNN(
            target_size=target_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            activation=activation,
            input_size=input_size,
            gradient_checkpointing=gradient_checkpointing,
            circular_padding=circular_padding,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply downsampling CNN and flatten output.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of flattened vectors for pipeline compatibility.
            Format: [[vector_0, vector_1, ..., vector_n]]
            Each vector has shape (target_size * target_size,)
        """
        single_sample = (x.ndim == 2)
        if single_sample:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]

        # Downsample: (B, N, N) → (B, target_size, target_size)
        # Sub-batch during eval to avoid OOM on large validation sets
        if not self.training and batch_size > self.cnn_eval_batch_size:
            chunks = []
            for i in range(0, batch_size, self.cnn_eval_batch_size):
                chunks.append(self.cnn(x[i:i + self.cnn_eval_batch_size]))
            x_down = torch.cat(chunks, dim=0)
        else:
            x_down = self.cnn(x)

        if self.standardize:
            if x_down.ndim == 3:
                mean = x_down.mean(dim=(-2, -1), keepdim=True)
                std = x_down.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                x_down = (x_down - mean) / std
            elif x_down.ndim == 2:
                mean = x_down.mean()
                std = x_down.std().clamp(min=1e-6)
                x_down = (x_down - mean) / std

        # Flatten each sample: (B, K, K) → list of (K²,) vectors
        vectors = []
        for i in range(batch_size):
            vectors.append(x_down[i].flatten())

        # Return in format compatible with IdentityVectorization
        return [vectors]

    def get_cnn_parameters(self):
        """Return CNN parameters for training."""
        return self.cnn.parameters()

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"LearnableDownsampleCNNFiltration("
            f"target_size={self.target_size}, "
            f"num_params={self.get_num_parameters()})"
        )
