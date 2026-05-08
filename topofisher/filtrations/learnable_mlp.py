"""
Learnable MLP filtration layer WITHOUT topological analysis.

This module implements a learnable MLP that transforms flattened input fields
directly to a lower-dimensional summary vector, bypassing any TDA.
The MLP is trained to maximize Fisher information.

Training Strategy:
    - MLP transforms field: N×N (flattened) → output_dim vector
    - MOPED compression on the output vector
    - Gradient flows through MLP via Fisher loss: minimize -log|F|
    - Data regenerated each epoch since MLP changes

Pipeline:
    Input (N×N) → Flatten → N² → MLP → output_dim vector

The MLP learns to extract features relevant for parameter inference,
without relying on topological or spatial features.
"""
from typing import List, Optional
import torch
import torch.nn as nn


class PreFiltrationMLP(nn.Module):
    """
    Simple MLP for pre-filtration feature extraction.
    
    Takes flattened input and produces a lower-dimensional summary.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension (N² for N×N field)
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('relu', 'leaky_relu', 'tanh', 'gelu')
            dropout: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        # Select activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Apply Xavier/Glorot initialization for better training stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


class LearnableMLPFiltration(nn.Module):
    """
    Learnable MLP filtration that bypasses TDA.

    Pipeline:
        Input field (N×N) → Flatten → MLP → Vector (output_dim)

    The MLP is trained end-to-end to maximize Fisher information by learning
    feature representations directly, without computing persistence diagrams
    or using convolutional spatial structure.

    The output format is compatible with IdentityVectorization, which expects:
        [[vector_0, vector_1, ..., vector_n]] (single "homology dimension")

    Training:
        - Generate fresh samples each epoch (since MLP changes)
        - Flatten input, pass through MLP to get output_dim vector
        - Apply MOPED compression
        - Maximize Fisher determinant: minimize -log|F|
        - Gradient flows through MLP

    Example:
        >>> # Create learnable MLP filtration
        >>> filtration = LearnableMLPFiltration(
        ...     input_size=32,  # For 32×32 fields
        ...     output_dim=32,  # Output 32-dimensional vector
        ...     hidden_dims=[256, 128]
        ... )
        >>>
        >>> # Forward pass
        >>> field = torch.randn(10, 32, 32, requires_grad=True)
        >>> vectors = filtration(field)  # List[List[Tensor]]
        >>>
        >>> # vectors[0] = list of 10 output vectors
        >>> # Each vector has shape (32,)
        >>>
        >>> # Use with IdentityVectorization to get (10, 32) tensor
        >>> from topofisher.vectorizations import IdentityVectorization
        >>> vec = IdentityVectorization()
        >>> summary = vec(vectors)  # Shape: (10, 32)
    """

    def __init__(
        self,
        input_size: int = 32,
        output_dim: int = 32,
        hidden_dims: List[int] = [256, 128],
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        """
        Initialize learnable MLP filtration.

        Args:
            input_size: Spatial size of input field (N for N×N)
            output_dim: Dimension of output vector
            hidden_dims: MLP hidden layer dimensions
            activation: Activation function ('relu', 'leaky_relu', 'tanh', 'gelu')
            dropout: Dropout rate (0 = no dropout)

        Note:
            BatchNorm is NOT used as it's incompatible with Fisher information
            estimation (removes mean structure which Fisher info measures).
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_dim = output_dim
        input_dim = input_size * input_size  # N²
        
        # MLP (learnable)
        self.mlp = PreFiltrationMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply MLP transformation.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List containing single "homology dimension" with output vectors:
            [[vec_0, vec_1, ..., vec_n]] where each vec_i has shape (output_dim,)
            
            This format is compatible with IdentityVectorization.
        """
        # Handle single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        # Flatten input: (batch, H, W) → (batch, H*W)
        x_flat = x.view(batch_size, -1)
        
        # Apply MLP: (batch, H*W) → (batch, output_dim)
        output = self.mlp(x_flat)
        
        # Format as list of vectors for compatibility with vectorization interface
        # Single "homology dimension" containing all sample vectors
        vectors = [[output[i] for i in range(batch_size)]]
        
        return vectors

    def get_trainable_parameters(self):
        """Return trainable parameters (the MLP)."""
        return self.mlp.parameters()
    
    def __repr__(self):
        return (
            f"LearnableMLPFiltration(\n"
            f"  input_size={self.input_size},\n"
            f"  output_dim={self.output_dim},\n"
            f"  mlp={self.mlp}\n"
            f")"
        )
