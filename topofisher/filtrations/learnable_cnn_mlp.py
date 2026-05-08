"""
Learnable CNN encoder + MLP head filtration layer WITHOUT topological analysis.

This module implements a learnable CNN encoder that downsamples the input
followed by an MLP head, bypassing any TDA (Topological Data Analysis).
This is more parameter-efficient than pure MLP for large inputs.

Training Strategy:
    - CNN encoder: N×N → reduced spatial dims (e.g., 8×8)
    - Flatten reduced features
    - MLP head: flattened features → output_dim vector
    - MOPED compression on the output vector
    - Gradient flows through CNN+MLP via Fisher loss: minimize -log|F|

Pipeline:
    Input (N×N) → CNN Encoder → (8×8 × channels) → Flatten → MLP → output_dim

Advantages over pure MLP:
    - Parameter efficiency: CNN shares weights spatially
    - Scales better: 64×64 input with pure MLP = 2M+ params
                     64×64 input with CNN+MLP = ~50k params
    - Preserves spatial locality in early layers
"""
from typing import List, Optional
import torch
import torch.nn as nn


class CNNEncoderMLP(nn.Module):
    """
    CNN encoder + MLP head for efficient feature extraction.
    
    The CNN uses strided convolutions to reduce spatial dimensions,
    then an MLP processes the flattened features.
    """
    
    def __init__(
        self,
        input_size: int = 64,
        encoder_channels: List[int] = [16, 32, 16],
        encoder_strides: List[int] = [2, 2, 2],
        kernel_size: int = 5,
        mlp_hidden_dims: List[int] = [64],
        output_dim: int = 32,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        """
        Initialize CNN encoder + MLP.
        
        Args:
            input_size: Input spatial size (N for N×N field)
            encoder_channels: List of CNN channel dimensions
            encoder_strides: List of strides for each conv layer (2 = downsample by 2)
            kernel_size: Convolution kernel size
            mlp_hidden_dims: List of MLP hidden layer dimensions
            output_dim: Final output dimension
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
            dropout: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        # Select activation
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build CNN encoder
        encoder_layers = []
        in_channels = 1
        
        for out_channels, stride in zip(encoder_channels, encoder_strides):
            padding = kernel_size // 2
            encoder_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding)
            )
            encoder_layers.append(act_fn())
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate spatial size after encoding
        spatial_size = input_size
        for stride in encoder_strides:
            spatial_size = (spatial_size + stride - 1) // stride  # ceiling division
        
        self.encoded_features = encoder_channels[-1] * spatial_size * spatial_size
        
        # Build MLP head
        mlp_layers = []
        prev_dim = self.encoded_features
        
        for hidden_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(act_fn())
            if dropout > 0:
                mlp_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        mlp_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Apply Xavier initialization
        self._init_weights()
        
        # Store config for debugging
        self.input_size = input_size
        self.output_dim = output_dim
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, H, W) or (batch_size, 1, H, W)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Ensure 4D input: (batch, channel, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        
        # CNN encoder
        encoded = self.encoder(x)
        
        # Flatten
        flattened = encoded.view(encoded.size(0), -1)
        
        # MLP head
        output = self.mlp(flattened)
        
        return output


class LearnableCNNMLPFiltration(nn.Module):
    """
    Learnable CNN encoder + MLP head filtration that bypasses TDA.

    Pipeline:
        Input field (N×N) → CNN Encoder → (reduced) → Flatten → MLP → Vector

    This is more parameter-efficient than pure MLP for large inputs (N > 32).

    Training:
        - Generate fresh samples each epoch (since network changes)
        - Apply MOPED compression
        - Maximize Fisher determinant: minimize -log|F|
        - Gradient flows through entire network

    Example:
        >>> filtration = LearnableCNNMLPFiltration(
        ...     input_size=64,
        ...     encoder_channels=[16, 32, 16],
        ...     mlp_hidden_dims=[64],
        ...     output_dim=32
        ... )
        >>>
        >>> field = torch.randn(10, 64, 64, requires_grad=True)
        >>> vectors = filtration(field)  # List[List[Tensor]]
    """

    def __init__(
        self,
        input_size: int = 64,
        encoder_channels: List[int] = [16, 32, 16],
        encoder_strides: List[int] = [2, 2, 2],
        kernel_size: int = 5,
        mlp_hidden_dims: List[int] = [64],
        output_dim: int = 32,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        """
        Initialize learnable CNN+MLP filtration.

        Args:
            input_size: Input spatial size (N for N×N field)
            encoder_channels: CNN encoder channel dimensions
            encoder_strides: Strides for each conv layer
            kernel_size: Convolution kernel size
            mlp_hidden_dims: MLP hidden layer dimensions
            output_dim: Final output dimension
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()

        self.network = CNNEncoderMLP(
            input_size=input_size,
            encoder_channels=encoder_channels,
            encoder_strides=encoder_strides,
            kernel_size=kernel_size,
            mlp_hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply CNN+MLP transformation.

        Args:
            x: Input tensor of shape (batch_size, H, W) or (H, W)

        Returns:
            List of lists of vectors, formatted for IdentityVectorization:
            [[vec_0, vec_1, ..., vec_n]] where each vec_i has shape (output_dim,)
        """
        # Handle single sample
        single_sample = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_sample = True

        # Forward through network
        output = self.network(x)  # (batch_size, output_dim)

        # Format output for IdentityVectorization
        vectors = [output[i] for i in range(output.shape[0])]
        
        if single_sample:
            return [[vectors[0]]]
        else:
            return [vectors]

    def parameters(self, recurse: bool = True):
        """Return network parameters for training."""
        return self.network.parameters(recurse=recurse)
