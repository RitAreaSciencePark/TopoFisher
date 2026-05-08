"""
CNN-based learned compression for persistence images.

Convolutional Neural Network trained to maximize Fisher information.
Operates on persistence images (2D representations of persistence diagrams).

Based on Yip et al. 2025 (https://doi.org/10.1088/2632-2153/ade114)
"""
from typing import List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from . import Compression


class CNNCompression(Compression):
    """
    CNN for compressing persistence images to maximize Fisher information.

    Architecture: Parallel CNN + Dense with LeakyReLU and Dropout.
    Input: (batch, n_channels, n_pixels, n_pixels) where n_channels = number of homology dimensions.
    Output: (batch, output_dim) where output_dim = number of parameters.
    """

    def __init__(
        self,
        n_channels: int,
        n_pixels: int,
        output_dim: int,
        dropout: float = 0.2,
        use_dense_path: bool = False
    ):
        """
        Initialize CNN compression.

        Args:
            n_channels: Number of input channels (homology dimensions, e.g., 2 for H0+H1)
            n_pixels: Resolution of persistence images (n_pixels x n_pixels)
            output_dim: Output dimension (n_params)
            dropout: Dropout probability (default 0.2)
            use_dense_path: If True, use both CNN and Dense paths (averaged).
                           If False, use only CNN path (default)
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_pixels = n_pixels
        self.output_dim = output_dim
        self.dropout_prob = dropout
        self.use_dense_path = use_dense_path

        # CNN side (operates on 2D images)
        # Architecture: conv blocks with maxpool, progressively increasing channels
        # With padding=1, conv preserves spatial size, maxpool halves it
        # n_pixels -> n_pixels/2 -> n_pixels/4 -> ... -> 2x2

        cnn_layers = []

        # Initial expansion: n_channels -> 32 channels
        cnn_layers.extend([
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU()
        ])

        # Number of blocks: keep going until we reach 2x2
        n_blocks = int(np.log2(n_pixels)) - 1  # 16->3, 32->4, 64->5

        in_ch = 32
        for i in range(n_blocks):
            out_ch = 32 * (2 ** min(i, 3))  # 32, 64, 128, 256, 256, ...
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # padding=1 preserves size
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # halves size
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_pixels, n_pixels)
            cnn_out = self.cnn(dummy)
            cnn_flat_size = cnn_out.view(1, -1).shape[1]

        # CNN dense layers
        self.cnn_dense = nn.Sequential(
            nn.Linear(cnn_flat_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        # Dense side (operates on summed 1D features)
        # Sum along birth and persistence axes: n_channels x 2 x n_pixels -> n_channels * 2 * n_pixels
        dense_input_size = n_channels * 2 * n_pixels
        self.dense_stack = nn.Sequential(
            nn.Linear(dense_input_size, 512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(
        self,
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply CNN compression to persistence image summaries.

        Args:
            summaries: List of persistence image tensors.
                      Each tensor has shape (batch, n_channels, n_pixels, n_pixels)

        Returns:
            Compressed summaries with shape (batch, output_dim)
        """
        # Apply CNN to each summary tensor
        compressed = []
        for s in summaries:
            # Handle 3D input (batch, H, W) by adding channel dimension
            if s.dim() == 3:
                s = s.unsqueeze(1)  # (batch, H, W) -> (batch, 1, H, W)
            
            if not self.use_dense_path:
                # CNN only path
                cnn_features = self.cnn(s)
                cnn_features = cnn_features.view(cnn_features.size(0), -1)
                out = self.cnn_dense(cnn_features)
            else:
                # CNN + Dense paths (averaged)
                # CNN side
                cnn_features = self.cnn(s)
                cnn_features = cnn_features.view(cnn_features.size(0), -1)
                cnn_out = self.cnn_dense(cnn_features)

                # Dense side - sum along spatial dimensions
                sum_birth = s.sum(dim=2)  # (batch, n_channels, n_pixels)
                sum_pers = s.sum(dim=3)   # (batch, n_channels, n_pixels)
                dense_features = torch.cat([sum_birth, sum_pers], dim=2)  # (batch, n_channels, 2*n_pixels)
                dense_features = dense_features.view(s.size(0), -1)  # (batch, n_channels * 2 * n_pixels)
                dense_out = self.dense_stack(dense_features)

                # Average outputs from both sides
                out = (cnn_out + dense_out) / 2

            compressed.append(out)

        return compressed

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'CNNCompression':
        """
        Load a pre-trained CNN compression from disk.

        Args:
            path: Path to saved model (.pth or .pt file)

        Returns:
            Loaded CNNCompression instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location='cpu')

        # Extract architecture from checkpoint
        if 'architecture' not in checkpoint:
            raise ValueError(f"Checkpoint missing 'architecture' field: {path}")

        arch = checkpoint['architecture']

        # Create instance
        model = cls(
            n_channels=arch['n_channels'],
            n_pixels=arch['n_pixels'],
            output_dim=arch['output_dim'],
            dropout=arch.get('dropout', 0.2),
            use_dense_path=arch.get('use_dense_path', False)
        )

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.

        Args:
            path: Path to save model (.pth or .pt file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'state_dict': self.state_dict(),
            'architecture': {
                'n_channels': self.n_channels,
                'n_pixels': self.n_pixels,
                'output_dim': self.output_dim,
                'dropout': self.dropout_prob,
                'use_dense_path': self.use_dense_path
            }
        }

        torch.save(checkpoint, path)

    def print_layer_shapes(self, input_shape: tuple):
        """
        Print how tensor shapes change through the network.

        Args:
            input_shape: Tuple (n_channels, height, width)
        """
        print("\nCNN Layer Shape Transformations:")
        print("=" * 60)

        # Get device from model parameters
        device = next(self.parameters()).device
        x = torch.zeros(1, *input_shape, device=device)
        print(f"Input: {tuple(x.shape)}")

        # Track through CNN layers
        for i, layer in enumerate(self.cnn):
            x = layer(x)
            layer_name = layer.__class__.__name__
            print(f"  Layer {i} ({layer_name}): {tuple(x.shape)}")

        # Flatten
        x_flat = x.view(x.size(0), -1)
        print(f"Flatten: {tuple(x_flat.shape)}")

        # CNN dense layers
        for i, layer in enumerate(self.cnn_dense):
            x_flat = layer(x_flat)
            layer_name = layer.__class__.__name__
            print(f"  CNN Dense {i} ({layer_name}): {tuple(x_flat.shape)}")

        print(f"Final output: {tuple(x_flat.shape)}")
        print("=" * 60)

    def __repr__(self):
        mode = "CNN+Dense" if self.use_dense_path else "CNN"
        return (f"CNNCompression({mode}, {self.n_channels} channels, {self.n_pixels}x{self.n_pixels} → "
                f"{self.output_dim}, dropout={self.dropout_prob})")
