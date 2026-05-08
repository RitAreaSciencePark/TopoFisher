"""
MLP-based learned compression.

Multi-layer Perceptron trained to maximize Fisher information.
Typically trained on Top-K vectorizations of persistence diagrams.

Uses PyTorch's nn.LazyLinear for automatic dimension inference on first forward pass.
"""
from typing import List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn

from . import Compression


class MLPCompression(Compression):
    """
    Multi-layer Perceptron for compressing summaries to maximize Fisher information.

    Uses PyTorch's lazy initialization - input dimension is automatically inferred
    on the first forward pass. This eliminates manual initialization tracking.
    """

    _ACTIVATIONS = {
        'gelu':       nn.GELU,
        'relu':       nn.ReLU,
        'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01),
        'silu':       nn.SiLU,
        'tanh':       nn.Tanh,
        'none':       None,
    }

    def __init__(
        self,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        activation: str = 'gelu',
    ):
        """
        Initialize MLP compression.

        Args:
            output_dim: Output dimension (typically n_params)
            hidden_dims: List of hidden layer dimensions.
                        None or [] for linear compression (no hidden layers)
                        [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
            dropout: Dropout probability (default 0.2)
            activation: Hidden-layer activation — 'gelu' (default), 'relu',
                'leaky_relu', 'silu', 'tanh', or 'none' (linear).
                GELU is preferred over LeakyReLU: it is smooth and symmetric
                around zero, so for approximately-Gaussian inputs the hidden
                activations remain close to Gaussian, making the final output
                easier to pass the KS-test Gaussianity check.
        """
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        if activation not in self._ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(self._ACTIVATIONS)}"
            )
        self.activation = activation

        # Build network
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Build MLP network with lazy first layer."""
        act_factory = self._ACTIVATIONS[self.activation]

        layers = []

        if not self.hidden_dims:
            # Linear compression (no hidden layers)
            layers.append(nn.LazyLinear(self.output_dim))
        else:
            # First layer is lazy (input dimension inferred on first forward)
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            if act_factory is not None:
                layers.append(act_factory())
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

            # Middle layers are regular (dimensions known)
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                if act_factory is not None:
                    layers.append(act_factory())
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))

            # Final layer (no activation — linear projection preserves Gaussianity)
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        return nn.Sequential(*layers)

    def is_trainable(self) -> bool:
        """Return True since MLP compression requires training."""
        return True

    def is_initialized(self) -> bool:
        """
        Check if lazy layers have been materialized.

        Returns:
            True if first forward pass has occurred and dimensions are known
        """
        first_layer = self.network[0]
        if isinstance(first_layer, nn.LazyLinear):
            return not first_layer.has_uninitialized_params()
        return True

    def forward(
        self,
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply MLP compression to summaries.

        On first call, lazy layers are automatically initialized based on input dimensions.

        Args:
            summaries: List of summary tensors

        Returns:
            Compressed summaries
        """
        # Apply network to each summary tensor
        # First forward automatically initializes lazy layers
        compressed = [self.network(s) for s in summaries]
        return compressed

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> 'MLPCompression':
        """
        Load a pre-trained MLP compression from disk.

        Args:
            path: Path to saved model (.pth or .pt file)

        Returns:
            Loaded MLPCompression instance
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

        # Create instance (dimensions will be inferred from state_dict)
        model = cls(
            output_dim=arch['output_dim'],
            hidden_dims=arch.get('hidden_dims'),
            dropout=arch.get('dropout', 0.2)
        )

        # Load state dict (this will materialize lazy layers)
        model.load_state_dict(checkpoint['state_dict'])

        return model

    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.

        Args:
            path: Path to save model (.pth or .pt file)
        """
        if not self.is_initialized():
            raise RuntimeError(
                "Cannot save uninitialized MLPCompression. "
                "Run a forward pass first to initialize lazy layers."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Get input dimension from materialized first layer
        first_layer = self.network[0]
        input_dim = first_layer.in_features

        checkpoint = {
            'state_dict': self.state_dict(),
            'architecture': {
                'input_dim': input_dim,
                'output_dim': self.output_dim,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout_prob
            }
        }

        torch.save(checkpoint, path)

    def __repr__(self):
        if not self.is_initialized():
            if self.hidden_dims:
                return f"MLPCompression(uninitialized, hidden_dims={self.hidden_dims}, output_dim={self.output_dim}, dropout={self.dropout_prob})"
            else:
                return f"MLPCompression(uninitialized, linear, output_dim={self.output_dim})"
        else:
            # Get input dimension from materialized first layer
            first_layer = self.network[0]
            input_dim = first_layer.in_features

            if not self.hidden_dims:
                return f"MLPCompression(Linear: {input_dim} → {self.output_dim})"
            else:
                dims_str = f"{input_dim} → " + " → ".join(map(str, self.hidden_dims)) + f" → {self.output_dim}"
                return f"MLPCompression({dims_str}, dropout={self.dropout_prob})"
