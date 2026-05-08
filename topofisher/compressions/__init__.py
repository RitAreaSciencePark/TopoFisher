"""
Compression methods for TopoFisher pipeline.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn as nn


class Compression(nn.Module, ABC):
    """Base class for compression methods."""

    @abstractmethod
    def forward(
        self,
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply compression to summaries.

        Args:
            summaries: [fid, minus_0, plus_0, minus_1, plus_1, ...]

        Returns:
            Compressed summaries with same structure
        """
        pass


class IdentityCompression(Compression):
    """Pass-through compression (no compression)."""

    def forward(
        self,
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return summaries


# Import concrete implementations
from .moped import MOPEDCompression
from .mlp import MLPCompression
from .cnn import CNNCompression

__all__ = [
    'Compression',
    'IdentityCompression',
    'MOPEDCompression',
    'MLPCompression',
    'CNNCompression',
]