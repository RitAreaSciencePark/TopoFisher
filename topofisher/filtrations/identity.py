"""
Identity filtration for vectors (no topological computation).

Useful for testing pipelines with simple vector data where no
persistence diagrams are needed.
"""
from typing import List
import torch
import torch.nn as nn


class IdentityFiltration(nn.Module):
    """
    Identity filtration that passes vectors through unchanged.

    This is useful for testing the pipeline with vector data
    that doesn't need topological analysis. Each vector is
    returned as-is, wrapped for compatibility with the pipeline.
    """

    def forward(self, data: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Pass vectors through unchanged.

        Args:
            data: Vectors of shape (n_samples, d) or (n_samples, ...)

        Returns:
            List of lists (for homology dimension compatibility).
            Returns [[vector_0, vector_1, ..., vector_n]] - single homology dim
        """
        # Return list of lists for homology dimension compatibility
        # Single "homology dimension" containing all vectors
        return [[data[i] for i in range(len(data))]]

    def __repr__(self):
        return "IdentityFiltration()"
