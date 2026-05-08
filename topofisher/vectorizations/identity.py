"""
Identity vectorization for vectors (no topological vectorization).

Useful for testing pipelines with simple vector data where the
input is already in vector form.
"""
import torch
import torch.nn as nn
from typing import List


class IdentityVectorization(nn.Module):
    """
    Identity vectorization that stacks vectors back into a batch.

    This is useful when the input data is already vectors (not persistence
    diagrams) and you want to pass them through the pipeline unchanged.

    Expects input from IdentityFiltration: [[vector_0, vector_1, ...]]
    Returns: stacked tensor of shape (n_samples, d)
    """

    def forward(self, diagrams: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Stack vectors back into a batch.

        Args:
            diagrams: List of lists from filtration.
                     Format: [[vector_0, vector_1, ..., vector_n]]
                     (Single homology dimension containing all vectors)

        Returns:
            Tensor of shape (n_samples, d)
        """
        # Extract vectors from the single homology dimension
        vectors = diagrams[0]  # Get the list of vectors from first (and only) homology dim
        return torch.stack(vectors)

    def __repr__(self):
        return "IdentityVectorization()"
