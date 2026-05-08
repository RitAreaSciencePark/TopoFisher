"""Pipeline for training learnable vectorization."""

from typing import List
import torch
import torch.nn as nn

from .base import LearnablePipeline


class LearnableVectorizationPipeline(LearnablePipeline):
    """
    Pipeline for training vectorization from persistence diagrams.

    This pipeline takes persistence diagrams and trains a vectorization
    component (e.g., learnable persistence images) to maximize Fisher information.
    PyTorch automatically tracks and trains any learnable parameters.
    """

    def _compute_summaries(self, data: List) -> List[torch.Tensor]:
        """
        Compute compressed summaries from persistence diagrams.

        Pipeline: diagrams → vectorization → compression

        Args:
            data: Persistence diagrams [fid, minus_0, plus_0, ...]

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        summaries = self.vectorization(data)
        compressed = self.compression(summaries)
        return compressed

    def generate_diagrams(self, config):
        """
        Helper to generate persistence diagrams from scratch.

        Args:
            config: AnalysisConfig with parameters

        Returns:
            List of persistence diagrams [fid, minus_0, plus_0, ...]
        """
        # Generate raw data
        all_data = self.generate_data(config)

        # Compute persistence diagrams
        all_diagrams = self.compute_diagrams(all_data, verbose=False)

        return all_diagrams