"""Cached pipeline for training learnable compression from cached data."""

from typing import List
import torch

from .base import CachedPipeline
from ..learnable.compression import LearnableCompressionPipeline
from ...config import AnalysisConfig


class CachedCompressionPipeline(CachedPipeline, LearnableCompressionPipeline):
    """
    Pipeline for training compression from cached diagrams or summaries.

    This pipeline is used in "load" mode to:
    1. Load cached diagrams or summaries from disk
    2. Vectorize diagrams if needed
    3. Train compression to maximize Fisher information

    Use GenerateCachePipeline first to create cached data.
    """

    def generate_data(self, config: AnalysisConfig) -> List[torch.Tensor]:
        """
        Load summaries from cache instead of generating from scratch.

        This overrides the parent method to load from disk, maintaining
        compatibility with the existing pipeline flow.

        Args:
            config: AnalysisConfig with cache configuration

        Returns:
            List of summary tensors [fid, minus_0, plus_0, ...]
        """
        if not config.cache:
            raise ValueError(
                "Cache configuration required for CachedCompressionPipeline. "
                "Set cache.mode='load' and cache.load_path in your config."
            )

        cache_config = config.cache

        if cache_config.mode != "load":
            raise ValueError(
                f"CachedCompressionPipeline requires mode='load', got '{cache_config.mode}'. "
                "Use GenerateCachePipeline for mode='generate'."
            )

        if not cache_config.load_path:
            raise ValueError("load_path required for load mode")

        # Load based on data_type
        if cache_config.data_type == "summaries":
            print(f"\nLoading summaries from: {cache_config.load_path}")
            summaries = self.load_summaries(cache_config.load_path)

        elif cache_config.data_type == "diagrams":
            print(f"\nLoading diagrams from: {cache_config.load_path}")
            diagrams = self.load_diagrams(cache_config.load_path)

            print("Vectorizing diagrams to summaries...")
            summaries = self.vectorize(diagrams)

        else:
            raise ValueError(
                f"Unknown data_type: {cache_config.data_type}. "
                "Use 'diagrams' or 'summaries'."
            )

        return summaries