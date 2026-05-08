"""Pipeline for generating and caching diagrams/summaries."""

from typing import List, Dict, Any
import torch

from .base import CachedPipeline
from ...config import AnalysisConfig


class GenerateCachePipeline(CachedPipeline):
    """
    Pipeline for generating and caching persistence diagrams or summaries.

    This pipeline is used in "generate" mode to:
    1. Generate raw simulation data
    2. Compute persistence diagrams
    3. Optionally vectorize to summaries
    4. Save to disk for later training

    Use this pipeline first to generate cached data, then use
    CachedCompressionPipeline to load and train.
    """

    def run(self, config: AnalysisConfig) -> Dict[str, Any]:
        """
        Generate and cache data based on cache configuration.

        Args:
            config: AnalysisConfig with cache settings specifying what to save

        Returns:
            Dictionary with generation results and paths
        """
        if not config.cache:
            raise ValueError("Cache configuration required for GenerateCachePipeline")

        cache_config = config.cache

        if cache_config.mode != "generate":
            raise ValueError(
                f"GenerateCachePipeline requires mode='generate', got '{cache_config.mode}'"
            )

        # 1. Generate raw simulation data
        print("\n" + "=" * 60)
        print("Generating Raw Data")
        print("=" * 60)
        all_data = self.generate_data(config)
        print(f"  Generated {len(all_data)} datasets")
        print(f"  Fiducial samples: {all_data[0].shape[0]}")
        print(f"  Derivative samples: {all_data[1].shape[0]}")

        # 2. Compute persistence diagrams
        print("\n" + "=" * 60)
        print("Computing Persistence Diagrams")
        print("=" * 60)
        all_diagrams = self.compute_diagrams(all_data)
        print(f"  Computed diagrams for {len(all_diagrams)} datasets")
        print(f"  Homology dimensions: {len(all_diagrams[0])}")

        # 3. Save based on data_type
        result = {
            'n_datasets': len(all_diagrams),
            'n_fid_samples': all_data[0].shape[0],
            'n_deriv_samples': all_data[1].shape[0],
        }

        if cache_config.data_type == "diagrams":
            # Save diagrams only
            if not cache_config.save_path:
                raise ValueError("save_path required for generate mode")

            self.save_diagrams(all_diagrams, cache_config.save_path, config)
            result['saved_diagrams'] = cache_config.save_path
            result['data_type'] = 'diagrams'

        elif cache_config.data_type == "summaries":
            # Vectorize and save summaries
            print("\n" + "=" * 60)
            print("Vectorizing to Summaries")
            print("=" * 60)
            all_summaries = self.vectorize(all_diagrams)
            print(f"  Feature dimension: {all_summaries[0].shape[-1]}")

            if not cache_config.save_path:
                raise ValueError("save_path required for generate mode")

            self.save_summaries(all_summaries, cache_config.save_path, config)
            result['saved_summaries'] = cache_config.save_path
            result['feature_dim'] = all_summaries[0].shape[-1]
            result['data_type'] = 'summaries'

        else:
            raise ValueError(
                f"Unknown data_type: {cache_config.data_type}. "
                "Use 'diagrams' or 'summaries'."
            )

        print("\n" + "=" * 60)
        print("Generation Complete")
        print("=" * 60)

        return result
