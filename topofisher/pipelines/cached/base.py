"""Base class for cached pipelines with diagram saving/loading capabilities."""

import os
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import torch

from ..learnable.base import LearnablePipeline
from ...config import AnalysisConfig


class CachedPipeline(LearnablePipeline):
    """
    Pipeline with caching capabilities for persistence diagrams and summaries.

    This class extends LearnablePipeline to add functionality for saving
    and loading persistence diagrams to/from disk, enabling reuse across
    different training runs.
    """

    def generate_metadata(self, config: AnalysisConfig) -> Dict[str, Any]:
        """
        Generate metadata dictionary from pipeline configuration.

        Args:
            config: AnalysisConfig containing analysis parameters

        Returns:
            Dictionary containing metadata for caching
        """
        metadata = {
            'theta_fid': config.theta_fid.cpu().numpy().tolist(),
            'delta_theta': config.delta_theta.cpu().numpy().tolist(),
            'n_s': config.n_s,
            'n_d': config.n_d,
            'seed_cov': config.seed_cov,
            'seed_ders': config.seed_ders,
            'created_at': datetime.now().isoformat()
        }

        # Add component information if available
        if self.simulator is not None:
            metadata['simulator'] = {
                'type': self.simulator.__class__.__name__,
                'params': getattr(self.simulator, 'get_config', lambda: {})()
            }
        if self.filtration is not None:
            metadata['filtration'] = {
                'type': self.filtration.__class__.__name__,
                'params': getattr(self.filtration, 'get_config', lambda: {})()
            }
        if self.vectorization is not None:
            metadata['vectorization'] = self.vectorization.__class__.__name__

        return metadata

    def save_diagrams(
        self,
        diagrams: List[List[List[torch.Tensor]]],
        save_path: str,
        config: AnalysisConfig
    ) -> None:
        """
        Save persistence diagrams to disk with metadata.

        Args:
            diagrams: List of diagram sets, structure: [dataset][hom_dim][sample]
                     Each dataset = [fid, minus_0, plus_0, minus_1, plus_1, ...]
                     Each hom_dim = diagrams for one homology dimension
                     Each sample = one persistence diagram
            save_path: Path to save the diagrams
            config: AnalysisConfig containing analysis parameters
        """
        # Convert diagrams to CPU and numpy for storage
        diagrams_np = []
        for diagram_set in diagrams:
            diagram_set_np = []
            for hom_dim_diagrams in diagram_set:
                sample_diagrams_np = []
                for d in hom_dim_diagrams:
                    if isinstance(d, torch.Tensor):
                        d_np = d.cpu().numpy()
                    else:
                        # Handle other formats if needed
                        d_np = d
                    sample_diagrams_np.append(d_np)
                diagram_set_np.append(sample_diagrams_np)
            diagrams_np.append(diagram_set_np)

        # Use generate_metadata for consistent metadata creation
        metadata = self.generate_metadata(config)
        metadata['data_type'] = 'diagrams'

        # Package data
        data = {
            'diagrams': diagrams_np,
            'metadata': metadata
        }

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # Save with pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n{'='*60}")
        print(f"Saved diagrams to: {save_path}")
        print(f"  Datasets: {len(diagrams)} ([fid, minus_0, plus_0, ...])")
        print(f"  Homology dimensions: {len(diagrams[0])}")
        print(f"  Samples per dataset: {len(diagrams[0][0])}")
        print(f"{'='*60}\n")

    def load_diagrams(self, load_path: str) -> List[List[List[torch.Tensor]]]:
        """
        Load persistence diagrams from disk.

        Args:
            load_path: Path to load the diagrams from

        Returns:
            Loaded persistence diagrams on the pipeline's device
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        # Convert numpy back to torch tensors on the pipeline's device
        diagrams = []
        for diagram_set_np in data['diagrams']:
            diagram_set = []
            for hom_dim_np in diagram_set_np:
                sample_diagrams = []
                for d in hom_dim_np:
                    if isinstance(d, np.ndarray):
                        d_torch = torch.from_numpy(d).float().to(self.device)
                    else:
                        d_torch = d.to(self.device) if hasattr(d, 'to') else d
                    sample_diagrams.append(d_torch)
                diagram_set.append(sample_diagrams)
            diagrams.append(diagram_set)

        print(f"\n{'='*60}")
        print(f"Loaded diagrams from: {load_path}")
        print(f"  Datasets: {len(diagrams)}")
        print(f"  Device: {self.device}")
        print(f"  Created: {data['metadata'].get('created_at', 'unknown')}")
        print(f"{'='*60}\n")

        return diagrams

    def save_summaries(
        self,
        summaries: List[torch.Tensor],
        save_path: str,
        config: AnalysisConfig
    ) -> None:
        """
        Save vectorized summaries to disk with metadata.

        Args:
            summaries: List of summary tensors [fid, minus_0, plus_0, ...]
            save_path: Path to save the summaries
            config: AnalysisConfig containing analysis parameters
        """
        # Convert summaries to CPU and numpy for storage
        summaries_np = [s.cpu().numpy() for s in summaries]

        # Use generate_metadata for consistent metadata creation
        metadata = self.generate_metadata(config)
        metadata['data_type'] = 'summaries'

        # Package data
        data = {
            'summaries': summaries_np,
            'metadata': metadata
        }

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # Save with pickle
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n{'='*60}")
        print(f"Saved summaries to: {save_path}")
        print(f"  Datasets: {len(summaries)}")
        print(f"  Feature dimension: {summaries[0].shape[-1]}")
        print(f"{'='*60}\n")

    def load_summaries(self, load_path: str) -> List[torch.Tensor]:
        """
        Load vectorized summaries from disk.

        Args:
            load_path: Path to load the summaries from

        Returns:
            Loaded summaries as list of tensors on the pipeline's device
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        # Convert numpy back to torch tensors on the pipeline's device
        summaries = [torch.from_numpy(s).float().to(self.device) for s in data['summaries']]

        print(f"\n{'='*60}")
        print(f"Loaded summaries from: {load_path}")
        print(f"  Datasets: {len(summaries)}")
        print(f"  Feature dimension: {summaries[0].shape[-1]}")
        print(f"  Device: {self.device}")
        print(f"  Created: {data['metadata'].get('created_at', 'unknown')}")
        print(f"{'='*60}\n")

        return summaries