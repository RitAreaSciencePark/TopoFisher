"""Pipeline for training learnable filtration."""

from typing import List
import torch
import torch.nn as nn

from .base import LearnablePipeline
from ..base import LazyStackedTensor, LAZY_TENSOR_TYPES

class LearnableFiltrationPipeline(LearnablePipeline):
    """
    Pipeline for training filtration from raw data.

    This pipeline takes raw data (e.g., images, point clouds) and trains a
    learnable filtration function to maximize Fisher information. This is
    the most end-to-end learnable approach. PyTorch automatically tracks
    and trains all learnable parameters in the pipeline.
    """

    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from raw data.

        Full pipeline: data → filtration → vectorization → compression

        Args:
            data: Raw data [fid, minus_0, plus_0, ...]

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        diagrams = self.compute_diagrams(data, verbose=False)  # Uses self.filtration
        summaries = self.vectorize(diagrams)
        compressed = self.compression(summaries)
        return compressed

    def _compute_summaries_batched(
        self,
        data: List[torch.Tensor],
        eval_batch_size: int = 500
    ) -> List[torch.Tensor]:
        """
        Compute summaries in batches to avoid GPU OOM during evaluation.

        Uses a 4-phase approach to batch the expensive per-sample GPU work
        (CNN → cubical persistence) while preserving global consistency of
        vectorization hyperparameters and compression weights:

        Phase 1: Batch filtration (CNN + persistence) in chunks → all diagrams
        Phase 2: Fit vectorization on ALL diagrams (TopK auto-selects k globally)
        Phase 3: Batch vectorization transform in chunks → all summaries
        Phase 4: Run MOPED compression on ALL summaries at once

        This ensures TopK k is determined from the global minimum across ALL
        diagrams, and MOPED sees all summaries for a consistent covariance
        matrix — unlike the previous buggy approach that ran fit()/MOPED
        independently per chunk.

        Args:
            data: Raw data [fid, minus_0, plus_0, ...]
            eval_batch_size: Maximum samples per batch during eval

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        n_samples = data[0].shape[0] if isinstance(data[0], (torch.Tensor,) + LAZY_TENSOR_TYPES) else len(data[0])
        if n_samples <= eval_batch_size:
            return self._compute_summaries(data)

        n_chunks = (n_samples + eval_batch_size - 1) // eval_batch_size

        # Decide path: only accumulate all diagrams when fitting is required.
        vec_needs_fitting = (
            hasattr(self.vectorization, 'fit')
            and not getattr(self, '_vectorization_fitted', False)
        )

        if vec_needs_fitting:
            # === Phase 1: Batch filtration — accumulate all diagrams for fit ===
            all_diagrams = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_data = [d[start:end] for d in data]

                chunk_diagrams = self.compute_diagrams(chunk_data, verbose=False)

                if all_diagrams is None:
                    all_diagrams = [
                        [[] for _ in range(len(chunk_diagrams[0]))]
                        for _ in range(len(chunk_diagrams))
                    ]
                for set_idx in range(len(chunk_diagrams)):
                    for hom_idx in range(len(chunk_diagrams[set_idx])):
                        all_diagrams[set_idx][hom_idx].extend(
                            chunk_diagrams[set_idx][hom_idx]
                        )

            # === Phase 2: Fit vectorization on ALL diagrams ===
            self.vectorization.fit(all_diagrams)
            self._vectorization_fitted = True

            # === Phase 3: Batch vectorization transform ===
            all_summaries = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_diagrams = [
                    [hom_dgms[start:end] for hom_dgms in set_dgms]
                    for set_dgms in all_diagrams
                ]
                chunk_summaries = [self.vectorization(diags) for diags in chunk_diagrams]

                if all_summaries is None:
                    all_summaries = chunk_summaries
                else:
                    for i in range(len(chunk_summaries)):
                        all_summaries[i] = torch.cat(
                            [all_summaries[i], chunk_summaries[i]], dim=0
                        )
            del all_diagrams
        else:
            # === Streaming path: fuse filtration + vectorization per chunk ===
            # Peak RAM: chunk_size × n_features × n_sets (not diagrams for all samples).
            all_summaries = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_data = [d[start:end] for d in data]
                chunk_diagrams = self.compute_diagrams(chunk_data, verbose=False)
                chunk_summaries = [self.vectorization(diags) for diags in chunk_diagrams]
                del chunk_diagrams

                if all_summaries is None:
                    all_summaries = chunk_summaries
                else:
                    for i in range(len(chunk_summaries)):
                        all_summaries[i] = torch.cat(
                            [all_summaries[i], chunk_summaries[i]], dim=0
                        )

        # === Phase 4: MOPED compression on ALL summaries ===
        # MOPED needs all samples for consistent covariance matrix
        compressed = self.compression(all_summaries)
        return compressed