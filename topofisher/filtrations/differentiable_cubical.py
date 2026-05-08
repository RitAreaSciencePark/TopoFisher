"""
Differentiable cubical complex filtration using GUDHI cofaces approach.

This module implements a differentiable cubical persistence layer following
GUDHI's TensorFlow implementation approach. The key idea:

1. Compute persistence topology using GUDHI (non-differentiable)
2. Extract critical cell indices using cofaces_of_persistence_pairs()
3. Gather values from input tensor using indices (differentiable!)
4. Gradients flow back through gather operation to input

This enables end-to-end training of networks that transform fields before
persistence computation, optimizing topological features for downstream tasks.

Key improvement: Uses GUDHI's cofaces API to directly get cell indices,
eliminating the need for noise-based value mapping.

Performance: GUDHI's C++ persistence computation releases the Python GIL,
so ThreadPoolExecutor gives true parallelism across samples. On a 256-core
node: ~10× speedup at N=64 with 16 threads.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import gudhi
import os
from concurrent.futures import ThreadPoolExecutor


class DifferentiableCubicalLayer(nn.Module):
    """
    Differentiable cubical persistence layer using torch.gather.

    Unlike standard cubical persistence which breaks the computational graph,
    this layer preserves differentiability by:

    1. Using GUDHI to find which pixels are topologically critical
    2. Gathering their values from the input tensor via torch.gather
    3. Constructing diagrams from gathered values (preserves gradients)

    The topology computation is non-differentiable, but gathering pixel values
    from the original tensor allows gradients to flow back to the input.

    Supports both sublevel (default) and superlevel filtrations:
    - sublevel: features appear as values increase (captures "peaks")
    - superlevel: features appear as values decrease (captures "valleys")

    Threading: When n_jobs > 1, GUDHI topology computations run in parallel
    using threads (GUDHI releases the GIL). The differentiable gather step
    still runs on the main thread to preserve the autograd graph.

    Skip-K topology caching: When skip_k > 1, GUDHI topology is recomputed
    only every skip_k forward passes during training. On intermediate steps,
    cached coface index arrays are reused with the current tensor values,
    so gradients still flow correctly through torch.gather. This exploits
    the fact that topological pairings change slowly as CNN weights update.
    Cache is always bypassed during eval (torch.no_grad) for exact results.

    Example:
        >>> layer = DifferentiableCubicalLayer(homology_dimensions=[0, 1], n_jobs=16)
        >>> field = torch.randn(10, 16, 16, requires_grad=True)
        >>> diagrams = layer(field)  # Gradients preserved!
        >>> loss = some_loss_function(diagrams)
        >>> loss.backward()  # Gradients flow back to field
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        n_jobs: int = 1,
        skip_k: int = 1,
        backend: str = 'gudhi',
        construction: str = 'T',
        gpu_sub_batch_size: int = 100,
        periodic: bool = False,
    ):
        """
        Initialize differentiable cubical layer.

        Args:
            homology_dimensions: List of homology dimensions to compute
            min_persistence: Minimum persistence threshold for each dimension
            superlevel: If True, compute superlevel filtration (negate values).
                       Use for fields where "valleys" are features of interest.
            n_jobs: Number of parallel threads for GUDHI topology computation.
                   1 = serial (default), -1 = use all available CPU cores,
                   >1 = use that many threads. GUDHI releases the GIL so
                   threads give true parallelism for the C++ compute.
            skip_k: Recompute GUDHI topology every skip_k forward passes
                   during training. 1 = always recompute (default/exact).
                   Values 5-10 give significant speedup with minimal accuracy
                   loss. Cache is always bypassed during eval mode.
            backend: Persistence computation backend.
                   'gudhi' (default): CPU-based GUDHI with optional threading.
                   'cmp_gpu': GPU-accelerated CMP CUDA kernels (requires cmp package).
                   'gudhi_gpu': GPU-accelerated GUDHI CUDA extension (requires GUDHI fork).
                   The GPU backends ignore n_jobs and skip_k (GPU is fast enough).
            construction: Cubical complex construction type ('T' or 'V').
                   'T' (default): T-construction — input pixels are top-dimensional cells.
                   'V': V-construction — input pixels are vertices, complex is built
                        by embedding vertices onto all cells via max of incident vertices.
                   Only used with gudhi_gpu backend. Other backends ignore this.
            periodic: If True, use periodic boundary conditions (torus topology).
                   For periodic inputs like GRF or lensing maps generated via FFT,
                   this correctly handles wrap-around topology. Uses
                   gudhi.PeriodicCubicalComplex on CPU or periodic GPU functions.
        """
        super().__init__()
        self.backend = backend
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(self.dimensions)
        self.superlevel = superlevel
        self.periodic = periodic

        # GPU backend: delegate to GPUCubicalLayer
        if backend == 'cmp_gpu':
            from topofisher.filtrations.gpu_cubical import GPUCubicalLayer
            self._gpu_layer = GPUCubicalLayer(
                homology_dimensions=homology_dimensions,
                min_persistence=self.min_persistence,
                superlevel=superlevel,
            )
        elif backend == 'gudhi_gpu':
            from topofisher.filtrations.gudhi_gpu_cubical import GUDHIGPUCubicalLayer
            self._gpu_layer = GUDHIGPUCubicalLayer(
                homology_dimensions=homology_dimensions,
                min_persistence=self.min_persistence,
                superlevel=superlevel,
                construction=construction,
                sub_batch_size=gpu_sub_batch_size,
                periodic=periodic,
            )
        else:
            self._gpu_layer = None

        if n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        else:
            self.n_jobs = max(1, n_jobs)

        # Skip-K topology caching
        self.skip_k = max(1, skip_k)
        self._call_count = 0
        self._cached_cof_pp = None      # List of coface pairs (one per sample)
        self._cached_n_samples = 0      # Batch size when cache was populated
        self._topology_recomputes = 0   # Counter for profiling
        self._topology_cache_hits = 0   # Counter for profiling

        assert len(self.min_persistence) == len(self.dimensions), \
            "min_persistence must have same length as homology_dimensions"

    def _should_recompute_topology(self, n_samples: int) -> bool:
        """
        Determine if GUDHI topology should be recomputed or cached indices reused.

        Topology is always recomputed when:
        - skip_k <= 1 (no caching requested)
        - Model is in eval mode (exact results needed for validation/test)
        - No cached indices exist yet (first call)
        - Batch size changed (cached indices don't match)
        - skip_k steps have elapsed since last recomputation

        Returns:
            True if GUDHI should be run, False if cached indices can be reused
        """
        if self.skip_k <= 1:
            return True
        if not self.training:
            return True  # Always exact during eval
        if self._cached_cof_pp is None:
            return True
        if self._cached_n_samples != n_samples:
            return True  # Batch size changed
        # Recompute every skip_k calls (first call in each window)
        return (self._call_count % self.skip_k == 0)

    def invalidate_cache(self):
        """Clear cached topology indices. Call when model changes significantly."""
        self._cached_cof_pp = None
        self._cached_n_samples = 0

    def get_cache_stats(self) -> dict:
        """Return topology cache statistics for profiling."""
        total = self._topology_recomputes + self._topology_cache_hits
        return {
            'recomputes': self._topology_recomputes,
            'cache_hits': self._topology_cache_hits,
            'total_calls': total,
            'hit_rate': self._topology_cache_hits / total if total > 0 else 0.0,
            'skip_k': self.skip_k,
        }

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute differentiable persistence diagrams.

        Args:
            X: Input tensor of shape (H, W) for single sample or (n_samples, H, W) for batch

        Returns:
            List of lists of persistence diagrams.
            Outer list: homology dimensions
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2) with (birth, death) values
        """
        # GPU backend: delegate entirely to GPUCubicalLayer
        if self._gpu_layer is not None:
            return self._gpu_layer(X)

        # Handle single sample vs batch
        if X.ndim == 2:
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # For superlevel filtration, negate values
        if self.superlevel:
            X = -X

        n_samples = X.shape[0]
        device = X.device

        # Determine if we need to recompute topology
        self._call_count += 1
        recompute = self._should_recompute_topology(n_samples)

        if recompute:
            self._topology_recomputes += 1
        else:
            self._topology_cache_hits += 1

        # Initialize output: [hom_dim][sample_idx] -> diagram
        all_diagrams = [[] for _ in self.dimensions]

        if self.n_jobs > 1 and n_samples > 1:
            # PARALLEL: run GUDHI topology in threads, gather on main thread
            self._forward_parallel(X, n_samples, device, all_diagrams, recompute)
        else:
            # SERIAL: original path
            self._forward_serial(X, n_samples, device, all_diagrams, recompute)

        # If single sample, unwrap batch dimension
        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams

    def _forward_serial(
        self,
        X: torch.Tensor,
        n_samples: int,
        device: torch.device,
        all_diagrams: List[List[torch.Tensor]],
        recompute: bool = True
    ) -> None:
        """Serial path: compute topology + gather for each sample sequentially."""
        if recompute:
            # Full computation: GUDHI topology + differentiable gather
            cached = []
            for i in range(n_samples):
                sample_diagrams, cof_pp = self._compute_differentiable_diagram_cached(X[i], device)
                cached.append(cof_pp)
                for dim_idx, dgm in enumerate(sample_diagrams):
                    all_diagrams[dim_idx].append(dgm)
            # Cache the coface indices
            self._cached_cof_pp = cached
            self._cached_n_samples = n_samples
        else:
            # Skip: reuse cached topology, just gather from current tensor
            for i in range(n_samples):
                sample_diagrams = self._gather_diagrams(X[i], self._cached_cof_pp[i], device)
                for dim_idx, dgm in enumerate(sample_diagrams):
                    all_diagrams[dim_idx].append(dgm)

    def _forward_parallel(
        self,
        X: torch.Tensor,
        n_samples: int,
        device: torch.device,
        all_diagrams: List[List[torch.Tensor]],
        recompute: bool = True
    ) -> None:
        """
        Parallel path: topology in threads, differentiable gather on main thread.

        GUDHI's C++ persistence computation releases the Python GIL, so
        ThreadPoolExecutor gives true parallelism for the expensive part.
        The cheap differentiable gather (torch indexing) runs on the main
        thread to keep the autograd graph intact.

        When recompute=False (skip-K cache hit), the expensive Phase 1 is
        skipped entirely — only the cheap Phase 2 gather runs.

        Two-phase approach:
          Phase 1 (parallel): GUDHI CubicalComplex + cofaces_of_persistence_pairs
                              → returns coface index arrays (numpy, no torch)
          Phase 2 (serial):   torch.gather from original tensor using indices
                              → builds differentiable diagrams with gradients
        """
        if recompute:
            # Phase 1: parallel topology computation
            X_numpy_all = X.detach().cpu().numpy()

            def _topology_worker(i):
                """Thread worker: compute GUDHI topology, return coface indices only."""
                sample = X_numpy_all[i]
                H, W = sample.shape
                cells = sample.flatten()
                # Guard against NaN/Inf values that crash GUDHI's C++ code
                if not np.all(np.isfinite(cells)):
                    cells = np.nan_to_num(cells, nan=0.0, posinf=1e6, neginf=-1e6)
                if self.periodic:
                    cc = gudhi.PeriodicCubicalComplex(
                        top_dimensional_cells=cells.reshape(H, W),
                        periodic_dimensions=[True, True],
                    )
                else:
                    cc = gudhi.CubicalComplex(
                        dimensions=[H, W],
                        top_dimensional_cells=cells
                    )
                cc.compute_persistence()
                return cc.cofaces_of_persistence_pairs()

            n_workers = min(self.n_jobs, n_samples)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                all_cof_pp = list(executor.map(_topology_worker, range(n_samples)))

            # Cache coface indices for skip-K reuse
            self._cached_cof_pp = all_cof_pp
            self._cached_n_samples = n_samples
        else:
            # Skip Phase 1: reuse cached topology indices
            all_cof_pp = self._cached_cof_pp

        # Phase 2: differentiable gather on main thread (preserves autograd)
        for i in range(n_samples):
            sample_diagrams = self._gather_diagrams(X[i], all_cof_pp[i], device)
            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)

    def _gather_diagrams(
        self,
        X_sample: torch.Tensor,
        cof_pp: tuple,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Build differentiable diagrams by gathering values from the input tensor.

        This is the differentiable part: indexing into the original PyTorch tensor
        preserves the autograd graph, allowing gradients to flow back to the input.

        Args:
            X_sample: Single grid sample of shape (H, W) - the ORIGINAL tensor with grad
            cof_pp: Cofaces of persistence pairs from GUDHI (tuple of arrays)
            device: Device for output tensors

        Returns:
            List of differentiable diagrams (one per homology dimension)
        """
        Xflat = X_sample.flatten()

        diagrams = []
        for idx_dim, dim in enumerate(self.dimensions):
            if len(cof_pp[0]) > dim and len(cof_pp[0][dim]) > 0:
                cof = cof_pp[0][dim]
                cof = torch.tensor(cof, device=device, dtype=torch.long)

                # Differentiable gather: Xflat[indices] preserves gradients
                gathered = Xflat[cof.flatten()]
                finite_dgm = gathered.reshape(-1, 2)

                # Apply minimum persistence threshold
                min_pers = self.min_persistence[idx_dim]
                if min_pers > 0:
                    persistence = torch.abs(finite_dgm[:, 1] - finite_dgm[:, 0])
                    finite_dgm = finite_dgm[persistence > min_pers]

                diagrams.append(finite_dgm)
            else:
                diagrams.append(torch.empty((0, 2), device=device))

        return diagrams

    def _compute_differentiable_diagram_cached(
        self,
        X_sample: torch.Tensor,
        device: torch.device
    ) -> Tuple[List[torch.Tensor], tuple]:
        """
        Compute differentiable diagram AND return coface indices for caching.

        Same as _compute_differentiable_diagram() but returns the GUDHI coface
        pairs alongside the diagrams. Used by skip-K caching: on recompute
        steps, we store these indices; on skip steps, we pass them directly
        to _gather_diagrams() without calling GUDHI.

        Args:
            X_sample: Single grid sample of shape (H, W)
            device: Device for output tensors

        Returns:
            Tuple of (diagrams, cof_pp) where:
              - diagrams: List of differentiable diagram tensors per dimension
              - cof_pp: GUDHI cofaces_of_persistence_pairs() result for caching
        """
        H, W = X_sample.shape
        Xflat = X_sample.flatten()
        Xflat_numpy = X_sample.detach().cpu().numpy().flatten()

        if not np.all(np.isfinite(Xflat_numpy)):
            Xflat_numpy = np.nan_to_num(Xflat_numpy, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.periodic:
            cc = gudhi.PeriodicCubicalComplex(
                top_dimensional_cells=Xflat_numpy.reshape(H, W),
                periodic_dimensions=[True, True],
            )
        else:
            cc = gudhi.CubicalComplex(
                dimensions=[H, W],
                top_dimensional_cells=Xflat_numpy
            )
        cc.compute_persistence()
        cof_pp = cc.cofaces_of_persistence_pairs()

        # Build differentiable diagrams using _gather_diagrams
        diagrams = self._gather_diagrams(X_sample, cof_pp, device)

        return diagrams, cof_pp

    def _compute_differentiable_diagram(
        self,
        X_sample: torch.Tensor,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Compute differentiable persistence diagram for a single sample (serial path).

        Direct translation of GUDHI TensorFlow CubicalLayer implementation:
        https://gudhi.inria.fr/python/latest/_modules/gudhi/tensorflow/cubical_layer.html

        Args:
            X_sample: Single grid sample of shape (H, W)
            device: Device for output tensors

        Returns:
            List of differentiable diagrams (one per homology dimension)
        """
        diagrams, _ = self._compute_differentiable_diagram_cached(X_sample, device)
        return diagrams
