"""
Cubical complex filtration as a PyTorch module.

Supports three persistence backends:
  - "gudhi" (default): GUDHI CubicalComplex with T-construction
  - "cripser": CubicalRipser (~3x faster), pip install cripser
  - "gudhi_gpu": GUDHI CUDA extension for GPU-accelerated persistence
"""
from typing import List, Optional, Literal
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
from multiprocessing import cpu_count

# Try to import cripser for faster persistence
try:
    import cripser as _cripser
    CRIPSER_AVAILABLE = True
except ImportError:
    _cripser = None
    CRIPSER_AVAILABLE = False


def _compute_persistence_worker(args):
    """Worker function for multiprocessing persistence computation.
    
    Runs in a separate process to enable parallel persistence across samples/bins.
    """
    X_numpy_flat, X_shape, dimensions, min_persistence, superlevel, periodic = args
    
    if superlevel:
        X_numpy_flat = -X_numpy_flat
    
    if periodic:
        H, W = X_shape
        cubical_complex = gudhi.PeriodicCubicalComplex(
            top_dimensional_cells=X_numpy_flat.reshape(H, W),
            periodic_dimensions=[True, True],
        )
    else:
        cubical_complex = gudhi.CubicalComplex(
            dimensions=X_shape,
            top_dimensional_cells=X_numpy_flat
        )
    cubical_complex.compute_persistence()
    
    diagrams = []
    for idx_dim, dimension in enumerate(dimensions):
        persistence_pairs = cubical_complex.persistence_intervals_in_dimension(dimension)
        finite_pairs = persistence_pairs[persistence_pairs[:, 1] < np.inf]
        
        min_pers = min_persistence[idx_dim]
        if min_pers > 0 and len(finite_pairs) > 0:
            persistence = np.abs(finite_pairs[:, 1] - finite_pairs[:, 0])
            finite_pairs = finite_pairs[persistence > min_pers]
        
        diagrams.append(finite_pairs.astype(np.float32))
    
    return diagrams


class CubicalLayer(nn.Module):
    """
    PyTorch module for computing persistent homology of cubical complexes.

    This layer takes grid data (e.g., GRF) and computes persistence diagrams
    using cubical complex filtration. Handles both single samples and batches.
    
    Supports both sublevel (default) and superlevel filtrations:
    - sublevel: features appear as values increase (captures "peaks")
    - superlevel: features appear as values decrease (captures "valleys")
    
    For GRFs with negative spectral index B, superlevel may be more appropriate
    since the field structure is inverted compared to positive B.
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: Optional[List[float]] = None,
        show_progress: bool = False,
        superlevel: bool = False,
        n_jobs: int = 1,
        backend: Literal["gudhi", "cripser", "gudhi_gpu"] = "gudhi",
        construction: str = 'T',
        periodic: bool = False,
    ):
        """
        Initialize cubical complex layer.

        Args:
            homology_dimensions: List of homology dimensions to compute (e.g., [0, 1])
            min_persistence: Minimum persistence threshold for each dimension
                           (default: 0 for all dimensions)
            show_progress: Whether to show progress bar during computation (default: False)
            superlevel: If True, compute superlevel filtration (negate values).
                       Use for fields where "valleys" are the features of interest,
                       e.g., GRFs with negative spectral index B. (default: False)
            n_jobs: Number of parallel workers for persistence computation.
                   1 = serial (default), -1 = use all available CPU cores,
                   >1 = use that many workers. Useful for large N or multi-bin inputs.
                   Ignored when backend='gudhi_gpu' (GPU handles parallelism).
            backend: Persistence computation backend.
                    "gudhi" (default): GUDHI CubicalComplex (T-construction).
                    "cripser": CubicalRipser, ~3x faster (V-construction).
                    "gudhi_gpu": GUDHI CUDA extension for GPU-accelerated persistence.
                    Falls back to "gudhi" if cripser/gudhi_gpu is not available.
            construction: Cubical complex construction type ('T' or 'V').
                    Only used with gudhi_gpu backend. Other backends ignore this.
                    'T' (default): T-construction — input pixels are top-dimensional cells.
                    'V': V-construction — input pixels are vertices.
            periodic: If True, use periodic boundary conditions (torus topology).
                    For periodic inputs like GRF or lensing maps generated via FFT.
                    Uses gudhi.PeriodicCubicalComplex on CPU or periodic GPU functions.
        """
        super().__init__()
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(self.dimensions)
        self.show_progress = show_progress
        self.superlevel = superlevel
        self.periodic = periodic
        
        if n_jobs == -1:
            # Respect Slurm CPU allocation if available, otherwise use all cores
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus is not None:
                self.n_jobs = int(slurm_cpus)
            else:
                self.n_jobs = os.cpu_count() or 1
        else:
            self.n_jobs = max(1, n_jobs)
        
        # Backend selection
        if backend == "cripser" and not CRIPSER_AVAILABLE:
            import warnings
            warnings.warn("cripser not installed, falling back to gudhi. Install with: pip install cripser")
            self.backend = "gudhi"
        elif backend == "gudhi_gpu":
            self.backend = "gudhi_gpu"
        else:
            self.backend = backend
        
        self.construction = construction
        
        # GPU backend: create GUDHIGPUCubicalLayer for delegation
        if self.backend == "gudhi_gpu":
            from topofisher.filtrations.gudhi_gpu_cubical import GUDHIGPUCubicalLayer
            self._gpu_layer = GUDHIGPUCubicalLayer(
                homology_dimensions=homology_dimensions,
                min_persistence=self.min_persistence,
                superlevel=superlevel,
                construction=construction,
                periodic=periodic,
            )
        else:
            self._gpu_layer = None
        
        self._max_dim = max(self.dimensions) if self.dimensions else 0

        assert len(self.min_persistence) == len(self.dimensions), \
            "min_persistence must have same length as homology_dimensions"

    def __repr__(self):
        """String representation showing configuration."""
        parts = [f"homology_dimensions={self.dimensions}"]
        if not all(p == 0.0 for p in self.min_persistence):
            parts.append(f"min_persistence={self.min_persistence}")
        if self.superlevel:
            parts.append("superlevel=True")
        if self.n_jobs > 1:
            parts.append(f"n_jobs={self.n_jobs}")
        if self.backend != "gudhi":
            parts.append(f"backend='{self.backend}'")
        return f"CubicalLayer({', '.join(parts)})"

    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        config = {
            'homology_dimensions': self.dimensions,
            'min_persistence': self.min_persistence,
            'superlevel': self.superlevel,
            'n_jobs': self.n_jobs,
            'backend': self.backend,
            'periodic': self.periodic,
        }
        if self.backend == 'gudhi_gpu':
            config['construction'] = self.construction
        return config

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams from cubical complex.

        Args:
            X: Input tensor of shape:
               - (H, W) for single 2D sample
               - (n_samples, H, W) for batch of 2D samples
               - (n_samples, n_bins, H, W) for batch of multi-bin 2D samples
               
               Multi-bin input: persistence is computed independently per bin.
               Output structure interleaves bins and homology dimensions:
               [bin0_H0, bin0_H1, bin1_H0, bin1_H1, ...][sample_idx]

        Returns:
            List of lists of persistence diagrams.
            Outer list: For single-bin: homology dimensions. 
                       For multi-bin: n_bins * n_hom_dims entries
                       (bin0_hom0, bin0_hom1, bin1_hom0, bin1_hom1, ...)
            Inner list: diagrams for each sample
            Each diagram: tensor of shape (n_pairs, 2) with (birth, death) pairs
        """
        # Detect multi-bin input: (n_samples, n_bins, H, W)
        multi_bin = False
        single_sample = False
        
        if X.ndim == 2:  # Single 2D sample (H, W)
            X = X.unsqueeze(0)
            single_sample = True
            n_bins = 1
        elif X.ndim == 3:  # Batch of 2D: (n_samples, H, W)
            n_bins = 1
        elif X.ndim == 4:  # Multi-bin: (n_samples, n_bins, H, W)
            multi_bin = True
            n_bins = X.shape[1]
        else:
            raise ValueError(f"Unexpected input shape {X.shape}. Expected 2D, 3D, or 4D tensor.")

        device = X.device
        n_samples = X.shape[0]
        n_hom = len(self.dimensions)
        
        if multi_bin:
            # Reshape (n_samples, n_bins, H, W) -> (n_samples * n_bins, H, W)
            H, W = X.shape[2], X.shape[3]
            X_flat = X.reshape(n_samples * n_bins, H, W)
        else:
            X_flat = X
        
        n_total = X_flat.shape[0]  # n_samples (single-bin) or n_samples * n_bins
        
        # GPU backend: delegate to GUDHIGPUCubicalLayer
        if self._gpu_layer is not None:
            # GPU layer handles (n_total, H, W) directly
            gpu_diagrams = self._gpu_layer(X_flat)
            # gpu_diagrams: [hom_dim][flat_sample] = (n_pairs, 2)
            
            if multi_bin:
                # Reorganize: [flat_idx] -> [bin*n_hom + hom][sample]
                n_output_dims = n_bins * n_hom
                all_diagrams = [[] for _ in range(n_output_dims)]
                
                for sample_idx in range(n_samples):
                    for bin_idx in range(n_bins):
                        flat_idx = sample_idx * n_bins + bin_idx
                        for hom_idx in range(n_hom):
                            output_idx = bin_idx * n_hom + hom_idx
                            all_diagrams[output_idx].append(gpu_diagrams[hom_idx][flat_idx])
            else:
                all_diagrams = gpu_diagrams
                if single_sample:
                    all_diagrams = [[dgms[0]] for dgms in all_diagrams]
            
            return all_diagrams
        
        # CPU backends: compute persistence (serial or parallel)
        if self.n_jobs > 1 and n_total > 1:
            all_sample_diagrams = self._compute_parallel(X_flat)
        else:
            all_sample_diagrams = self._compute_serial(X_flat, device)
        
        if multi_bin:
            # Reorganize: [flat_idx] -> [bin][sample] then interleave hom dims
            # all_sample_diagrams[flat_idx] = list of n_hom diagrams
            # Output: [bin*n_hom + hom][sample]
            n_output_dims = n_bins * n_hom
            all_diagrams = [[] for _ in range(n_output_dims)]
            
            for sample_idx in range(n_samples):
                for bin_idx in range(n_bins):
                    flat_idx = sample_idx * n_bins + bin_idx
                    sample_dgms = all_sample_diagrams[flat_idx]
                    for hom_idx in range(n_hom):
                        output_idx = bin_idx * n_hom + hom_idx
                        all_diagrams[output_idx].append(sample_dgms[hom_idx])
        else:
            # Standard single-bin: [hom_dim][sample]
            all_diagrams = [[] for _ in self.dimensions]
            for i in range(n_total):
                sample_dgms = all_sample_diagrams[i]
                for dim_idx in range(n_hom):
                    all_diagrams[dim_idx].append(sample_dgms[dim_idx])
            
            if single_sample:
                all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        return all_diagrams

    def _compute_serial(self, X: torch.Tensor, device: torch.device) -> List[List[torch.Tensor]]:
        """Compute persistence diagrams serially."""
        n_total = X.shape[0]
        results = []
        
        # For superlevel filtration, negate values
        if self.superlevel:
            X = -X
        
        iterator = tqdm(range(n_total), desc="Computing Cubical Complex") if self.show_progress else range(n_total)
        
        if self.backend == "cripser":
            X_numpy = X.detach().cpu().numpy()
            for i in iterator:
                dgms = self._compute_single_cripser(X_numpy[i], device)
                results.append(dgms)
        else:
            for i in iterator:
                dgms = self._compute_single_diagram(X[i], device)
                results.append(dgms)
        
        return results
    
    def _compute_parallel(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """Compute persistence diagrams in parallel using ThreadPoolExecutor.
        
        Uses threads instead of processes to avoid:
        - fork + JAX deadlock (os.fork() copies JAX's thread pool)
        - spawn serialization overhead (pickling large numpy arrays)
        
        Both GUDHI's and CubicalRipser's C++ persistence computations release
        the GIL, so threads give true parallelism for the compute-heavy part.
        """
        n_total = X.shape[0]
        device = X.device
        
        # Pre-convert all samples to numpy (shared memory, no pickling needed)
        X_numpy_all = X.detach().cpu().numpy()
        
        if self.backend == "cripser":
            def _worker(i):
                """Thread worker: compute persistence for sample i using CubicalRipser."""
                sample = X_numpy_all[i].astype(np.float64)
                if self.superlevel:
                    sample = -sample
                
                ph = _cripser.computePH(sample, maxdim=self._max_dim)
                # cripser uses DBL_MAX (not np.inf) for essential features
                _DEATH_THRESH = np.finfo(np.float64).max * 0.5
                
                diagrams = []
                for idx_dim, dimension in enumerate(self.dimensions):
                    mask = (ph[:, 0] == dimension) & (ph[:, 2] < _DEATH_THRESH)
                    pairs = ph[mask][:, 1:3].astype(np.float32)  # birth, death
                    
                    min_p = self.min_persistence[idx_dim]
                    if min_p > 0 and len(pairs) > 0:
                        pers = np.abs(pairs[:, 1] - pairs[:, 0])
                        pairs = pairs[pers > min_p]
                    
                    diagrams.append(torch.from_numpy(pairs).to(device))
                
                return diagrams
        else:
            def _worker(i):
                """Thread worker: compute persistence for sample i using GUDHI."""
                sample = X_numpy_all[i]
                X_shape = sample.shape
                X_flat = sample.flatten()
                
                if self.superlevel:
                    X_flat = -X_flat
                
                if self.periodic:
                    H, W = X_shape
                    cc = gudhi.PeriodicCubicalComplex(
                        top_dimensional_cells=X_flat.reshape(H, W),
                        periodic_dimensions=[True, True],
                    )
                else:
                    cc = gudhi.CubicalComplex(
                        dimensions=X_shape,
                        top_dimensional_cells=X_flat
                    )
                cc.compute_persistence()
                
                diagrams = []
                for idx_dim, dimension in enumerate(self.dimensions):
                    pairs = cc.persistence_intervals_in_dimension(dimension)
                    finite = pairs[pairs[:, 1] < np.inf]
                    
                    min_p = self.min_persistence[idx_dim]
                    if min_p > 0 and len(finite) > 0:
                        pers = np.abs(finite[:, 1] - finite[:, 0])
                        finite = finite[pers > min_p]
                    
                    diagrams.append(torch.from_numpy(finite.astype(np.float32)).to(device))
                
                return diagrams
        
        n_workers = min(self.n_jobs, n_total)
        backend_name = self.backend.upper()
        desc = f"Computing Cubical Complex [{backend_name}] ({n_workers} threads)"
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            if self.show_progress:
                results = list(tqdm(
                    executor.map(_worker, range(n_total)),
                    total=n_total, desc=desc
                ))
            else:
                results = list(executor.map(_worker, range(n_total)))
        
        return results

    def _compute_single_cripser(
        self,
        X_numpy: np.ndarray,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Compute persistence diagram for a single sample using CubicalRipser.
        
        CubicalRipser uses V-construction (pixels = vertices) by default.
        ~3x faster than GUDHI for 2D grids.

        Args:
            X_numpy: Single grid sample as numpy array (H, W)
            device: Device to place output tensors

        Returns:
            List of diagrams (one per homology dimension)
        """
        ph = _cripser.computePH(X_numpy.astype(np.float64), maxdim=self._max_dim)
        # cripser uses DBL_MAX (not np.inf) for essential features
        _DEATH_THRESH = np.finfo(np.float64).max * 0.5
        
        diagrams = []
        for idx_dim, dimension in enumerate(self.dimensions):
            # Filter by dimension and finite death times (DBL_MAX = essential)
            mask = (ph[:, 0] == dimension) & (ph[:, 2] < _DEATH_THRESH)
            pairs = ph[mask][:, 1:3].astype(np.float32)  # birth, death
            
            # Apply minimum persistence threshold
            min_pers = self.min_persistence[idx_dim]
            if min_pers > 0 and len(pairs) > 0:
                persistence = np.abs(pairs[:, 1] - pairs[:, 0])
                pairs = pairs[persistence > min_pers]
            
            diagrams.append(torch.from_numpy(pairs).to(device))
        
        return diagrams

    def _compute_single_diagram(
        self,
        X_sample: torch.Tensor,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Compute persistence diagram for a single sample.

        Args:
            X_sample: Single grid sample
            device: Device to place output tensors

        Returns:
            List of diagrams (one per homology dimension)
        """
        # Convert to numpy for GUDHI
        X_shape = X_sample.shape
        X_numpy = X_sample.detach().cpu().numpy().flatten()

        # Create cubical complex
        if self.periodic:
            H, W = X_shape
            cubical_complex = gudhi.PeriodicCubicalComplex(
                top_dimensional_cells=X_numpy.reshape(H, W),
                periodic_dimensions=[True, True],
            )
        else:
            cubical_complex = gudhi.CubicalComplex(
                dimensions=X_shape,
                top_dimensional_cells=X_numpy
            )

        # Compute persistence
        cubical_complex.compute_persistence()

        # Extract diagrams for each homology dimension
        diagrams = []
        for idx_dim, dimension in enumerate(self.dimensions):
            # Get persistence intervals
            persistence_pairs = cubical_complex.persistence_intervals_in_dimension(dimension)

            # Filter out infinite death times
            finite_pairs = persistence_pairs[persistence_pairs[:, 1] < np.inf]

            # Apply minimum persistence threshold
            min_pers = self.min_persistence[idx_dim]
            if min_pers > 0 and len(finite_pairs) > 0:
                persistence = np.abs(finite_pairs[:, 1] - finite_pairs[:, 0])
                finite_pairs = finite_pairs[persistence > min_pers]

            # Convert to torch tensor
            diagram = torch.from_numpy(finite_pairs).float().to(device)
            diagrams.append(diagram)

        return diagrams
