"""
Alpha complex filtration for point clouds as a PyTorch module.
"""
from typing import List, Optional, Union
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm


class AlphaComplexLayer(nn.Module):
    """
    PyTorch module for computing persistent homology of alpha complexes.

    This layer takes point cloud data and computes persistence diagrams
    using alpha complex filtration (Delaunay triangulation with circumradius).
    Handles both single samples and batches.

    Input/Output type preservation:
        - If input is torch.Tensor, output diagrams are torch.Tensor
        - If input is np.ndarray, output diagrams are np.ndarray
    """

    def __init__(
        self,
        homology_dimensions: List[int],
        min_persistence: Optional[List[float]] = None,
        show_progress: bool = False
    ):
        """
        Initialize alpha complex layer.

        Args:
            homology_dimensions: List of homology dimensions to compute (e.g., [0, 1])
            min_persistence: Minimum persistence threshold for each dimension
                           (default: 0 for all dimensions)
            show_progress: Whether to show progress bar during computation (default: False)
        """
        super().__init__()
        self.dimensions = homology_dimensions
        self.min_persistence = min_persistence if min_persistence is not None else [0.0] * len(self.dimensions)
        self.show_progress = show_progress

        assert len(self.min_persistence) == len(self.dimensions), \
            "min_persistence must have same length as homology_dimensions"

    def forward(self, X: Union[torch.Tensor, np.ndarray]) -> List[List[Union[torch.Tensor, np.ndarray]]]:
        """
        Compute persistence diagrams from alpha complex.

        Args:
            X: Input of shape (n_points, d) for single point cloud or
               (n_samples, n_points, d) for batch of point clouds.
               Can be torch.Tensor or np.ndarray.

        Returns:
            List of lists of persistence diagrams (same type as input).
            Outer list: homology dimensions (in order of self.dimensions)
            Inner list: diagrams for each sample
            Each diagram: shape (n_pairs, 2) with (birth, death) pairs
        """
        # Detect input type
        is_torch = isinstance(X, torch.Tensor)
        device = X.device if is_torch else None

        # Convert to torch if needed
        if not is_torch:
            X = torch.from_numpy(X).float()

        # Handle single sample vs batch
        if X.ndim == 2:  # Single point cloud
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        n_samples = X.shape[0]

        # Initialize output: [hom_dim][sample_idx] -> diagram
        all_diagrams = [[] for _ in self.dimensions]

        # Compute diagrams for each sample
        iterator = tqdm(range(n_samples), desc="Computing Alpha Complex") if self.show_progress else range(n_samples)
        for i in iterator:
            sample_diagrams = self._compute_single_diagram(X[i])

            # Organize by homology dimension
            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)

        # If single sample, unwrap batch dimension
        if single_sample:
            all_diagrams = [[dgms[0]] for dgms in all_diagrams]

        # Convert output to match input type
        if not is_torch:
            all_diagrams = [[dgm.numpy() for dgm in dgms] for dgms in all_diagrams]
        elif device is not None:
            all_diagrams = [[dgm.to(device) for dgm in dgms] for dgms in all_diagrams]

        return all_diagrams

    def _compute_single_diagram(self, X_sample: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute persistence diagram for a single point cloud.

        Args:
            X_sample: Single point cloud of shape (n_points, d)

        Returns:
            List of torch tensors (one per homology dimension)
        """
        # Convert to numpy for GUDHI
        X_numpy = X_sample.detach().cpu().numpy()

        # Create alpha complex and simplex tree
        alpha_complex = gudhi.AlphaComplex(points=X_numpy)
        simplex_tree = alpha_complex.create_simplex_tree()
        simplex_tree.compute_persistence()

        # Extract diagrams for each homology dimension
        diagrams = []
        for idx_dim, dimension in enumerate(self.dimensions):
            persistence_pairs = simplex_tree.persistence_intervals_in_dimension(dimension)

            # Filter out infinite death times
            finite_pairs = persistence_pairs[persistence_pairs[:, 1] < np.inf]

            # Apply minimum persistence threshold
            min_pers = self.min_persistence[idx_dim]
            if min_pers > 0 and len(finite_pairs) > 0:
                persistence = np.abs(finite_pairs[:, 1] - finite_pairs[:, 0])
                finite_pairs = finite_pairs[persistence > min_pers]

            # Convert to torch tensor (CPU)
            diagram = torch.from_numpy(finite_pairs).float()
            diagrams.append(diagram)

        return diagrams
