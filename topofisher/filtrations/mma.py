"""
MMA (Multiparameter Module Approximation) filtration as a PyTorch module.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import multipers as mp
from multipers.filtrations import Cubical
from tqdm import tqdm


class MMALayer(nn.Module):
    """
    PyTorch module for computing MMA (Multiparameter Module Approximation) from scalar fields.

    This layer takes a scalar field (e.g., GRF), computes its gradient magnitude,
    constructs a bifiltration from (field, gradient), and returns MMA corner representations.
    """

    def __init__(self, homology_dimensions: List[int] = [0, 1], show_progress: bool = False):
        """
        Initialize MMA layer.

        Args:
            homology_dimensions: List of homology dimensions to compute (e.g., [0, 1])
            show_progress: Whether to show progress bar during computation (default: False)
        """
        super().__init__()
        self.dimensions = homology_dimensions
        self.show_progress = show_progress

    def compute_gradient(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude of a 2D field using finite differences.

        Args:
            field: Tensor of shape (H, W) for single sample or (n_samples, H, W) for batch

        Returns:
            Gradient magnitude tensor of same shape as field
        """
        if field.ndim == 2:
            # Single field
            field = field.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute gradients using central differences
        # pad to handle boundaries
        padded = torch.nn.functional.pad(field, (1, 1, 1, 1), mode='replicate')

        # Central differences
        grad_x = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / 2.0
        grad_y = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / 2.0

        # Gradient magnitude
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)

        if squeeze_output:
            return grad_mag.squeeze(0)
        return grad_mag

    def forward(self, X: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute MMA corner representations from scalar fields.

        Args:
            X: Input tensor of shape (H, W) for single 2D sample or
               (n_samples, H, W) for batch of 2D samples

        Returns:
            List of lists of MMA corner data.
            Outer list: homology dimensions (in order of self.dimensions)
            Inner list: corner data for each sample
            Each corner data: list of (births, deaths) tuples where births and deaths
                             are tensors of shape (n_corners, 2) representing 2D corners
        """
        # Handle single sample vs batch
        if X.ndim == 2:  # Single 2D sample
            X = X.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        n_samples = X.shape[0]
        device = X.device

        # Initialize output: [hom_dim][sample_idx] -> corner_data
        all_corners = [[] for _ in self.dimensions]

        # Compute MMA for each sample
        iterator = tqdm(range(n_samples), desc="Computing MMA") if self.show_progress else range(n_samples)
        for i in iterator:
            sample_corners = self._compute_single_mma(X[i], device)

            # Organize by homology dimension
            for dim_idx, corners in enumerate(sample_corners):
                all_corners[dim_idx].append(corners)

        # If single sample, unwrap batch dimension
        if single_sample:
            all_corners = [[corners[0]] for corners in all_corners]

        return all_corners

    def _compute_single_mma(
        self,
        field: torch.Tensor,
        device: torch.device
    ) -> List[List[tuple]]:
        """
        Compute MMA corner representation for a single scalar field.

        Args:
            field: Single field sample of shape (H, W)
            device: Device to place output tensors

        Returns:
            List of corner data (one per homology dimension)
            Each element is a list of (births, deaths) tuples
        """
        # Compute gradient (keep on same device as field for computation)
        gradient = self.compute_gradient(field)

        # Construct bifiltration (move to CPU for multipers)
        field_cpu = field.cpu()
        gradient_cpu = gradient.cpu()
        bifiltration = torch.stack([field_cpu, gradient_cpu], dim=-1)
        cubical = Cubical(bifiltration)
        mma = mp.module_approximation(cubical)

        # Extract corners for each homology dimension
        corner_data = []
        for dimension in self.dimensions:
            module = mma.get_module_of_degree(dimension)

            # Create grid from filtration values (use CPU tensors)
            from multipers.torch.diff_grids import get_grid, evaluate_mod_in_grid

            filtration_values = [field_cpu.flatten(), gradient_cpu.flatten()]
            grid_function = get_grid('exact')
            grid = grid_function(filtration_values)

            # Evaluate module on grid to get corners
            result = evaluate_mod_in_grid(module, grid)

            # result is a list of (births, deaths) tuples
            # Convert to proper device
            result_device = []
            for births, deaths in result:
                births_dev = births.to(device) if births.numel() > 0 else births
                deaths_dev = deaths.to(device) if deaths.numel() > 0 else deaths
                result_device.append((births_dev, deaths_dev))

            corner_data.append(result_device)

        return corner_data
