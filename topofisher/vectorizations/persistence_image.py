"""
Persistence Image vectorization using GUDHI.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from gudhi.representations import PersistenceImage


def _persistence_weight(x):
    """Weight function using sqrt(persistence) for persistence images."""
    if len(x.shape) == 1:
        return np.sqrt(x[1])
    return np.sqrt(x[:, 1])


def _uniform_weight(x):
    """Uniform weight function for persistence images."""
    return 1.0


class PersistenceImageLayer(nn.Module):
    """
    Persistence Image vectorization using GUDHI.

    Converts persistence diagrams to persistence images using Gaussian kernels.
    Based on Adams et al. 2017 (https://jmlr.org/papers/v18/16-337.html).
    """

    def __init__(
        self,
        n_pixels: int = 16,
        bandwidth: float = 1.0,
        homology_dimensions: List[int] = [0, 1],
        weighting: str = "persistence"
    ):
        """
        Initialize Persistence Image layer.

        Args:
            n_pixels: Resolution of the image (n_pixels x n_pixels)
            bandwidth: Bandwidth for Gaussian kernel (default 1.0, tune as hyperparameter)
            homology_dimensions: Which homology dimensions to include (default: [0, 1])
            weighting: Weighting scheme - "persistence" uses sqrt(persistence), "uniform" uses 1.0
        """
        super().__init__()
        self.n_pixels = n_pixels
        self.bandwidth = bandwidth
        self.homology_dimensions = homology_dimensions
        self.weighting = weighting
        self.n_features = len(homology_dimensions) * n_pixels * n_pixels

        # Image bounds will be computed from ALL data (fiducial + all derivatives)
        self.im_range = None
        self.pi_layers = {}  # One PersistenceImage per homology dimension

        # Use named functions instead of lambdas for pickle compatibility
        if weighting == "persistence":
            self.weight_fn = _persistence_weight
        elif weighting == "uniform":
            self.weight_fn = _uniform_weight
        else:
            raise ValueError(f"Unknown weighting: {weighting}. Use 'persistence' or 'uniform'")

    def fit(self, all_diagrams_list):
        """
        Compute fixed bounds across ALL diagrams (fiducial, theta_minus, theta_plus for all params).

        Args:
            all_diagrams_list: Either:
                - List of diagram sets [fiducial_diagrams, theta_minus_0, ...],
                  where each element has structure List[hom_dim][sample] of diagrams.
                - Flat list of diagram tensors (when called from CombinedVectorization).
        """
        # Handle flat list of tensors (from CombinedVectorization)
        # Treat as single homology dimension h_dim=0
        if (all_diagrams_list
                and isinstance(all_diagrams_list[0], torch.Tensor)):
            flat_diagrams = all_diagrams_list
            # Override to single-dimension mode
            self.homology_dimensions = [0]
            self.n_features = self.n_pixels * self.n_pixels
            # Wrap into expected structure: one set, one hom_dim containing all diagrams
            all_diagrams_list = [[flat_diagrams]]
        # Collect all birth-persistence values for each homology dimension
        all_data = {h_dim: {'births': [], 'pers': []} for h_dim in self.homology_dimensions}

        for diagrams_set in all_diagrams_list:
            # diagrams_set is List[hom_dim][sample] - iterate over homology dimensions
            for hom_dim_idx, h_dim in enumerate(self.homology_dimensions):
                if hom_dim_idx < len(diagrams_set):
                    # Get all diagrams for this homology dimension
                    diagrams_for_dim = diagrams_set[hom_dim_idx]
                    for dgm in diagrams_for_dim:
                        if dgm.shape[0] > 0:
                            dgm_np = dgm.detach().cpu().numpy()
                            births = dgm_np[:, 0]
                            pers = dgm_np[:, 1] - dgm_np[:, 0]
                            # Filter positive persistence
                            valid = pers > 0
                            if valid.any():
                                all_data[h_dim]['births'].extend(births[valid])
                                all_data[h_dim]['pers'].extend(pers[valid])

        # Compute bounds for each homology dimension
        for h_dim in self.homology_dimensions:
            if len(all_data[h_dim]['births']) == 0:
                # No valid points, use default bounds
                im_range = [0.0, 1.0, 0.0, 1.0]
                import warnings
                warnings.warn(f"No valid points found for homology dimension {h_dim}. Using default bounds {im_range}.")
            else:
                births = np.array(all_data[h_dim]['births'])
                pers = np.array(all_data[h_dim]['pers'])

                # Compute bounds with 10% margin
                birth_min, birth_max = births.min(), births.max()
                pers_min, pers_max = pers.min(), pers.max()

                birth_range = birth_max - birth_min
                pers_range = pers_max - pers_min

                if birth_range < 1e-6:
                    birth_min, birth_max = 0.0, 1.0
                else:
                    margin = 0.1 * birth_range
                    birth_min -= margin
                    birth_max += margin

                if pers_range < 1e-6:
                    pers_min, pers_max = 0.0, 1.0
                else:
                    margin = 0.1 * pers_range
                    pers_min = max(0.0, pers_min - margin)
                    pers_max += margin

                # GUDHI expects im_range as [x_min, x_max, y_min, y_max]
                im_range = [birth_min, birth_max, pers_min, pers_max]

            # Create GUDHI PersistenceImage for this homology dimension
            self.pi_layers[h_dim] = PersistenceImage(
                bandwidth=self.bandwidth,
                resolution=[self.n_pixels, self.n_pixels],
                weight=self.weight_fn,
                im_range=im_range
            )

        if not getattr(self, '_fitted', False):
            print(f"Fitted PersistenceImageLayer with bounds:")
            for h_dim in self.homology_dimensions:
                bounds = self.pi_layers[h_dim].im_range
                # im_range is [birth_min, birth_max, pers_min, pers_max]
                print(f"  H{h_dim}: birth [{bounds[0]:.3f}, {bounds[1]:.3f}], "
                      f"pers [{bounds[2]:.3f}, {bounds[3]:.3f}]")
        self._fitted = True

    def forward(self, all_diagrams) -> torch.Tensor:
        """
        Convert persistence diagrams to persistence images.

        Args:
            all_diagrams: Either:
                - List[hom_dim][sample_idx] of diagrams (standard format)
                - Flat List[sample_idx] of diagrams (from CombinedVectorization)

        Returns:
            Tensor of shape (n_samples, total_features) where 
            total_features = n_channels * n_pixels * n_pixels
        """
        if not self.pi_layers:
            raise RuntimeError("PersistenceImageLayer must be fitted before use. Call .fit() first.")

        # Handle flat list of tensors (from CombinedVectorization)
        if all_diagrams and isinstance(all_diagrams[0], torch.Tensor):
            all_diagrams = [all_diagrams]

        # Determine batch size from first homology dimension
        batch_size = len(all_diagrams[0]) if len(all_diagrams) > 0 else 0
        n_channels = len(self.homology_dimensions)

        # Determine device from first non-empty diagram
        device = torch.device('cpu')
        for hom_dim_diagrams in all_diagrams:
            if len(hom_dim_diagrams) > 0 and hom_dim_diagrams[0].numel() > 0:
                device = hom_dim_diagrams[0].device
                break

        images = torch.zeros(
            batch_size, n_channels, self.n_pixels, self.n_pixels,
            device=device
        )

        # Iterate over homology dimensions
        for ch_idx, h_dim in enumerate(self.homology_dimensions):
            if h_dim < len(all_diagrams):
                diagrams_for_dim = all_diagrams[h_dim]
                # Iterate over samples
                for sample_idx, dgm in enumerate(diagrams_for_dim):
                    if dgm.shape[0] > 0:
                        images[sample_idx, ch_idx] = self._diagram_to_image(dgm, h_dim, device)

        # Flatten to (batch_size, n_channels * n_pixels * n_pixels) for MOPED
        return images.view(batch_size, -1)

    def _diagram_to_image(self, dgm: torch.Tensor, h_dim: int, device: torch.device) -> torch.Tensor:
        """
        Convert a single persistence diagram to an image using GUDHI.

        Args:
            dgm: Diagram of shape (n_points, 2) with (birth, death) pairs
            h_dim: Homology dimension
            device: Target device for output tensor

        Returns:
            Image of shape (n_pixels, n_pixels)
        """
        if dgm.shape[0] == 0:
            return torch.zeros(self.n_pixels, self.n_pixels, device=device)

        # Convert to numpy for GUDHI
        dgm_np = dgm.detach().cpu().numpy()

        # Filter out zero/negative persistence
        persistence = dgm_np[:, 1] - dgm_np[:, 0]
        valid_mask = persistence > 0
        if not valid_mask.any():
            return torch.zeros(self.n_pixels, self.n_pixels, device=device)

        dgm_np = dgm_np[valid_mask]

        # GUDHI expects (birth, death) coordinates
        # It will internally transform to (birth, persistence) using BirthPersistenceTransform
        birth_death = dgm_np  # Already in (birth, death) format

        # Generate persistence image using GUDHI
        # fit_transform expects a list of diagrams
        image_flat = self.pi_layers[h_dim].fit_transform([birth_death])[0]

        # Reshape to 2D image
        image = image_flat.reshape(self.n_pixels, self.n_pixels)

        # Convert back to torch tensor
        return torch.from_numpy(image).float().to(device)
