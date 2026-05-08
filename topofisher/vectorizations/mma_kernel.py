"""
MMA Kernel vectorization layer.

Self-contained implementation with all kernel and distance functions included.
Fully GPU-compatible by preserving tensor devices throughout computation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Literal, Optional
from multipers import grids


# ============= Distance and Weight Functions (GPU-compatible) =============

def is_finite(I):
    """Check if interval is finite."""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    return ~(inf_birth_idx.all())


def dI0(I):
    """Interleaving to 0 of an interval (max diag in support)."""
    inf_birth_idx = torch.any(I[0] == torch.inf, axis=-1)
    B = I[0][~inf_birth_idx, None, :]
    D = I[1][None, :, :]
    pairwise_d = (torch.nn.functional.relu(D - B)).min(dim=2).values
    interleaving_to_0 = pairwise_d.max(0).values.max(0).values

    # GPU-compatible: create tensor on same device as input
    if interleaving_to_0 == torch.inf:
        return torch.tensor(0., device=I[0].device, dtype=I[0].dtype)
    return interleaving_to_0


def interleaving_weights(mma_diff):
    """Compute interleaving weights for all intervals."""
    if len(mma_diff) == 0:
        return torch.tensor([], device=mma_diff[0][0].device if mma_diff else torch.device('cpu'))

    weights = []
    for I in mma_diff:
        weights.append(dI0(I).unsqueeze(0))

    return torch.cat(weights)


def distance_birth(B, x):
    """Compute distance from points to birth corners."""
    B_exp = B.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (B_exp - x_exp).max(-1).values.min(-1).values
    return dist


def distance_death(D, x):
    """Compute distance from points to death corners."""
    D_exp = D.unsqueeze(1)
    x_exp = x.unsqueeze(0).unsqueeze(2)
    dist = (x_exp - D_exp).max(-1).values.min(-1).values
    return dist


def distance_to(mma_diff, x):
    """Compute distance from points to all intervals."""
    if x.ndim == 1:
        x = x.unsqueeze(0)
    assert x.ndim == 2

    if len(mma_diff) == 0:
        return torch.zeros(0, x.shape[0], device=x.device, dtype=x.dtype)

    nB = max([I[0].shape[0] for I in mma_diff])
    nD = max([I[1].shape[0] for I in mma_diff])

    # Create tensors on same device as x
    device = x.device
    dtype = x.dtype

    Bs = torch.zeros(size=(len(mma_diff), nB, 2), dtype=dtype, device=device) + torch.inf
    Ds = torch.zeros(size=(len(mma_diff), nD, 2), dtype=dtype, device=device) - torch.inf

    for i, (B, D) in enumerate(mma_diff):
        Bs[i, :len(B), :] = B.to(dtype).to(device)
        Ds[i, :len(D), :] = D.to(dtype).to(device)

    birth_dist = distance_birth(Bs, x)
    death_dist = distance_death(Ds, x)

    return torch.maximum(birth_dist, death_dist)


# ============= Kernel Functions (GPU-compatible) =============

def linear_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Linear kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.where(x < bandwidth, (bandwidth - x) / bandwidth, torch.zeros_like(x)) * (weights[:, None] ** p)).sum(0)


def gaussian_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Gaussian kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-0.5 * ((x / bandwidth) ** 2)) * (weights[:, None] ** p)).sum(0)


def exponential_kernel(dist, weights, bandwidth, p=2, signed=False):
    """Exponential kernel for MMA vectorization."""
    s = torch.where(dist >= 0, 1, -1) if signed else 1
    x = torch.abs(dist) if signed else torch.relu(dist)
    return (s * torch.exp(-(x / bandwidth)) * (weights[:, None] ** p)).sum(0)


# ============= PyTorch Module =============

class MMAKernelLayer(nn.Module):
    """
    MMA Kernel vectorization for a single homology dimension.

    Converts MMA corner data into a fixed-size vector by evaluating a kernel function
    on a regular grid in the 2-parameter space. Fully GPU-compatible.

    Use with CombinedVectorization to handle multiple homology dimensions:
        vectorization = CombinedVectorization([
            MMAGaussianLayer(resolution=30, bandwidth=0.05),  # H0
            MMAGaussianLayer(resolution=30, bandwidth=0.05)   # H1
        ])
    """

    def __init__(
        self,
        kernel: Literal['gaussian', 'linear', 'exponential'] = 'gaussian',
        resolution: int = 30,
        bandwidth: float = 0.05,
        p: float = 2,
        signed: bool = False,
        return_flat: bool = True,
        fixed_box: Optional[np.ndarray] = None
    ):
        """
        Initialize MMA Kernel layer.

        Args:
            kernel: Type of kernel function:
                - 'gaussian': exp(-0.5 * (distance/bandwidth)^2) - smooth, differentiable
                - 'linear': max(0, (bandwidth - distance) / bandwidth) - compact support
                - 'exponential': exp(-distance/bandwidth) - heavier tails than gaussian
            resolution: Grid resolution for evaluation (creates resolution x resolution grid)
            bandwidth: Kernel bandwidth parameter (controls kernel width)
            p: Power for interleaving weights (weights^p in kernel sum)
            signed: Whether to use signed kernel (assigns sign based on distance direction)
            return_flat: If True, returns flattened vector of size resolution^2
            fixed_box: Optional fixed bounding box of shape (2, 2) [[min_field, max_field], [min_grad, max_grad]]
                      If provided, all samples use this box. If None, compute per-sample boxes.
                      Use fit() method to compute from samples.
        """
        super().__init__()

        self.kernel = kernel
        self.resolution = resolution
        self.bandwidth = bandwidth
        self.p = p
        self.signed = signed
        self.return_flat = return_flat
        self.fixed_box = fixed_box

        # Select kernel function
        self.kernel_func = {
            'linear': linear_kernel,
            'gaussian': gaussian_kernel,
            'exponential': exponential_kernel
        }[kernel]

        # Output features: resolution^2 for flat vector
        self.n_features = resolution * resolution if return_flat else None

    def fit(self, corner_data: List[List[tuple]]) -> 'MMAKernelLayer':
        """
        Fit the bounding box from sample data.

        Args:
            corner_data: List[sample][interval] -> (births, deaths)
                        Corner data from multiple samples to compute bounding box

        Returns:
            self (for method chaining)
        """
        all_corners = []

        for sample in corner_data:
            for births, deaths in sample:
                if births.numel() > 0:
                    all_corners.append(births)
                if deaths.numel() > 0:
                    all_corners.append(deaths)

        if len(all_corners) == 0:
            raise ValueError("No corners found in data to fit bounding box")

        all_corners_cat = torch.cat(all_corners, dim=0)
        box_min = all_corners_cat.min(dim=0).values.cpu().numpy()
        box_max = all_corners_cat.max(dim=0).values.cpu().numpy()
        self.fixed_box = np.stack([box_min, box_max]).T

        print(f"Fitted bounding box: [{box_min[0]:.3f}, {box_max[0]:.3f}] × [{box_min[1]:.3f}, {box_max[1]:.3f}]")

        return self

    def forward(self, corner_data: List[List[tuple]]) -> torch.Tensor:
        """
        Vectorize MMA corners using kernel methods.

        Args:
            corner_data: List[sample][interval] -> (births, deaths)
                        where births and deaths are tensors of shape (n_corners, 2)
                        This is the corner data for a SINGLE homology dimension.

        Returns:
            Tensor of shape (n_samples, resolution^2) if return_flat=True
            Tensor of shape (n_samples, resolution, resolution) if return_flat=False

        Note:
            Fully GPU-compatible. If input data is on GPU, all computations
            will be performed on GPU.
        """
        n_samples = len(corner_data)
        features = []

        for sample_idx in range(n_samples):
            # Get intervals for this sample (for single homology dimension)
            intervals = corner_data[sample_idx]

            if len(intervals) == 0:
                # No intervals - return zeros
                if self.return_flat:
                    vec = torch.zeros(self.resolution * self.resolution)
                else:
                    vec = torch.zeros(self.resolution, self.resolution)
                features.append(vec)
                continue

            # Collect all corners to compute bounding box
            all_corners = []
            for births, deaths in intervals:
                if births.numel() > 0:
                    all_corners.append(births)
                if deaths.numel() > 0:
                    all_corners.append(deaths)

            if len(all_corners) == 0:
                # No corners - return zeros
                if self.return_flat:
                    vec = torch.zeros(self.resolution * self.resolution)
                else:
                    vec = torch.zeros(self.resolution, self.resolution)
                features.append(vec)
                continue

            all_corners_cat = torch.cat(all_corners, dim=0)

            # Use fixed box if available, otherwise compute per-sample box
            if self.fixed_box is not None:
                box = self.fixed_box
            else:
                # Create bounding box (no margin - matches vectorize_kernel behavior)
                box_min = all_corners_cat.min(dim=0).values.cpu().numpy()
                box_max = all_corners_cat.max(dim=0).values.cpu().numpy()
                box = np.stack([box_min, box_max]).T

            # Create evaluation grid using multipers
            R = grids.compute_grid(box, strategy="regular", resolution=self.resolution)
            R_dense = grids.todense(R)
            R_dense = torch.from_numpy(R_dense).to(torch.float32)

            # Move to same device as data
            device = all_corners_cat.device
            R_dense = R_dense.to(device)

            # Compute interleaving weights (GPU-compatible)
            w = interleaving_weights(intervals)

            # Compute distances (GPU-compatible)
            SD = distance_to(intervals, R_dense)

            # Apply kernel function
            img = self.kernel_func(SD, w, self.bandwidth, p=self.p, signed=self.signed)

            # Reshape if needed
            if not self.return_flat:
                img = img.reshape(self.resolution, self.resolution)

            features.append(img)

        return torch.stack(features)

    def __repr__(self):
        return (f"MMAKernelLayer(kernel='{self.kernel}', resolution={self.resolution}, "
                f"bandwidth={self.bandwidth}, p={self.p}, signed={self.signed})")


# Convenience classes for specific kernels
class MMAGaussianLayer(MMAKernelLayer):
    """
    MMA Gaussian Kernel Vectorization Layer.

    Uses Gaussian kernel: exp(-0.5 * (distance/bandwidth)^2)
    Recommended for gradient-based learning due to smoothness.
    Fully GPU-compatible.
    """
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None):
        super().__init__('gaussian', resolution, bandwidth, p, signed, return_flat, fixed_box)


class MMALinearLayer(MMAKernelLayer):
    """
    MMA Linear Kernel Vectorization Layer.

    Uses linear kernel: max(0, (bandwidth - distance) / bandwidth)
    Has compact support (zero outside bandwidth), computationally faster.
    Fully GPU-compatible.
    """
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None):
        super().__init__('linear', resolution, bandwidth, p, signed, return_flat, fixed_box)


class MMAExponentialLayer(MMAKernelLayer):
    """
    MMA Exponential Kernel Vectorization Layer.

    Uses exponential kernel: exp(-distance/bandwidth)
    Compromise between Gaussian (smooth) and Linear (compact support).
    Fully GPU-compatible.
    """
    def __init__(self, resolution=30, bandwidth=0.05, p=2, signed=False, return_flat=True, fixed_box=None):
        super().__init__('exponential', resolution, bandwidth, p, signed, return_flat, fixed_box)
