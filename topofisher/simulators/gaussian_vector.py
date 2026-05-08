"""
Simple Gaussian vector simulator for testing.

Generates d-dimensional vectors from a multivariate normal distribution
with customizable mean and covariance matrix.
"""
from typing import Optional
import torch
import torch.nn as nn

from . import Simulator


class GaussianVectorSimulator(Simulator):
    """
    Simple simulator for d-dimensional Gaussian vectors.

    Useful for testing the pipeline with a well-understood analytical case.
    The theoretical Fisher matrix is known: F = Σ^{-1} for mean parameters.
    """

    def __init__(
        self,
        d: int,
        covariance: Optional[torch.Tensor] = None,
        device: str = "cpu"
    ):
        """
        Initialize Gaussian vector simulator.

        Args:
            d: Dimension of vectors
            covariance: Covariance matrix (d x d). If None, uses identity.
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()
        self.d = d
        self.device = device

        # Set covariance
        if covariance is None:
            self.covariance = torch.eye(d, device=device)
        else:
            assert covariance.shape == (d, d), f"Covariance must be {d}x{d}, got {covariance.shape}"
            self.covariance = covariance.to(device)

        # Compute inverse covariance for theoretical Fisher
        self.inv_covariance = torch.linalg.inv(self.covariance)

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples from multivariate normal distribution.

        Calls parent class generate() and moves result to device.

        Args:
            theta: Mean vector of shape (d,)
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (n_samples, d)
        """
        # Call parent class generate (handles seeding, looping, stacking)
        result = super().generate(theta, n_samples, seed)

        # Move to device
        return result.to(self.device)

    def generate_single(self, theta: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Generate a single sample from multivariate normal distribution.

        Args:
            theta: Mean vector of shape (d,)
            seed: Random seed for this sample

        Returns:
            Single sample tensor of shape (d,)
        """
        # Set torch seed for this sample
        torch.manual_seed(seed)

        # Create multivariate normal distribution
        mean = theta.to(self.device)
        dist = torch.distributions.MultivariateNormal(mean, self.covariance)

        # Generate single sample
        sample = dist.sample()

        return sample

    def theoretical_fisher_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute theoretical Fisher information matrix.

        For a multivariate Gaussian with fixed covariance and mean parameters,
        the Fisher matrix is simply F = Σ^{-1}.

        Args:
            theta: Mean parameters (not actually used, included for interface consistency)

        Returns:
            Fisher matrix of shape (d, d)
        """
        return self.inv_covariance.clone()

    def __repr__(self):
        return f"GaussianVectorSimulator(d={self.d}, device={self.device})"
