"""
Noisy Ring (Circle) simulator for point cloud generation.

Generates 2D point clouds with points sampled from:
- A noisy ring with radius ~ Normal(mu, sigma)
- Uniform background points
"""
from typing import Optional
import torch
import numpy as np

from . import Simulator


class NoisyRingSimulator(Simulator):
    """
    Noisy Ring simulator for generating 2D point clouds.

    The point cloud consists of:
    - ncirc points from a noisy ring: radius ~ Normal(theta[0], theta[1])
    - nback points from background: radius ~ Uniform(0, 2*bgm_avg)

    All points have uniform angular distribution.

    Parameters theta = [radius_mean, radius_std]
    """

    def __init__(
        self,
        ncirc: int,
        nback: int,
        bgm_avg: float,
        device: str = "cpu"
    ):
        """
        Initialize NoisyRing simulator.

        Args:
            ncirc: Number of points drawn from the noisy ring
            nback: Number of points drawn from the background
            bgm_avg: Mean distance to center for background points
            device: Device to place generated tensors on ('cpu' or 'cuda')

        Example:
            >>> sim = NoisyRingSimulator(ncirc=200, nback=20, bgm_avg=1.0)
            >>> theta = torch.tensor([1.0, 0.2])  # radius_mean, radius_std
            >>> points = sim.generate(theta, n_samples=100)  # (100, 220, 2)
        """
        self.ncirc = ncirc
        self.nback = nback
        self.bgm_avg = bgm_avg
        self.ntot = ncirc + nback
        self.p = nback / self.ntot  # Probability of background point
        self.device = device

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate noisy ring point cloud samples.

        Calls parent class generate() and moves result to device.

        Args:
            theta: Parameter tensor [radius_mean, radius_std]
            n_samples: Number of point clouds to generate
            seed: Random seed for reproducibility

        Returns:
            Tensor of shape (n_samples, ntot, 2) where ntot = ncirc + nback
        """
        # Validate parameters
        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [radius_mean, radius_std], got {theta.numel()}")

        # Call parent class generate (handles seeding, looping, stacking)
        result = super().generate(theta, n_samples, seed)

        # Move to device
        return result.to(self.device)

    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single noisy ring point cloud.

        Args:
            theta: Parameter tensor [radius_mean, radius_std]
            seed: Random seed for this sample

        Returns:
            Numpy array of shape (ntot, 2) - 2D point cloud
        """
        # Extract parameters
        radius_mean = float(theta[0])
        radius_std = float(theta[1])

        # Set random seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Generate radii using mixture distribution
        # nback points: Uniform(0, 2*bgm_avg)
        # ncirc points: Normal(radius_mean, radius_std)
        rad_back = torch.rand(self.nback, generator=generator) * 2 * self.bgm_avg
        rad_circ = torch.randn(self.ncirc, generator=generator) * radius_std + radius_mean

        # Concatenate radii
        radii = torch.cat([rad_back, rad_circ])

        # Generate uniform angles [0, 2π]
        angles = 2 * np.pi * torch.rand(self.ntot, generator=generator)

        # Convert polar to Cartesian coordinates
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        points = torch.stack([x, y], dim=-1)

        # Return as numpy array
        return points.numpy()

    def theoretical_fisher_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculate the theoretical Fisher information matrix for noisy ring.

        Computes Fisher matrix by numerical integration of the score function
        correlation over the radius distribution.

        Args:
            theta: Parameter tensor [radius_mean, radius_std]

        Returns:
            Theoretical Fisher matrix of shape (2, 2)
        """
        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [radius_mean, radius_std], got {theta.numel()}")

        radius_mean = theta[0]
        radius_std = theta[1]

        # Create integration range for radii
        # Cover [0, 2*bgm_avg] to capture full distribution support
        xr = torch.linspace(0.0, 2 * self.bgm_avg, 2000)

        # Mixture distribution: p*Uniform + (1-p)*Normal
        p = self.p

        # Uniform component: 1 / (2*bgm_avg) for x in [0, 2*bgm_avg]
        uniform_prob = torch.ones_like(xr) / (2 * self.bgm_avg)

        # Normal component: Normal(radius_mean, radius_std)
        normal_dist = torch.distributions.Normal(radius_mean, radius_std)
        normal_prob = torch.exp(normal_dist.log_prob(xr))

        # Mixture probability
        prob = p * uniform_prob + (1 - p) * normal_prob

        # Analytical score function: gradient of log probability
        # For mixture: ∂/∂θ log p(x|θ) = (1-p) * N(x|μ,σ) / p(x|θ) * ∂/∂θ log N(x|μ,σ)
        #
        # For Normal component:
        #   d/dμ log N(x|μ,σ) = (x-μ)/σ²
        #                       d/dσ log N(x|μ,σ) = -1/σ + (x-μ)²/σ³
        score_mean = (1 - p) * normal_prob * (xr - radius_mean) / (radius_std ** 2) / prob
        score_std = (1 - p) * normal_prob * (-1/radius_std + (xr - radius_mean)**2 / radius_std**3) / prob

        # Stack scores
        score = torch.stack([score_mean, score_std], dim=-1)  # (2000, 2)

        # Normalize probabilities for weighted average
        probs_normalized = prob / torch.sum(prob)

        # Weighted score
        weighted_score = score * probs_normalized.unsqueeze(-1)

        # Fisher matrix: F = E[score * score^T] * ntot
        fisher_matrix = torch.matmul(score.T, weighted_score) * self.ntot

        return fisher_matrix.float()

    def sorted_distance_summary(self, point_clouds):
        """
        Compute sorted distances to origin for each point cloud.

        Args:
            point_clouds: List of point clouds or tensor of shape (n_samples, ntot, 2)

        Returns:
            List of sorted distance tensors, each of shape (ntot,)
        """
        if isinstance(point_clouds, torch.Tensor):
            # Convert to list of tensors
            point_clouds = [point_clouds[i] for i in range(point_clouds.shape[0])]

        sorted_dists = []
        for pts in point_clouds:
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)

            # Compute distances to origin
            dists = torch.norm(pts, dim=-1)

            # Sort in ascending order
            sorted_dists.append(torch.sort(dists)[0])

        return sorted_dists

    def mean_distance_summary(self, point_clouds):
        """
        Compute mean and std of distances to origin for each point cloud.

        Args:
            point_clouds: List of point clouds or tensor of shape (n_samples, ntot, 2)

        Returns:
            List of [mean, std] tensors, each of shape (2,)
        """
        if isinstance(point_clouds, torch.Tensor):
            # Convert to list of tensors
            point_clouds = [point_clouds[i] for i in range(point_clouds.shape[0])]

        summaries = []
        for pts in point_clouds:
            if not isinstance(pts, torch.Tensor):
                pts = torch.tensor(pts, dtype=torch.float32)

            # Compute distances to origin
            dists = torch.norm(pts, dim=-1)

            # Compute mean and std
            mean = torch.mean(dists)
            std = torch.std(dists)

            summaries.append(torch.stack([mean, std]))

        return summaries
