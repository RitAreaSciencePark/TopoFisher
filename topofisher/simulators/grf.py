"""
Gaussian Random Field (GRF) simulator.

Supports pivot-scale reparameterization for decorrelating amplitude
and spectral index parameters in Fisher analysis.
"""
from typing import Optional, Union
import torch
import numpy as np

try:
    import powerbox as pbox
except ImportError:
    raise ImportError("powerbox is required for GRF simulation. Install with: pip install powerbox")

from . import Simulator


class GRFSimulator(Simulator):
    """
    Gaussian Random Field simulator using power-law power spectrum.

    The power spectrum is: P(k) = A_s * (k / k_pivot)^(-B)
    where A_s is the amplitude at the pivot scale and B is the spectral index.

    When k_pivot = 1 (default), this reduces to the original P(k) = A * k^(-B).
    When k_pivot = 'auto', k_pivot is computed as the geometric mean of the
    nonzero FFT k-modes: k_pivot = exp(mean(log(k[k>0]))).

    The pivot-scale parameterization decorrelates the Fisher information
    between A_s and B, giving a diagonal (or near-diagonal) theoretical
    Fisher matrix. This is analogous to the CMB (A_s, n_s) parameterization.

    At fiducial B=0, the power spectrum is flat and A_s = A regardless of
    k_pivot, so generated fields are identical.
    """

    def __init__(
        self,
        N: int,
        dim: int = 2,
        boxlength: float = None,
        ensure_physical: bool = False,
        vol_normalised_power: bool = True,
        device: str = "cpu",
        k_pivot: Union[float, str, None] = None,
    ):
        """
        Initialize GRF simulator.

        Args:
            N: Number of grid points along each dimension
            dim: Number of dimensions (2 or 3)
            boxlength: Physical size of the simulation box (default: N)
            ensure_physical: Whether to ensure physicality of the field
            vol_normalised_power: Whether power spectrum is volume normalized
            device: Device to place generated tensors on ('cpu' or 'cuda')
            k_pivot: Pivot scale for power spectrum parameterization.
                - None or 1.0: No pivot (original P(k) = A * k^(-B))
                - 'auto': Compute geometric mean of nonzero FFT k-modes
                - float > 0: Use specified pivot scale
        """
        self.N = N
        self.dim = dim
        # TODO: Accommodate other boxlength values. Currently fixed to N for consistency
        # between real-space and Fourier-space sampling (dx = dk = 1).
        self.boxlength = float(N) if boxlength is None else boxlength
        self.ensure_physical = ensure_physical
        self.vol_normalised_power = vol_normalised_power
        self.device = device

        # Resolve k_pivot
        if k_pivot is None:
            self.k_pivot = 1.0  # Backward compatible: P(k) = A * k^(-B)
        elif isinstance(k_pivot, str) and k_pivot.lower() == 'auto':
            self.k_pivot = self._compute_k_pivot()
        else:
            self.k_pivot = float(k_pivot)
            if self.k_pivot <= 0:
                raise ValueError(f"k_pivot must be positive, got {self.k_pivot}")

    def _compute_k_pivot(self) -> float:
        """
        Compute the optimal pivot scale as the geometric mean of nonzero
        FFT k-modes. This choice makes the theoretical Fisher matrix
        exactly diagonal for Gaussian fields.

        Returns:
            k_pivot: Geometric mean of nonzero k-modes
        """
        # Create a temporary powerbox to get the k-grid
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: np.ones_like(k),  # Dummy spectrum
            boxlength=self.boxlength,
        )
        k_grid = pb.k()
        k_flat = k_grid.flatten()
        k_nonzero = k_flat[k_flat > 0]
        k_pivot = float(np.exp(np.mean(np.log(k_nonzero))))
        return k_pivot

    def __repr__(self):
        """String representation showing configuration."""
        pivot_str = f", k_pivot={self.k_pivot:.4f}" if self.k_pivot != 1.0 else ""
        return (f"GRFSimulator(N={self.N}, dim={self.dim}, "
                f"boxlength={self.boxlength}{pivot_str}, device='{self.device}')")

    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        return {
            'N': self.N,
            'dim': self.dim,
            'boxlength': self.boxlength,
            'ensure_physical': self.ensure_physical,
            'vol_normalised_power': self.vol_normalised_power,
            'k_pivot': self.k_pivot,
        }

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        desc: str = None
    ) -> torch.Tensor:
        """
        Generate GRF samples.

        Calls parent class generate() and moves result to device.

        Args:
            theta: Parameter tensor [A_s, B] where A_s is amplitude at pivot
                   scale, B is spectral index
            n_samples: Number of GRF realizations to generate
            seed: Random seed for reproducibility
            desc: Description for progress bar (if None, no progress bar)

        Returns:
            Tensor of shape (n_samples, N, N) for 2D or (n_samples, N, N, N) for 3D
        """
        # Validate parameters
        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [A, B], got {theta.numel()}")

        # Call parent class generate (handles seeding, looping, stacking)
        result = super().generate(theta, n_samples, seed, desc=desc)

        # Move to device
        return result.to(self.device)

    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single GRF realization.

        Uses the pivot-scale power spectrum: P(k) = A_s * (k / k_pivot)^(-B)

        Args:
            theta: Parameter tensor [A_s, B] where A_s is amplitude at pivot
                   scale, B is spectral index
            seed: Random seed for this sample

        Returns:
            Numpy array of shape (N, N) for 2D or (N, N, N) for 3D
        """
        # Extract parameters
        A = float(theta[0])
        B = float(theta[1])
        k_piv = self.k_pivot

        # Generate single GRF
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * (k / k_piv)**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
            seed=seed
        )
        grf = pb.delta_x()  # Real space field (numpy array)

        return grf

    def theoretical_fisher_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Calculate the theoretical Fisher information matrix for GRF.

        This is the Fisher matrix for the power spectrum parameters [A_s, B]
        using the pivot-scale parameterization:
            P(k) = A_s * (k / k_pivot)^(-B)

        The pivot scale is chosen so that the Fisher matrix is diagonal
        (or nearly so), decorrelating the amplitude and spectral index.

        Args:
            theta: Parameter tensor [A_s, B] where A_s is amplitude at pivot
                   scale, B is spectral index

        Returns:
            Theoretical Fisher matrix of shape (2, 2)

        Raises:
            ValueError: If dim is not 2 (only implemented for 2D)
        """
        if self.dim != 2:
            raise ValueError("Theoretical Fisher matrix only implemented for 2D GRFs")

        if theta.numel() != 2:
            raise ValueError(f"Expected 2 parameters [A_s, B], got {theta.numel()}")

        A = float(theta[0])
        B = float(theta[1])
        k_piv = self.k_pivot

        # Create powerbox with same settings as simulation
        pb = pbox.PowerBox(
            N=self.N,
            dim=self.dim,
            pk=lambda k: A * (k / k_piv)**(-B),
            boxlength=self.boxlength,
            vol_normalised_power=self.vol_normalised_power,
            ensure_physical=self.ensure_physical,
        )

        # Get k values (Fourier modes)
        # Sum over ALL nonzero modes with the 1/2 prefactor from the
        # Gaussian Fisher trace formula: F_ij = (1/2) Tr(C^-1 dC/di C^-1 dC/dj).
        # For diagonal C in Fourier space this simplifies to scalar sums.
        k = pb.k()
        k_all = k.flatten()
        k_modes = k_all[k_all > 0]  # exclude only the DC (k=0) mode

        # For diagonal covariance, the trace products reduce to scalar sums:
        #   F_ij = (1/2) * sum_k [ (1/P_k) * dP_k/dθ_i * (1/P_k) * dP_k/dθ_j ]
        # dP/dA = (k/k_pivot)^(-B)  =>  (1/P) dP/dA = 1/A
        # dP/dB = -P * ln(k/k_pivot)  =>  (1/P) dP/dB = -ln(k/k_pivot)
        log_ratio = np.log(k_modes / k_piv)

        F_AA = 0.5 * np.sum(1.0 / A**2 * np.ones_like(k_modes))
        F_AB = 0.5 * np.sum(-1.0 / A * log_ratio)
        F_BA = F_AB
        F_BB = 0.5 * np.sum(log_ratio**2)

        fisher_matrix = np.array([[F_AA, F_AB], [F_BA, F_BB]])

        return torch.tensor(fisher_matrix, dtype=torch.float32)
