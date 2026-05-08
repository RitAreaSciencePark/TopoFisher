"""
Fourier-transformed Gaussian Random Field (GRF) simulator.

Generates GRF and returns its Fourier transform (magnitude).
Inherits pivot-scale reparameterization from GRFSimulator.
"""
from typing import Optional
import torch
import numpy as np

from .grf import GRFSimulator


class GRFFourierSimulator(GRFSimulator):
    """
    Gaussian Random Field simulator that returns Fourier-transformed output.

    The power spectrum is: P(k) = A_s * (k / k_pivot)^(-B)
    where A_s is the amplitude at the pivot scale and B is the spectral index.
    See GRFSimulator for details on pivot-scale parameterization.

    Output is the magnitude of the 2D FFT of the GRF.
    """

    def __init__(
        self,
        N: int,
        dim: int = 2,
        boxlength: float = None,
        ensure_physical: bool = False,
        vol_normalised_power: bool = True,
        device: str = "cpu",
        output_type: str = "magnitude",  # 'magnitude', 'log_magnitude', 'power', 'real', 'imag'
        shift_fft: bool = True,  # Center zero frequency
        k_pivot = None,
    ):
        """
        Initialize Fourier GRF simulator.

        Args:
            N: Number of grid points along each dimension
            dim: Number of dimensions (2 or 3)
            boxlength: Physical size of the simulation box (default: N)
            ensure_physical: Whether to ensure physicality of the field
            vol_normalised_power: Whether power spectrum is volume normalized
            device: Device to place generated tensors on ('cpu' or 'cuda')
            output_type: What to output from FFT:
                - 'magnitude': |FFT| (default)
                - 'log_magnitude': log(|FFT| + eps)
                - 'power': |FFT|^2
                - 'real': Real part of FFT
                - 'imag': Imaginary part of FFT
            shift_fft: Whether to shift zero frequency to center (fftshift)
            k_pivot: Pivot scale (see GRFSimulator for details).
                None (no pivot), 'auto', or a positive float.
        """
        super().__init__(
            N=N,
            dim=dim,
            boxlength=boxlength,
            ensure_physical=ensure_physical,
            vol_normalised_power=vol_normalised_power,
            device=device,
            k_pivot=k_pivot,
        )
        self.output_type = output_type
        self.shift_fft = shift_fft

    def __repr__(self):
        """String representation showing configuration."""
        return (f"GRFFourierSimulator(N={self.N}, dim={self.dim}, "
                f"output_type='{self.output_type}', shift_fft={self.shift_fft}, "
                f"device='{self.device}')")

    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        config = super().get_config()
        config['output_type'] = self.output_type
        config['shift_fft'] = self.shift_fft
        return config

    def generate_single(self, theta: torch.Tensor, seed: int):
        """
        Generate a single Fourier-transformed GRF realization.

        Args:
            theta: Parameter tensor [A_s, B] where A_s is amplitude at pivot
                   scale, B is spectral index
            seed: Random seed for this sample

        Returns:
            Numpy array of shape (N, N) for 2D - Fourier transform of GRF
        """
        # Generate real-space GRF using parent class
        grf = super().generate_single(theta, seed)
        
        # Compute 2D FFT
        fft = np.fft.fft2(grf)
        
        # Optionally shift zero frequency to center
        if self.shift_fft:
            fft = np.fft.fftshift(fft)
        
        # Convert to desired output type
        if self.output_type == 'magnitude':
            output = np.abs(fft)
        elif self.output_type == 'log_magnitude':
            output = np.log(np.abs(fft) + 1e-10)
        elif self.output_type == 'power':
            output = np.abs(fft) ** 2
        elif self.output_type == 'real':
            output = np.real(fft)
        elif self.output_type == 'imag':
            output = np.imag(fft)
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
        return output
