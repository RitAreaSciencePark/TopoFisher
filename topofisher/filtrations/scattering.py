"""
Wavelet Scattering Transform filtration for 2D fields.

A principled, non-learned feature extractor that captures non-Gaussian
information at multiple scales via cascaded wavelet convolutions + modulus.

The scattering transform S[x] = {S0, S1, S2} computes:
  S0 = <x>                                  (mean)
  S1(j,l) = <|x * psi_{j,l}|>               (first-order — variance per scale)
  S2(j1,l1,j2,l2) = <||x * psi_{j1,l1}| * psi_{j2,l2}|>  (non-Gaussianity)

where psi_{j,l} are Morlet wavelets at scale j and angle l.

Advantages for Fisher analysis:
  - Deterministic: no training → no Gaussianity lottery across seeds
  - Translation/rotation equivariant → stable features
  - Second-order coefficients capture higher-order statistics
  - Feature dimension grows only polynomially with J, L

Default output dimension for 512×512 input with J=7, L=8:
  S0: 1, S1: J*L = 56, S2: sum_{j2>j1}(L^2) = 21*64 = 1344
  Total: 1401

Designed for use with IdentityVectorization and MOPED compression.
"""
import math
from typing import List, Optional
import torch
import torch.nn as nn


def morlet_2d(M, N, sigma, theta, xi, slant=0.5):
    """
    Create a 2D Morlet wavelet in spatial domain.

    Args:
        M, N: spatial dimensions
        sigma: width of the Gaussian envelope
        theta: rotation angle
        xi: carrier frequency
        slant: anisotropy ratio
    Returns:
        Complex tensor of shape (M, N)
    """
    gx = torch.arange(M, dtype=torch.float32) - M // 2
    gy = torch.arange(N, dtype=torch.float32) - N // 2
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing='ij')

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_rot = cos_t * grid_x + sin_t * grid_y
    y_rot = -sin_t * grid_x + cos_t * grid_y

    gauss = torch.exp(-0.5 * (x_rot**2 / sigma**2 + y_rot**2 / (slant * sigma)**2))
    osc = torch.exp(1j * xi * x_rot)

    psi = gauss * osc
    psi = psi - psi.mean()  # zero-mean correction
    return psi


def build_filter_bank(M, N, J, L, sigma0=0.8, xi0=3.0 / 4.0 * math.pi):
    """
    Build Morlet wavelet filter bank for scattering transform.

    Args:
        M, N: spatial size of input
        J: number of scales (octaves)
        L: number of orientations
        sigma0: base width of Gaussian envelope
        xi0: base carrier frequency

    Returns:
        dict with 'psi' (list of (j, l, fft) tuples) and 'phi' (lowpass fft)
    """
    filters = {'psi': [], 'phi': None}

    for j in range(J):
        for l in range(L):
            sigma = sigma0 * (2 ** j)
            xi = xi0 / (2 ** j)
            theta = l * math.pi / L

            psi = morlet_2d(M, N, sigma, theta, xi)
            psi_shifted = torch.fft.ifftshift(psi)
            psi_fft = torch.fft.fft2(psi_shifted)
            filters['psi'].append((j, l, psi_fft))

    # Low-pass filter (Gaussian at scale 2^J)
    sigma_phi = sigma0 * (2 ** J)
    gx = torch.arange(M, dtype=torch.float32) - M // 2
    gy = torch.arange(N, dtype=torch.float32) - N // 2
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing='ij')
    phi = torch.exp(-0.5 * (grid_x**2 + grid_y**2) / sigma_phi**2)
    phi = phi / phi.sum()
    phi_shifted = torch.fft.ifftshift(phi)
    phi_fft = torch.fft.fft2(phi_shifted)
    filters['phi'] = phi_fft

    return filters


class ScatteringTransform2D(nn.Module):
    """
    2D Wavelet Scattering Transform (non-learned).

    Computes zeroth, first, and second-order scattering coefficients.
    All convolutions are in the Fourier domain for efficiency.
    """

    def __init__(self, M, N, J=7, L=8, max_order=2):
        super().__init__()
        self.M = M
        self.N = N
        self.J = J
        self.L = L
        self.max_order = max_order

        fb = build_filter_bank(M, N, J, L)

        self.register_buffer('phi_fft', fb['phi'].unsqueeze(0))  # (1, M, N)

        psi_ffts = []
        psi_jl = []
        for j, l, psi_fft in fb['psi']:
            psi_ffts.append(psi_fft)
            psi_jl.append((j, l))
        self.register_buffer('psi_ffts', torch.stack(psi_ffts, dim=0))  # (J*L, M, N)
        self.psi_jl = psi_jl

        # Compute output dimension
        n_s0 = 1
        n_s1 = J * L
        n_s2 = 0
        if max_order >= 2:
            for j2 in range(J):
                for j1 in range(j2):
                    n_s2 += L * L
        self.output_dim = n_s0 + n_s1 + n_s2

    def _convolve_and_modulus(self, x_fft, psi_fft):
        """Convolve in Fourier domain and take modulus."""
        out_fft = x_fft * psi_fft.unsqueeze(0)
        out = torch.fft.ifft2(out_fft).real
        return torch.abs(out)

    def forward(self, x):
        """
        Compute scattering coefficients.

        Args:
            x: (batch, H, W) or (batch, 1, H, W)
        Returns:
            (batch, output_dim) tensor of scattering coefficients
        """
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x_fft = torch.fft.fft2(x)

        # S0: zeroth order
        s0 = x.mean(dim=[-2, -1]).unsqueeze(1)  # (batch, 1)

        # S1: first-order
        s1_list = []
        u1_dict = {}
        for idx, (j, l) in enumerate(self.psi_jl):
            u1 = self._convolve_and_modulus(x_fft, self.psi_ffts[idx])
            s1_list.append(u1.mean(dim=[-2, -1]))
            u1_dict[(j, l)] = u1

        s1 = torch.stack(s1_list, dim=1)
        coeffs = [s0, s1]

        # S2: second-order
        if self.max_order >= 2:
            s2_list = []
            for j2 in range(self.J):
                for j1 in range(j2):
                    for l1 in range(self.L):
                        u1 = u1_dict[(j1, l1)]
                        u1_fft = torch.fft.fft2(u1)
                        for l2 in range(self.L):
                            idx2 = j2 * self.L + l2
                            u2 = self._convolve_and_modulus(u1_fft, self.psi_ffts[idx2])
                            s2_list.append(u2.mean(dim=[-2, -1]))

            if s2_list:
                s2 = torch.stack(s2_list, dim=1)
                coeffs.append(s2)

        return torch.cat(coeffs, dim=1)


class ScatteringFiltration(nn.Module):
    """
    Scattering Transform as a topofisher filtration component.

    Non-learnable: no trainable parameters, deterministic output.
    Optionally applies log1p to scattering coefficients to reduce dynamic range.

    Args:
        M: spatial height of input
        N: spatial width of input
        J: number of wavelet scales (default 7)
        L: number of orientations (default 8)
        max_order: maximum scattering order (default 2)
        log_scattering: if True, apply log1p to positive coefficients (default True)

    Example:
        >>> filt = ScatteringFiltration(M=512, N=512, J=7, L=8)
        >>> x = torch.randn(100, 512, 512)
        >>> out = filt(x)  # [[vec_0, ..., vec_99]], each vec is (1401,)
    """

    def __init__(
        self,
        M: int = 512,
        N: int = 512,
        J: int = 7,
        L: int = 8,
        max_order: int = 2,
        log_scattering: bool = True,
    ):
        super().__init__()
        self.scatter = ScatteringTransform2D(M, N, J, L, max_order)
        self.log_scattering = log_scattering
        self.output_dim = self.scatter.output_dim

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Compute scattering features.

        Args:
            x: (batch, H, W) or (H, W)

        Returns:
            List[List[Tensor]] in identity filtration format:
            [[vec_0, vec_1, ...]] where each vec has shape (output_dim,)
        """
        coeffs = self.scatter(x)

        if self.log_scattering:
            # S1 and S2 are positive by construction (spatial averages of |·|)
            # S0 can be negative, handle separately
            s0 = coeffs[:, :1]
            s_rest = coeffs[:, 1:]
            s_rest = torch.log1p(torch.clamp(s_rest, min=0))
            coeffs = torch.cat([s0, s_rest], dim=1)

        vectors = [coeffs[i] for i in range(coeffs.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters (always 0)."""
        return 0

    def __repr__(self):
        return (f"ScatteringFiltration(J={self.scatter.J}, L={self.scatter.L}, "
                f"max_order={self.scatter.max_order}, "
                f"output_dim={self.output_dim}, "
                f"log_scattering={self.log_scattering})")
