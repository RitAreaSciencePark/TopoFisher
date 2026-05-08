"""
Weak Gravitational Lensing Log-Normal Simulator.

This is a simplified wrapper around the sbi_lens LogNormal simulator,
adapted for TopoFisher's deterministic sampling interface.

The simulator generates log-normal weak lensing convergence maps
with configurable cosmological parameters.

Optimized with JAX JIT compilation and caching of cosmology-dependent
computations for efficient batch generation.
"""
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import torch
from functools import partial

from . import Simulator

# JAX imports (will be imported lazily to allow optional dependency)
_JAX_AVAILABLE = None


def _check_jax():
    """Check if JAX and required dependencies are available."""
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            import jax
            import jax.numpy as jnp
            import jax_cosmo as jc
            _JAX_AVAILABLE = True
        except ImportError as e:
            _JAX_AVAILABLE = False
            raise ImportError(
                "JAX dependencies required for LensingLogNormalSimulator. "
                "Install with: pip install jax jax_cosmo"
            ) from e
    return _JAX_AVAILABLE


class LensingLogNormalSimulator(Simulator):
    """
    Log-normal weak lensing convergence map simulator.
    
    Wrapper around sbi_lens simulator adapted for TopoFisher.
    Generates lensing convergence maps parameterized by cosmological parameters.
    
    The default parameterization uses two varying parameters:
    - Omega_m (total matter density)
    - sigma_8 (amplitude of matter fluctuations)
    
    Other cosmological parameters are fixed at fiducial values.
    
    Performance: The simulator caches cosmology-dependent computations (power
    spectrum, covariance matrices) and uses JAX JIT compilation for the random
    field generation, making batch generation very efficient.
    
    Parameters
    ----------
    N : int
        Number of pixels on each side of the map (default: 128)
    map_size : float
        Angular size of the map in degrees (default: 5.0)
    bin_index : int
        Which redshift bin to use (0-indexed, default: 0)
    nbins : int
        Total number of redshift bins (default: 5)
    gal_per_arcmin2 : float
        Galaxy number density per arcmin^2 (default: 27)
    sigma_e : float
        Shape noise dispersion (default: 0.26)
    a : float
        Redshift distribution parameter (default: 2.0)
    b : float
        Redshift distribution parameter (default: 0.68)
    z0 : float
        Redshift distribution parameter (default: 0.11)
    with_noise : bool
        Whether to include shape noise (default: True)
    omega_b : float
        Fixed baryon density (default: 0.0492)
    h_0 : float
        Fixed Hubble parameter (default: 0.6727)
    n_s : float
        Fixed spectral index (default: 0.9645)
    w_0 : float
        Fixed dark energy equation of state (default: -1.0)
    all_bins : bool
        If True, return all tomographic bins instead of a single bin.
        Output shape becomes (n_samples, nbins, N, N). (default: False)
    sbi_lens_path : str, optional
        Path to sbi_lens package (if not in PYTHONPATH)
    device : str
        Device for output tensors ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        N: int = 128,
        map_size: float = 5.0,
        bin_index: int = 0,
        nbins: int = 5,
        gal_per_arcmin2: float = 27.0,
        sigma_e: float = 0.26,
        a: float = 2.0,
        b: float = 0.68,
        z0: float = 0.11,
        with_noise: bool = True,
        all_bins: bool = False,
        omega_b: float = 0.0492,
        h_0: float = 0.6727,
        n_s: float = 0.9645,
        w_0: float = -1.0,
        sbi_lens_path: Optional[str] = None,
        device: str = "cpu"
    ):
        _check_jax()
        
        self.N = N
        self.map_size = map_size
        self.bin_index = bin_index
        self.nbins = nbins
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e
        self.a = a
        self.b = b
        self.z0 = z0
        self.with_noise = with_noise
        self.all_bins = all_bins
        
        # Fixed cosmological parameters
        self.omega_b = omega_b
        self.h_0 = h_0
        self.n_s = n_s
        self.w_0 = w_0
        
        self.device = device
        
        # Add sbi_lens to path if specified
        if sbi_lens_path is not None:
            import sys
            if sbi_lens_path not in sys.path:
                sys.path.insert(0, sbi_lens_path)
        
        # Load lognormal shift parameters
        self._load_shift_params()
        
        # Pre-compute static parameters
        self._setup_static_params()
        
        # Cache for cosmology-dependent computations
        self._cached_theta = None
        self._cached_L = None  # Cholesky-like matrix
        self._cached_shift = None
        self._cached_noise_std = None
        
        # JIT-compiled generation function
        self._jit_generate_field = None
        self._jit_add_noise = None
    
    def __getstate__(self):
        """Prepare state for pickling (exclude JIT-compiled functions)."""
        state = self.__dict__.copy()
        # Remove JIT-compiled functions as they can't be pickled
        state['_jit_generate_field'] = None
        state['_jit_add_noise'] = None
        # Also clear the cache since JAX arrays may have issues
        state['_cached_theta'] = None
        state['_cached_L'] = None
        state['_cached_shift'] = None
        state['_cached_noise_std'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state from pickling."""
        self.__dict__.update(state)
        # JIT functions will be re-created on first use
    
    def _load_shift_params(self):
        """Load the lognormal shift interpolation table."""
        # Try to find sbi_lens data directory
        try:
            from sbi_lens.simulator.LogNormal_field import DATA_DIR
            shift_file = DATA_DIR / "lognormal_shifts_LSSTY10_om_s8_w_bin.npy"
        except ImportError:
            # Fallback: look in common locations
            possible_paths = [
                Path("/u/area/mbiagetti/Codes/sbi_lens/sbi_lens/data/lognormal_shifts_LSSTY10_om_s8_w_bin.npy"),
                Path.home() / "sbi_lens/sbi_lens/data/lognormal_shifts_LSSTY10_om_s8_w_bin.npy",
            ]
            shift_file = None
            for p in possible_paths:
                if p.exists():
                    shift_file = p
                    break
            if shift_file is None:
                raise FileNotFoundError(
                    "Could not find lognormal_shifts_LSSTY10_om_s8_w_bin.npy. "
                    "Please ensure sbi_lens is properly installed or specify sbi_lens_path."
                )
        
        self.lognormal_shifts_params = np.load(shift_file)
    
    def _setup_static_params(self):
        """Pre-compute parameters that don't depend on cosmology."""
        import jax.numpy as jnp
        
        # Field parameters
        self.pix_area = (self.map_size * 60 / self.N) ** 2  # arcmin^2 per pixel
        self.map_size_rad = self.map_size / 180 * jnp.pi
        
        # Ell values for power spectrum
        self.ell_tab = 2 * jnp.pi * jnp.abs(
            jnp.fft.fftfreq(2 * self.N, d=self.map_size_rad / (2 * self.N))
        )
        
        # k-space grid (static)
        k = 2 * jnp.pi * jnp.fft.fftfreq(self.N, d=self.map_size_rad / self.N)
        kx, ky = jnp.meshgrid(k, k)
        self._k_mag = jnp.sqrt(kx ** 2 + ky ** 2)
    
    def _compute_cosmology_cache(self, omega_m: float, sigma_8: float):
        """
        Compute and cache all cosmology-dependent quantities.
        
        This is the expensive part that we want to do only once per theta.
        """
        import jax
        import jax.numpy as jnp
        import jax_cosmo as jc
        from jax.scipy.ndimage import map_coordinates
        
        # Compute omega_c from omega_m
        omega_c = omega_m - self.omega_b
        
        # Create cosmology object
        cosmo = jc.Planck15(
            Omega_c=omega_c,
            Omega_b=self.omega_b,
            h=self.h_0,
            n_s=self.n_s,
            sigma8=sigma_8,
            w0=self.w_0
        )
        
        # Compute shift parameters via interpolation
        params = self.lognormal_shifts_params
        theta_vec = jnp.array([omega_m, sigma_8, self.w_0])
        ntheta = len(params.shape[:-2])
        
        shift = []
        for i in range(self.nbins):
            coords = jnp.stack([
                (theta_vec[j] - params[..., j].min()) / 
                (params[..., j].max() - params[..., j].min()) * 
                params.shape[j] - 0.5
                for j in range(ntheta)
            ], axis=0).reshape([ntheta, -1])
            
            shift_val = map_coordinates(
                params[..., i, -1],
                coords,
                order=1,
                mode="nearest"
            ).squeeze()
            shift.append(shift_val)
        
        shift = jnp.stack(shift)
        
        # Fill shift array (products for cross-correlations)
        idx = np.mask_indices(len(shift), np.triu)
        shift_array = jnp.outer(shift, shift)[idx]
        
        # Define redshift distribution
        nz = jc.redshift.smail_nz(
            self.a, self.b, self.z0, 
            gals_per_arcmin2=self.gal_per_arcmin2
        )
        
        # Subdivide into bins
        try:
            from sbi_lens.simulator.redshift import subdivide
            nz_bins = subdivide(nz, nbins=self.nbins, zphot_sigma=0.05)
        except ImportError:
            nz_bins = [nz]
        
        tracer = jc.probes.WeakLensing(nz_bins, sigma_e=self.sigma_e)
        
        # Calculate power spectrum C_ell
        cell_tab = jc.angular_cl.angular_cl(cosmo, self.ell_tab, [tracer])
        
        # Build power maps for each auto/cross correlation
        power_maps = []
        for cl, l_shift in zip(cell_tab, shift_array):
            # Interpolate C_ell to k-grid
            ps_map = jc.scipy.interpolate.interp(
                self._k_mag.flatten(), self.ell_tab, cl
            ).reshape(self._k_mag.shape)
            ps_map = ps_map.at[0, 0].set(0.0)
            power_map = ps_map * (self.N / self.map_size_rad) ** 2
            
            # Transform to lognormal power
            power_ln = jnp.fft.ifft2(power_map).real
            power_ln = jnp.log(1 + power_ln / l_shift)
            power_ln = jnp.abs(jnp.fft.fft2(power_ln))
            power_ln = power_ln.at[0, 0].set(0.0)
            power_maps.append(power_ln)
        
        power = jnp.stack(power_maps, axis=-1)
        
        # Build covariance matrices at each k-point
        n_cross = len(cell_tab)
        
        def fill_cov_mat(m):
            idx_tri = np.triu_indices(self.nbins)
            cov_mat = jnp.zeros((self.nbins, self.nbins)).at[idx_tri].set(m).T.at[idx_tri].set(m)
            return cov_mat
        
        cov_mat = jax.vmap(fill_cov_mat)(power.reshape(-1, n_cross))
        
        # Eigendecomposition for "square root" of covariance
        eigval, A = jnp.linalg.eigh(cov_mat)
        L = jax.vmap(
            lambda M, v: M.dot(jnp.diag(jnp.sqrt(jnp.clip(v, a_min=0))).dot(M.T))
        )(A, eigval)
        L = L.reshape([self.N, self.N, self.nbins, self.nbins])
        L = L.at[0, 0].set(jnp.zeros((self.nbins, self.nbins)))
        L = L.transpose([2, 3, 0, 1])  # (nbins, nbins, N, N)
        
        # Noise standard deviation
        noise_std = jnp.array([
            self.sigma_e / jnp.sqrt(nz_bin.gals_per_arcmin2 * self.pix_area)
            for nz_bin in nz_bins
        ])
        
        return L, shift, noise_std
    
    def _get_cached_or_compute(self, omega_m: float, sigma_8: float):
        """Get cached values or recompute if theta changed."""
        theta_key = (omega_m, sigma_8)
        
        if self._cached_theta != theta_key:
            # Recompute cache
            L, shift, noise_std = self._compute_cosmology_cache(omega_m, sigma_8)
            self._cached_theta = theta_key
            self._cached_L = L
            self._cached_shift = shift
            self._cached_noise_std = noise_std
        
        return self._cached_L, self._cached_shift, self._cached_noise_std
    
    def _setup_jit_generate(self):
        """Create JIT-compiled field generation function."""
        import jax
        import jax.numpy as jnp
        
        nbins = self.nbins
        N = self.N
        
        @jax.jit
        def generate_field(L, shift, key):
            """JIT-compiled random field generation."""
            z = jax.random.normal(key, shape=(nbins, N, N))
            field = jnp.fft.fft2(z) * L
            field = jnp.fft.ifft2(jnp.sum(field, axis=1)).real
            field = jnp.einsum(
                "i, ijk -> ijk",
                shift,
                jnp.exp(field - jnp.var(field, axis=(1, 2), keepdims=True) / 2) - 1
            )
            return field
        
        @jax.jit
        def add_noise(field, noise_std, key):
            """JIT-compiled noise addition."""
            noise = noise_std[:, None, None] * jax.random.normal(
                key, shape=(nbins, N, N)
            )
            return field + noise
        
        self._jit_generate_field = generate_field
        self._jit_add_noise = add_noise
    
    def generate_single(self, theta: torch.Tensor, seed: int) -> np.ndarray:
        """
        Generate a single lensing convergence map.
        
        Parameters
        ----------
        theta : torch.Tensor
            Cosmological parameters [Omega_m, sigma_8]
            - Omega_m: total matter density (omega_c + omega_b)
            - sigma_8: amplitude of matter fluctuations
        seed : int
            Random seed for reproducibility
        
        Returns
        -------
        convergence_map : np.ndarray
            Lensing convergence map of shape (N, N)
        """
        import jax
        import jax.numpy as jnp
        
        # Extract parameters
        omega_m = float(theta[0])
        sigma_8 = float(theta[1])
        
        # Get cached cosmology-dependent quantities (or compute if needed)
        L, shift, noise_std = self._get_cached_or_compute(omega_m, sigma_8)
        
        # Setup JIT functions if not done
        if self._jit_generate_field is None:
            self._setup_jit_generate()
        
        # Generate random field (fast, JIT-compiled)
        key = jax.random.PRNGKey(seed)
        field = self._jit_generate_field(L, shift, key)
        
        # Add noise if requested
        if self.with_noise:
            key, subkey = jax.random.split(key)
            field = self._jit_add_noise(field, noise_std, subkey)
        
        # Extract the requested bin(s)
        if self.all_bins:
            convergence_map = np.array(field)  # (nbins, N, N)
        else:
            convergence_map = np.array(field[self.bin_index])  # (N, N)
        
        return convergence_map
    
    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        desc: str = None
    ) -> torch.Tensor:
        """
        Generate multiple lensing convergence maps.
        
        Optimized for batch generation at the same theta value.
        
        Parameters
        ----------
        theta : torch.Tensor
            Cosmological parameters [Omega_m, sigma_8]
        n_samples : int
            Number of maps to generate
        seed : int, optional
            Starting random seed
        desc : str, optional
            Description for progress bar
        
        Returns
        -------
        maps : torch.Tensor
            Tensor of shape (n_samples, N, N)
        """
        import jax
        import jax.numpy as jnp
        
        # Extract parameters
        omega_m = float(theta[0])
        sigma_8 = float(theta[1])
        
        # Get cached cosmology-dependent quantities (computed once!)
        L, shift, noise_std = self._get_cached_or_compute(omega_m, sigma_8)
        
        # Setup JIT functions if not done
        if self._jit_generate_field is None:
            self._setup_jit_generate()
        
        # Generate all samples efficiently
        samples = []
        seed_start = seed if seed is not None else 0
        
        # Use tqdm if desc provided
        if desc is not None:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc=desc, leave=False)
        else:
            iterator = range(n_samples)
        
        for i in iterator:
            key = jax.random.PRNGKey(seed_start + i)
            field = self._jit_generate_field(L, shift, key)
            
            if self.with_noise:
                key, subkey = jax.random.split(key)
                field = self._jit_add_noise(field, noise_std, subkey)
            
            if self.all_bins:
                samples.append(field)  # (nbins, N, N)
            else:
                samples.append(field[self.bin_index])  # (N, N)
        
        # Stack and convert to torch
        # all_bins: (n_samples, nbins, N, N), single bin: (n_samples, N, N)
        result = np.stack([np.array(s) for s in samples])
        return torch.from_numpy(result).float().to(self.device)
    
    def __repr__(self) -> str:
        return (
            f"LensingLogNormalSimulator(N={self.N}, map_size={self.map_size}, "
            f"bin_index={self.bin_index}, nbins={self.nbins}, all_bins={self.all_bins}, "
            f"with_noise={self.with_noise}, device='{self.device}')"
        )
    
    def get_config(self) -> dict:
        """Return configuration dictionary for serialization."""
        return {
            'N': self.N,
            'map_size': self.map_size,
            'bin_index': self.bin_index,
            'nbins': self.nbins,
            'gal_per_arcmin2': self.gal_per_arcmin2,
            'sigma_e': self.sigma_e,
            'a': self.a,
            'b': self.b,
            'z0': self.z0,
            'with_noise': self.with_noise,
            'all_bins': self.all_bins,
            'omega_b': self.omega_b,
            'h_0': self.h_0,
            'n_s': self.n_s,
            'w_0': self.w_0,
        }
