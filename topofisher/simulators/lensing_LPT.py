"""
Weak Gravitational Lensing LPT (Lagrangian Perturbation Theory) Simulator.

Wrapper around sbi_lens Lpt_field model, adapted for TopoFisher's
deterministic sampling interface. Generates physically realistic weak
lensing convergence maps using first-order LPT + Born approximation.

Unlike the log-normal simulator, each sample requires a full N-body
forward pass (initial conditions → LPT displacement → CIC painting →
convergence via Born approximation), making it ~10-100x slower but
physically more accurate.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from functools import partial

from . import Simulator

_JAX_AVAILABLE = None


def _patch_jax_xla():
    """Compatibility shim: tensorflow_probability<=0.25 accesses removed JAX internals."""
    try:
        import jax
        import numpy as np
        import jax.interpreters.xla as jax_xla
        if not hasattr(jax_xla, 'pytype_aval_mappings'):
            jax_xla.pytype_aval_mappings = {
                np.ndarray: jax.core.pytype_aval_mappings[np.ndarray]
            }
        if not hasattr(jax_xla, 'canonicalize_dtype_handlers'):
            jax_xla.canonicalize_dtype_handlers = {}
    except Exception:
        pass  # Non-fatal; TFP import may still fail but with a clearer error


def _patch_jaxpm():
    """Compatibility shim: sbi_lens was written for jaxpm pre-0.0.2 API.

    The old pm_forces(positions, delta=...) signature became
    pm_forces(positions, mesh_shape, delta=...) in jaxpm 0.0.2+.
    We wrap it so the old positional-less call still works.
    """
    try:
        import jaxpm.pm as jpm
        import inspect
        sig = inspect.signature(jpm.pm_forces)
        params = list(sig.parameters.keys())
        # Needs patching only if 'mesh_shape' is positional (i.e., index 1)
        if len(params) >= 2 and params[1] == 'mesh_shape':
            _orig = jpm.pm_forces
            def _compat_pm_forces(positions, mesh_shape=None, delta=None, r_split=0):
                if mesh_shape is None and delta is not None:
                    mesh_shape = list(delta.shape)
                return _orig(positions, mesh_shape, delta=delta, r_split=r_split)
            jpm.pm_forces = _compat_pm_forces
            # Also patch the reference inside Lpt_field module if already imported
            try:
                import sbi_lens.simulator.Lpt_field as _lf
                if hasattr(_lf, 'pm_forces'):
                    _lf.pm_forces = _compat_pm_forces
            except Exception:
                pass
    except Exception:
        pass


def _check_jax():
    global _JAX_AVAILABLE
    if _JAX_AVAILABLE is None:
        try:
            _patch_jax_xla()
            _patch_jaxpm()
            import jax
            import jax.numpy as jnp
            import jax_cosmo as jc
            _JAX_AVAILABLE = True
        except ImportError as e:
            _JAX_AVAILABLE = False
            raise ImportError(
                "JAX dependencies required for LensingLPTSimulator. "
                "Install with: pip install jax jaxlib jax_cosmo jaxpm numpyro"
            ) from e
    return _JAX_AVAILABLE


class LensingLPTSimulator(Simulator):
    """
    LPT weak lensing convergence map simulator.

    Wraps sbi_lens Lpt_field at fixed cosmology (omega_m, sigma_8) with
    other parameters fixed at fiducial values. Each call to generate()
    produces independent maps by drawing independent initial conditions.

    Parameters
    ----------
    N : int
        Output map pixels per side (default: 512)
    map_size : float
        Angular map size in degrees (default: 10.0)
    box_size : list of float
        LPT box physical size [Lx, Ly, Lz] in Mpc/h (default: [700, 700, 4000])
    box_shape : list of int
        LPT mesh shape [Nx, Ny, Nz]. box_shape[0] must be divisible by N.
        (default: [N, N, 128])
    bin_index : int
        Tomographic bin index to return when all_bins=False (default: 0)
    nbins : int
        Total number of tomographic bins (default: 5)
    gal_per_arcmin2 : float
        Galaxy number density per arcmin^2 (default: 27)
    sigma_e : float
        Shape noise dispersion (default: 0.26)
    a, b, z0 : float
        Smail redshift distribution parameters (default: 2.0, 0.68, 0.11)
    with_noise : bool
        Include shape noise (default: True)
    all_bins : bool
        Return all tomographic bins; output shape (n_samples, nbins, N, N).
        (default: False → single bin, shape (n_samples, N, N))
    omega_b, h_0, n_s, w_0 : float
        Fixed cosmological parameters
    sbi_lens_path : str, optional
        Add sbi_lens to sys.path if not already installed
    device : str
        Output tensor device ('cpu' or 'cuda')
    """

    def __init__(
        self,
        N: int = 512,
        map_size: float = 10.0,
        box_size: Optional[list] = None,
        box_shape: Optional[list] = None,
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
        device: str = "cpu",
    ):
        _check_jax()

        self.N = N
        self.map_size = map_size
        # box_shape[0] must be divisible by N for the downsampling in make_full_field_model
        self.box_shape = box_shape if box_shape is not None else [N, N, 128]
        self.box_size = box_size if box_size is not None else [700.0, 700.0, 4000.0]
        self.bin_index = bin_index
        self.nbins = nbins
        self.gal_per_arcmin2 = gal_per_arcmin2
        self.sigma_e = sigma_e
        self.a = a
        self.b = b
        self.z0 = z0
        self.with_noise = with_noise
        self.all_bins = all_bins

        self.omega_b = omega_b
        self.h_0 = h_0
        self.n_s = n_s
        self.w_0 = w_0
        self.device = device

        if sbi_lens_path is not None:
            import sys
            if sbi_lens_path not in sys.path:
                sys.path.insert(0, sbi_lens_path)

        # Cached JIT model for fixed cosmology
        self._cached_theta = None
        self._jit_fn = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_jit_fn'] = None
        state['_cached_theta'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _make_jit_model(self, omega_m: float, sigma_8: float):
        """Build and return a JIT-compiled function that draws one map sample."""
        _patch_jax_xla()  # must precede sbi_lens imports (TFP compat)
        _patch_jaxpm()    # must precede Lpt_field import (pm_forces compat)
        import jax
        from numpyro.handlers import condition
        from numpyro.handlers import seed as numpyro_seed
        from numpyro.handlers import trace
        from sbi_lens.simulator.Lpt_field import lensingLpt

        omega_c = omega_m - self.omega_b

        base_model = partial(
            lensingLpt,
            self.N,
            self.map_size,
            self.box_size,
            self.box_shape,
            self.gal_per_arcmin2,
            self.sigma_e,
            self.nbins,
            self.a,
            self.b,
            self.z0,
            self.with_noise,
        )

        cond_model = condition(
            base_model,
            {
                "omega_c": float(omega_c),
                "omega_b": float(self.omega_b),
                "sigma_8": float(sigma_8),
                "h_0": float(self.h_0),
                "n_s": float(self.n_s),
                "w_0": float(self.w_0),
            },
        )

        @jax.jit
        def run_single(key):
            seeded = numpyro_seed(cond_model, key)
            model_trace = trace(seeded).get_trace()
            return model_trace["y"]["value"]  # shape: (N, N, nbins)

        return run_single

    def _get_jit_fn(self, omega_m: float, sigma_8: float):
        """Return cached JIT fn, rebuilding if cosmology changed."""
        theta_key = (omega_m, sigma_8)
        if self._cached_theta != theta_key:
            print(f"  [LPT] Compiling forward model for Omega_m={omega_m:.4f}, "
                  f"sigma_8={sigma_8:.4f} ...", flush=True)
            import jax
            self._jit_fn = self._make_jit_model(omega_m, sigma_8)
            # Warm-up: forces JIT compilation before timing the loop
            dummy_key = jax.random.PRNGKey(999999)
            _ = self._jit_fn(dummy_key).block_until_ready()
            self._cached_theta = theta_key
            print("  [LPT] Compilation done.", flush=True)
        return self._jit_fn

    def generate_single(self, theta: torch.Tensor, seed: int) -> np.ndarray:
        """
        Generate a single convergence map.

        Parameters
        ----------
        theta : torch.Tensor
            [Omega_m, sigma_8]
        seed : int
            Random seed (controls initial conditions + noise)

        Returns
        -------
        np.ndarray of shape (N, N) or (nbins, N, N) if all_bins
        """
        import jax

        omega_m = float(theta[0])
        sigma_8 = float(theta[1])

        fn = self._get_jit_fn(omega_m, sigma_8)
        key = jax.random.PRNGKey(seed)
        field = fn(key)  # (N, N, nbins)

        field_np = np.array(field)
        if self.all_bins:
            return field_np.transpose(2, 0, 1)  # (nbins, N, N)
        return field_np[:, :, self.bin_index]    # (N, N)

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed: Optional[int] = None,
        desc: str = None,
    ) -> torch.Tensor:
        """
        Generate multiple convergence maps at fixed cosmology.

        Parameters
        ----------
        theta : torch.Tensor
            [Omega_m, sigma_8]
        n_samples : int
        seed : int, optional
            Starting seed; sample i uses PRNGKey(seed + i)
        desc : str, optional
            tqdm progress-bar description

        Returns
        -------
        torch.Tensor of shape (n_samples, N, N) or (n_samples, nbins, N, N)
        """
        import jax

        omega_m = float(theta[0])
        sigma_8 = float(theta[1])

        fn = self._get_jit_fn(omega_m, sigma_8)
        seed_start = seed if seed is not None else 0

        if desc is not None:
            from tqdm import tqdm
            iterator = tqdm(range(n_samples), desc=desc, leave=False)
        else:
            iterator = range(n_samples)

        samples = []
        for i in iterator:
            key = jax.random.PRNGKey(seed_start + i)
            field = fn(key)              # (N, N, nbins) JAX array
            field_np = np.array(field)   # copy to CPU numpy
            if self.all_bins:
                samples.append(field_np.transpose(2, 0, 1))  # (nbins, N, N)
            else:
                samples.append(field_np[:, :, self.bin_index])  # (N, N)

        result = np.stack(samples)
        return torch.from_numpy(result).float().to(self.device)

    def get_config(self) -> dict:
        return {
            "N": self.N,
            "map_size": self.map_size,
            "box_size": self.box_size,
            "box_shape": self.box_shape,
            "bin_index": self.bin_index,
            "nbins": self.nbins,
            "gal_per_arcmin2": self.gal_per_arcmin2,
            "sigma_e": self.sigma_e,
            "a": self.a,
            "b": self.b,
            "z0": self.z0,
            "with_noise": self.with_noise,
            "all_bins": self.all_bins,
            "omega_b": self.omega_b,
            "h_0": self.h_0,
            "n_s": self.n_s,
            "w_0": self.w_0,
        }

    def __repr__(self) -> str:
        return (
            f"LensingLPTSimulator(N={self.N}, map_size={self.map_size}, "
            f"box_shape={self.box_shape}, nbins={self.nbins}, "
            f"all_bins={self.all_bins}, with_noise={self.with_noise}, "
            f"device='{self.device}')"
        )
