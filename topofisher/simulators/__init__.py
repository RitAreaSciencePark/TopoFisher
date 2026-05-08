"""Simulators for TopoFisher pipeline."""

# Compatibility patches for JAX/TFP versions
def _patch_jax_xla_compat():
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
        pass  # Non-fatal; ignore if JAX not available

_patch_jax_xla_compat()

from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import numpy as np
from tqdm import tqdm


class Simulator(ABC):
    """Base class for simulators."""

    @abstractmethod
    def generate_single(self, theta: torch.Tensor, seed: int) -> np.ndarray:
        """
        Generate a single sample at given parameters.

        Args:
            theta: Parameter values
            seed: Random seed for reproducibility

        Returns:
            Generated data sample
        """
        pass

    def generate(
        self,
        theta: torch.Tensor,
        n_samples: int,
        seed_start: int = 0,
        desc: str = None
    ) -> torch.Tensor:
        """
        Generate multiple samples at given parameters.

        Args:
            theta: Parameter values
            n_samples: Number of samples to generate
            seed_start: Starting seed value
            desc: Description for progress bar (if None, no progress bar)

        Returns:
            Tensor of shape (n_samples, ...)
        """
        samples = []
        iterator = range(n_samples)
        if desc is not None:
            iterator = tqdm(iterator, desc=desc, leave=False)
        for i in iterator:
            sample = self.generate_single(theta, seed=seed_start + i)
            samples.append(torch.from_numpy(sample).float())
        return torch.stack(samples)


# Import concrete implementations
from .grf import GRFSimulator
from .grf_fourier import GRFFourierSimulator
from .gaussian_vector import GaussianVectorSimulator
from .noisy_ring import NoisyRingSimulator
from .swiss_roll_3d import SwissRollSimulator_3d

# Optional: Lensing simulators (require JAX)
try:
    from .lensing_lognormal import LensingLogNormalSimulator
    _LENSING_AVAILABLE = True
except ImportError:
    _LENSING_AVAILABLE = False

try:
    from .lensing_LPT import LensingLPTSimulator
    _LENSING_LPT_AVAILABLE = True
except ImportError:
    _LENSING_LPT_AVAILABLE = False

__all__ = [
    'Simulator',
    'GRFSimulator',
    'GRFFourierSimulator',
    'GaussianVectorSimulator',
    'NoisyRingSimulator',
    'SwissRollSimulator_3d',
]

# Add optional simulators if available
if _LENSING_AVAILABLE:
    __all__.append('LensingLogNormalSimulator')
if _LENSING_LPT_AVAILABLE:
    __all__.append('LensingLPTSimulator')