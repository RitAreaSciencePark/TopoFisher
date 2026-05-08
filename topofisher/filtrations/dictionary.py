"""
Dictionary-based parametric filtration for cubical persistence.

Instead of feeding raw κ to cubical persistence, we pre-compute a dictionary
of K geometric feature maps {φ_i(x,y)} at each pixel, then form the
filtration function as a learned linear combination:

    f(x,y) = Σ_i  α_i · φ̃_i(x,y)

where φ̃_i is standardized (zero-mean, unit-variance) and α_i are learnable
mixing weights.  Persistence is computed on f, so the α control *what
topology is measured*.

The search space is only K ≈ 10 real numbers, so derivative-free optimization
(CMA-ES, Bayesian opt) is practical — no backprop through persistence needed.

Feature dictionary (default 10 features):
    identity      : κ                     (standard sublevel filtration)
    gradient_mag  : |∇κ|                  (edges, filaments)
    laplacian     : ∇²κ                   (curvature: peaks vs voids)
    hessian_det   : det H(κ)              (peak/void classifier)
    squared       : κ²                    (non-Gaussianity probe)
    smooth_2      : G_σ=2 * κ             (large-scale structure)
    smooth_4      : G_σ=4 * κ             (larger-scale structure)
    highpass_2    : κ - G_σ=2 * κ         (small-scale detail)
    phase_only    : Re[F⁻¹(F[κ]/|F[κ]|)] (all non-Gaussian info)
    local_std     : σ_local(κ, r=3)       (local variability)
"""
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Feature computation helpers (all operate on (B, H, W) tensors)
# ═══════════════════════════════════════════════════════════════════════

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure input is (B, H, W)."""
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _conv2d_fixed(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a fixed 2D convolution kernel to (B, H, W) input with replicate padding."""
    B, H, W = x.shape
    kh, kw = kernel.shape[-2], kernel.shape[-1]
    ph, pw = kh // 2, kw // 2
    x4d = x.unsqueeze(1)                       # (B, 1, H, W)
    x_pad = F.pad(x4d, (pw, pw, ph, ph), mode='replicate')
    k = kernel.to(x.device, x.dtype)
    if k.ndim == 2:
        k = k.reshape(1, 1, kh, kw)
    out = F.conv2d(x_pad, k)                    # (B, 1, H, W)
    return out.squeeze(1)                        # (B, H, W)


def _gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """FFT-based Gaussian smoothing (efficient for any σ)."""
    B, H, W = x.shape
    ky = torch.fft.fftfreq(H, device=x.device, dtype=x.dtype)
    kx = torch.fft.fftfreq(W, device=x.device, dtype=x.dtype)
    ky, kx = torch.meshgrid(ky, kx, indexing='ij')
    k2 = kx ** 2 + ky ** 2
    gauss_ft = torch.exp(-2 * (np.pi * sigma) ** 2 * k2)  # Gaussian in freq domain
    x_ft = torch.fft.fft2(x)
    return torch.fft.ifft2(x_ft * gauss_ft).real


# ── Individual feature functions ──────────────────────────────────────

def feat_identity(x: torch.Tensor) -> torch.Tensor:
    """Raw field κ."""
    return x


def feat_gradient_mag(x: torch.Tensor) -> torch.Tensor:
    """|∇κ| via central differences."""
    dx_kernel = torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                             dtype=torch.float32) / 2.0
    dy_kernel = torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]],
                             dtype=torch.float32) / 2.0
    gx = _conv2d_fixed(x, dx_kernel)
    gy = _conv2d_fixed(x, dy_kernel)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def feat_laplacian(x: torch.Tensor) -> torch.Tensor:
    """∇²κ (discrete Laplacian)."""
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                          dtype=torch.float32)
    return _conv2d_fixed(x, kernel)


def feat_hessian_det(x: torch.Tensor) -> torch.Tensor:
    """det(H) = ∂²κ/∂x² · ∂²κ/∂y² − (∂²κ/∂x∂y)²."""
    dxx_kernel = torch.tensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]],
                              dtype=torch.float32)
    dyy_kernel = torch.tensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]],
                              dtype=torch.float32)
    dxy_kernel = torch.tensor([[1, 0, -1], [0, 0, 0], [-1, 0, 1]],
                              dtype=torch.float32) / 4.0
    fxx = _conv2d_fixed(x, dxx_kernel)
    fyy = _conv2d_fixed(x, dyy_kernel)
    fxy = _conv2d_fixed(x, dxy_kernel)
    return fxx * fyy - fxy ** 2


def feat_squared(x: torch.Tensor) -> torch.Tensor:
    """κ² (non-Gaussianity probe)."""
    return x ** 2


def feat_smooth(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Gaussian-smoothed field at scale σ."""
    return _gaussian_smooth(x, sigma)


def feat_highpass(x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """High-pass: κ − G_σ * κ (small-scale detail)."""
    return x - _gaussian_smooth(x, sigma)


def feat_phase_only(x: torch.Tensor) -> torch.Tensor:
    """Phase-only reconstruction: Re[F⁻¹(F[κ]/|F[κ]|)].
    
    Contains ALL non-Gaussian information (the power spectrum = amplitudes,
    so phases encode everything beyond C_ℓ).
    """
    ft = torch.fft.fft2(x)
    amp = ft.abs().clamp(min=1e-10)
    phase = ft / amp
    return torch.fft.ifft2(phase).real


def feat_local_std(x: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Local standard deviation in a (2r+1)×(2r+1) window."""
    size = 2 * radius + 1
    # Box-filter kernel for local mean and local mean-of-squares
    kernel = torch.ones(1, 1, size, size, dtype=x.dtype, device=x.device) / (size * size)
    B, H, W = x.shape
    x4d = x.unsqueeze(1)
    x_pad = F.pad(x4d, (radius, radius, radius, radius), mode='replicate')
    local_mean = F.conv2d(x_pad, kernel).squeeze(1)
    local_mean_sq = F.conv2d(F.pad(x4d ** 2, (radius, radius, radius, radius),
                                    mode='replicate'), kernel).squeeze(1)
    var = (local_mean_sq - local_mean ** 2).clamp(min=0)
    return torch.sqrt(var + 1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Default feature registry
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_FEATURES = [
    'identity',
    'gradient_mag',
    'laplacian',
    'hessian_det',
    'squared',
    'smooth_2',
    'smooth_4',
    'highpass_2',
    'phase_only',
    'local_std',
]

# Map name → callable(x) → (B, H, W)
FEATURE_REGISTRY = {
    'identity':     feat_identity,
    'gradient_mag': feat_gradient_mag,
    'laplacian':    feat_laplacian,
    'hessian_det':  feat_hessian_det,
    'squared':      feat_squared,
    'smooth_2':     lambda x: feat_smooth(x, sigma=2.0),
    'smooth_4':     lambda x: feat_smooth(x, sigma=4.0),
    'smooth_8':     lambda x: feat_smooth(x, sigma=8.0),
    'highpass_2':   lambda x: feat_highpass(x, sigma=2.0),
    'highpass_4':   lambda x: feat_highpass(x, sigma=4.0),
    'phase_only':   feat_phase_only,
    'local_std':    feat_local_std,
    # Fourier band features (parameterized by N, added dynamically)
}


# ═══════════════════════════════════════════════════════════════════════
# DictionaryFiltration Module
# ═══════════════════════════════════════════════════════════════════════

class DictionaryFiltration(nn.Module):
    """
    Parametric dictionary filtration for cubical persistence.

    Pre-computes K geometric feature maps per pixel, then linearly combines
    them with weights α before feeding to cubical persistence.

    The weights α are the ONLY learnable parameters (~10 numbers),
    enabling efficient derivative-free optimization over the filtration.

    Supports:
        - 2D input (H, W) — single sample
        - 3D input (B, H, W) — batch of single-bin maps
        - 4D input (B, nbins, H, W) — batch of multi-bin maps
          (features computed per bin, α shared across bins)

    Example:
        >>> filt = DictionaryFiltration(feature_names=['identity', 'gradient_mag', 'phase_only'])
        >>> # Set weights (e.g., from CMA-ES)
        >>> filt.set_alpha([0.7, 0.2, 0.1])
        >>> # Compute persistence on mixed field
        >>> diagrams = filt(field_batch)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        homology_dimensions: List[int] = [0, 1],
        n_jobs: int = 1,
        show_progress: bool = False,
        superlevel: bool = False,
    ):
        super().__init__()

        self.feature_names = feature_names or DEFAULT_FEATURES
        self.K = len(self.feature_names)

        # Validate feature names
        for name in self.feature_names:
            if name not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature: {name}. Available: {list(FEATURE_REGISTRY.keys())}")

        # Mixing weights (initialized to identity-only: α = [1, 0, 0, ...])
        alpha_init = torch.zeros(self.K)
        alpha_init[0] = 1.0  # Start from standard sublevel filtration
        self.register_buffer('alpha', alpha_init)

        # Standardization stats (computed via fit())
        self.register_buffer('feat_mean', torch.zeros(self.K))
        self.register_buffer('feat_std', torch.ones(self.K))
        self._is_fitted = False

        # Store cubical params for external use
        self.homology_dimensions = homology_dimensions
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.superlevel = superlevel

    def set_alpha(self, alpha):
        """Set mixing weights."""
        if isinstance(alpha, (list, np.ndarray)):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha.copy_(alpha.to(self.alpha.device))

    def get_alpha_dict(self) -> Dict[str, float]:
        """Return {feature_name: weight} dictionary."""
        return {name: self.alpha[i].item() for i, name in enumerate(self.feature_names)}

    def compute_feature_maps(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute all K feature maps for input batch.

        Args:
            X: (B, H, W) input fields

        Returns:
            (B, K, H, W) tensor of feature maps
        """
        X = _ensure_3d(X)
        B, H, W = X.shape

        maps = []
        for name in self.feature_names:
            fn = FEATURE_REGISTRY[name]
            feat = fn(X)  # (B, H, W)
            maps.append(feat)

        return torch.stack(maps, dim=1)  # (B, K, H, W)

    def fit(self, X_fiducial: torch.Tensor):
        """
        Compute standardization stats from fiducial data.

        Args:
            X_fiducial: (B, H, W) or (B, nbins, H, W) fiducial field samples
        """
        # For multi-bin, compute features for each bin and pool stats
        if X_fiducial.ndim == 4:
            B, nbins, H, W = X_fiducial.shape
            X_flat = X_fiducial.reshape(B * nbins, H, W)
        else:
            X_flat = _ensure_3d(X_fiducial)

        with torch.no_grad():
            feat_maps = self.compute_feature_maps(X_flat)  # (B*nbins, K, H, W)
            # Global mean and std per feature (across all samples and pixels)
            for k in range(self.K):
                fk = feat_maps[:, k]  # (B*nbins, H, W)
                self.feat_mean[k] = fk.mean()
                self.feat_std[k] = fk.std().clamp(min=1e-8)

        self._is_fitted = True

    def mix(self, X: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the mixed filtration field f = Σ α_i φ̃_i.

        Args:
            X: (B, H, W) input fields
            alpha: optional override weights

        Returns:
            (B, H, W) mixed field
        """
        X = _ensure_3d(X)
        alpha = alpha if alpha is not None else self.alpha

        feat_maps = self.compute_feature_maps(X)  # (B, K, H, W)

        # Standardize each feature
        mean = self.feat_mean.view(1, self.K, 1, 1)
        std = self.feat_std.view(1, self.K, 1, 1)
        feat_maps = (feat_maps - mean) / std

        # Linear combination: α · φ̃ → (B, H, W)
        w = alpha.view(1, self.K, 1, 1).to(feat_maps.device, feat_maps.dtype)
        mixed = (w * feat_maps).sum(dim=1)  # (B, H, W)

        return mixed

    def mix_multibin(self, X: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Mix features for multi-bin input, with α shared across bins.

        Args:
            X: (B, nbins, H, W) multi-bin input

        Returns:
            (B, nbins, H, W) mixed fields (one per bin)
        """
        B, nbins, H, W = X.shape
        alpha = alpha if alpha is not None else self.alpha

        # Process each bin with the same α
        result = torch.zeros_like(X)
        for b in range(nbins):
            result[:, b] = self.mix(X[:, b], alpha)

        return result

    def forward(self, X: torch.Tensor, alpha: Optional[torch.Tensor] = None):
        """
        Compute mixed field from dictionary features.

        Does NOT compute persistence — returns the mixed field to be passed
        to CubicalLayer externally. This keeps the module composable.

        Args:
            X: (H, W), (B, H, W), or (B, nbins, H, W) input
            alpha: optional override weights

        Returns:
            Mixed field with same spatial dims as input
        """
        if X.ndim == 4:
            return self.mix_multibin(X, alpha)
        else:
            single = X.ndim == 2
            result = self.mix(_ensure_3d(X), alpha)
            return result.squeeze(0) if single else result

    def __repr__(self):
        alpha_dict = self.get_alpha_dict()
        # Show top 3 features by |weight|
        sorted_feats = sorted(alpha_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top = ', '.join(f'{n}={v:.3f}' for n, v in sorted_feats[:3])
        return f"DictionaryFiltration(K={self.K}, top=[{top}], fitted={self._is_fitted})"
