"""
Persistence Landscape and Persistence Silhouette vectorizations.

Persistence Landscape (Bubenik 2015):
    λ_k(t) = k-th largest of Λ_(b,d)(t) over diagram points.

Persistence Silhouette (Chazal et al. 2014):
    sil(t) = (1/N) Σ_{p} Λ_(b_p,d_p)(t)   [uniform weighting]

where Λ_(b,d)(t) = max(0, min(t − b, d − t)) is the tent function.

The silhouette uses mean-pooling instead of top-k, which is critical for
MOPED compatibility on rough fields (e.g. B=2 GRF): at any moderate grid
point t, O(N_pairs) tents contribute → CLT → near-Gaussian MOPED output.
The landscape (top-k) follows extreme-value statistics → non-Gaussian.

References:
  Bubenik 2015 https://www.jmlr.org/papers/v16/bubenik15a.html
  Chazal et al. 2014 https://arxiv.org/abs/1312.0308
"""
from typing import List
import math
import warnings
import torch
import torch.nn as nn
import numpy as np


class DifferentiablePersistenceLandscapeLayer(nn.Module):
    """k-th Persistence Landscapes evaluated on a fixed grid.

    Same interface as the other vectorization layers: partial_fit /
    finalize_fit / forward, single homology dimension, designed for use
    inside CombinedVectorization.

    Param budget for the learnable variant: 1 scalar (log_slope_scale).
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_grid: int = 16,
        percentile_clip: float = 99.0,
        learn_slope: bool = False,
        slope_clamp_factor: float = 3.0,
    ):
        """Initialise persistence landscape layer.

        Args:
            n_layers: K, number of landscapes (top-K of triangle values).
            n_grid: Resolution of the 1D grid in t.
            percentile_clip: Upper percentile for grid range from
                pooled (births ∪ deaths). Lower bound = (100 − clip).
            learn_slope: If True, the triangle slope is multiplied by a
                learnable scalar log_slope_scale; default False.
            slope_clamp_factor: Multiplicative range for the learnable slope
                (clamped to [1/factor, factor] around 1).
        """
        super().__init__()
        self.n_layers = int(n_layers)
        self.n_grid = int(n_grid)
        self.percentile_clip = float(percentile_clip)
        self.learn_slope = bool(learn_slope)
        self.slope_clamp_factor = float(slope_clamp_factor)
        self.n_features = self.n_layers * self.n_grid
        self.fitted = False

        # Grid populated in finalize_fit().
        self.register_buffer("grid", torch.linspace(0.0, 1.0, self.n_grid))

        if self.learn_slope:
            # Slope multiplier in log-space; init at 1.0.
            self.log_slope_scale = nn.Parameter(torch.zeros(()))
            log_factor = math.log(self.slope_clamp_factor)
            self.register_buffer("log_slope_min", torch.tensor(-log_factor))
            self.register_buffer("log_slope_max", torch.tensor(log_factor))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """Pool births and deaths to determine the grid range."""
        if not hasattr(self, "_pool"):
            self._pool = []
        for dgm in diagrams:
            if dgm.shape[0] > 0:
                pts = dgm.detach().cpu().numpy()
                pers = pts[:, 1] - pts[:, 0]
                valid = pers > 0
                if valid.any():
                    self._pool.append(pts[valid].reshape(-1))

    def finalize_fit(self):
        """Set the t-grid from pooled birth/death percentiles."""
        if not hasattr(self, "_pool") or len(self._pool) == 0:
            warnings.warn(
                "PersistenceLandscape: no valid points; using default [0, 1] grid.",
                RuntimeWarning,
            )
            self.grid = torch.linspace(0.0, 1.0, self.n_grid)
            self.fitted = True
            self._cleanup_pool()
            return

        pool = np.concatenate(self._pool)
        lo_pct = 100.0 - self.percentile_clip
        hi_pct = self.percentile_clip
        v_lo = float(np.percentile(pool, lo_pct))
        v_hi = float(np.percentile(pool, hi_pct))

        if v_hi - v_lo < 1e-8:
            v_lo, v_hi = v_lo - 0.5, v_lo + 0.5

        self.grid = torch.linspace(v_lo, v_hi, self.n_grid)
        self.fitted = True
        self._cleanup_pool()

        print(
            f"  PersistenceLandscapeLayer fitted: "
            f"grid [{v_lo:.4f}, {v_hi:.4f}], n_grid={self.n_grid}, "
            f"n_layers={self.n_layers}, learn_slope={self.learn_slope}, "
            f"{pool.shape[0]} pooled values"
        )

    def fit(self, diagrams: List[torch.Tensor]):
        """One-shot fit on a flat list of diagrams."""
        self._pool = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def _cleanup_pool(self):
        if hasattr(self, "_pool"):
            del self._pool

    def get_slope_scale(self) -> torch.Tensor:
        if self.learn_slope:
            log_s = self.log_slope_scale.clamp(self.log_slope_min, self.log_slope_max)
            return log_s.exp()
        return torch.tensor(1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """Compute (n_samples, n_layers * n_grid) landscape features."""
        if not self.fitted:
            raise RuntimeError(
                "DifferentiablePersistenceLandscapeLayer must be fitted "
                "before forward(). Call .fit() first."
            )

        batch_size = len(diagrams)
        if batch_size == 0:
            return torch.empty((0, self.n_features))

        device = self.grid.device
        for dgm in diagrams:
            if dgm.numel() > 0:
                device = dgm.device
                break

        grid = self.grid.to(device)                                     # (G,)

        n_pairs_list = [int(dgm.shape[0]) for dgm in diagrams]
        max_K = max(n_pairs_list) if n_pairs_list else 0
        if max_K == 0:
            return torch.zeros((batch_size, self.n_features), device=device)

        diagrams_safe = [
            dgm if dgm.shape[0] > 0
            else torch.zeros((1, 2), device=device, dtype=torch.float32)
            for dgm in diagrams
        ]
        n_pairs = torch.tensor(n_pairs_list, device=device)             # (B,)
        dgm_pad = torch.nn.utils.rnn.pad_sequence(
            diagrams_safe, batch_first=True, padding_value=0.0
        )                                                               # (B, M, 2)
        M = dgm_pad.shape[1]
        idx_range = torch.arange(M, device=device).unsqueeze(0)         # (1, M)
        original_mask = idx_range < n_pairs.unsqueeze(1)                # (B, M)
        pers_all = dgm_pad[:, :, 1] - dgm_pad[:, :, 0]                  # (B, M)
        valid_mask = original_mask & (pers_all > 0)                     # (B, M)

        # Triangle tent at every grid point: shape (B, M, G).
        # Λ_(b,d)(t) = relu(min(t - b, d - t)) * slope_scale.
        births = dgm_pad[:, :, 0].unsqueeze(-1)                         # (B, M, 1)
        deaths = dgm_pad[:, :, 1].unsqueeze(-1)                         # (B, M, 1)
        t = grid.view(1, 1, -1)                                         # (1, 1, G)
        left = t - births
        right = deaths - t
        tent = torch.minimum(left, right).clamp(min=0.0)                # (B, M, G)

        if self.learn_slope:
            slope = self.get_slope_scale().to(device)
            tent = tent * slope

        # Mask invalid (padded or zero-persistence) rows to 0 — landscapes are
        # non-negative, so 0 effectively excludes them from the top-k unless
        # all rows are zero (in which case the output is correctly all-zero).
        tent = tent * valid_mask.unsqueeze(-1).to(tent.dtype)

        # Top-k over M (the diagram-points axis) → (B, K, G).
        K = min(self.n_layers, tent.shape[1])
        top = torch.topk(tent, k=K, dim=1).values                       # (B, K, G)

        if K < self.n_layers:
            pad = torch.zeros(batch_size, self.n_layers - K, self.n_grid,
                              device=device, dtype=top.dtype)
            top = torch.cat([top, pad], dim=1)

        return top.reshape(batch_size, self.n_features)

    def __repr__(self):
        fitted_str = "fitted" if self.fitted else "not fitted"
        slope_str = (
            f"learn_slope=True, scale={self.get_slope_scale().item():.3f}"
            if self.learn_slope else "learn_slope=False"
        )
        return (
            f"DifferentiablePersistenceLandscapeLayer("
            f"n_layers={self.n_layers}, n_grid={self.n_grid}, "
            f"features={self.n_features}, {slope_str}, {fitted_str})"
        )


# ---------------------------------------------------------------------------
# Persistence Silhouette
# ---------------------------------------------------------------------------

class PersistenceSilhouetteLayer(nn.Module):
    """Persistence Silhouette on a fixed 1D grid (Chazal et al. 2014).

    sil(t) = (1/N) Σ_{p valid}  Λ_(b_p, d_p)(t)   [uniform weighting]
           = weighted_mean_{p} Λ_(b_p, d_p)(t)      [sqrt-persistence weighting]

    Mean-pooling over all diagram points (rather than top-k as in the
    landscape) is the key property: for many pairs (rough fields, B=2 GRF),
    CLT applies → near-Gaussian MOPED summaries.

    Default weighting is "uniform" (w=1 for all valid pairs), which
    maximises the effective sample size and is required for Gaussianity when
    the high-persistence tail is sparse.  "sqrt_persistence" is provided for
    the learnable variant where a learnable slope can compensate.

    Feature dimension: n_grid (one function per homology dimension).
    """

    def __init__(
        self,
        n_grid: int = 50,
        percentile_clip: float = 99.0,
        weighting: str = "uniform",
        learn_slope: bool = False,
        slope_clamp_factor: float = 3.0,
    ):
        """
        Args:
            n_grid: Number of evaluation points on the t-axis.
            percentile_clip: Grid range from pooled (births ∪ deaths)
                percentiles: [100-clip, clip].
            weighting: "uniform" (w=1, recommended for MOPED Gaussianity)
                or "sqrt_persistence" (w=sqrt(d-b), for learnable variant).
            learn_slope: If True, add a learnable log_slope_scale parameter.
            slope_clamp_factor: Clamp range for the learnable slope.
        """
        super().__init__()
        if weighting not in ("uniform", "sqrt_persistence"):
            raise ValueError(f"Unknown weighting '{weighting}'.")
        self.n_grid = int(n_grid)
        self.percentile_clip = float(percentile_clip)
        self.weighting = weighting
        self.learn_slope = bool(learn_slope)
        self.slope_clamp_factor = float(slope_clamp_factor)
        self.n_features = self.n_grid
        self.fitted = False

        self.register_buffer("grid", torch.linspace(0.0, 1.0, self.n_grid))

        if self.learn_slope:
            self.log_slope_scale = nn.Parameter(torch.zeros(()))
            log_factor = math.log(self.slope_clamp_factor)
            self.register_buffer("log_slope_min", torch.tensor(-log_factor))
            self.register_buffer("log_slope_max", torch.tensor(log_factor))

    # ------------------------------------------------------------------
    # Fitting — identical to landscape layer
    # ------------------------------------------------------------------

    def partial_fit(self, diagrams: List[torch.Tensor]):
        if not hasattr(self, "_pool"):
            self._pool = []
        for dgm in diagrams:
            if dgm.shape[0] > 0:
                pts = dgm.detach().cpu().numpy()
                pers = pts[:, 1] - pts[:, 0]
                valid = pers > 0
                if valid.any():
                    self._pool.append(pts[valid].reshape(-1))

    def finalize_fit(self):
        if not hasattr(self, "_pool") or len(self._pool) == 0:
            warnings.warn(
                "PersistenceSilhouette: no valid points; using default [0, 1] grid.",
                RuntimeWarning,
            )
            self.grid = torch.linspace(0.0, 1.0, self.n_grid)
            self.fitted = True
            self._cleanup_pool()
            return

        pool = np.concatenate(self._pool)
        lo_pct = 100.0 - self.percentile_clip
        hi_pct = self.percentile_clip
        v_lo = float(np.percentile(pool, lo_pct))
        v_hi = float(np.percentile(pool, hi_pct))
        if v_hi - v_lo < 1e-8:
            v_lo, v_hi = v_lo - 0.5, v_lo + 0.5

        self.grid = torch.linspace(v_lo, v_hi, self.n_grid)
        self.fitted = True
        self._cleanup_pool()

        print(
            f"  PersistenceSilhouetteLayer fitted: "
            f"grid [{v_lo:.4f}, {v_hi:.4f}], n_grid={self.n_grid}, "
            f"weighting={self.weighting}, {pool.shape[0]} pooled values"
        )

    def fit(self, diagrams: List[torch.Tensor]):
        self._pool = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def _cleanup_pool(self):
        if hasattr(self, "_pool"):
            del self._pool

    def get_slope_scale(self) -> torch.Tensor:
        if self.learn_slope:
            log_s = self.log_slope_scale.clamp(self.log_slope_min, self.log_slope_max)
            return log_s.exp()
        return torch.tensor(1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """Return (n_samples, n_grid) silhouette features."""
        if not self.fitted:
            raise RuntimeError(
                "PersistenceSilhouetteLayer must be fitted before forward()."
            )

        batch_size = len(diagrams)
        if batch_size == 0:
            return torch.empty((0, self.n_features))

        device = self.grid.device
        for dgm in diagrams:
            if dgm.numel() > 0:
                device = dgm.device
                break

        grid = self.grid.to(device)                                     # (G,)

        n_pairs_list = [int(dgm.shape[0]) for dgm in diagrams]
        max_K = max(n_pairs_list) if n_pairs_list else 0
        if max_K == 0:
            return torch.zeros((batch_size, self.n_features), device=device)

        diagrams_safe = [
            dgm if dgm.shape[0] > 0
            else torch.zeros((1, 2), device=device, dtype=torch.float32)
            for dgm in diagrams
        ]
        n_pairs = torch.tensor(n_pairs_list, device=device)
        dgm_pad = torch.nn.utils.rnn.pad_sequence(
            diagrams_safe, batch_first=True, padding_value=0.0
        )                                                               # (B, M, 2)
        M = dgm_pad.shape[1]
        idx_range = torch.arange(M, device=device).unsqueeze(0)
        original_mask = idx_range < n_pairs.unsqueeze(1)                # (B, M)
        pers_all = dgm_pad[:, :, 1] - dgm_pad[:, :, 0]                 # (B, M)
        valid_mask = original_mask & (pers_all > 0)                     # (B, M)

        # Tent functions (B, M, G)
        births = dgm_pad[:, :, 0].unsqueeze(-1)
        deaths = dgm_pad[:, :, 1].unsqueeze(-1)
        t = grid.view(1, 1, -1)
        tent = torch.minimum(t - births, deaths - t).clamp(min=0.0)

        if self.learn_slope:
            slope = self.get_slope_scale().to(device)
            tent = tent * slope

        # Per-point weights (B, M): uniform or sqrt-persistence
        if self.weighting == "uniform":
            weights = valid_mask.float()
        else:  # sqrt_persistence
            weights = (pers_all.clamp(min=0.0) + 1e-12).sqrt() * valid_mask.float()

        # Weighted mean over diagram points → (B, G)
        total_w = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)    # (B, 1)
        out = (tent * weights.unsqueeze(-1)).sum(dim=1) / total_w       # (B, G)

        return out

    def __repr__(self):
        fitted_str = "fitted" if self.fitted else "not fitted"
        slope_str = (
            f"learn_slope=True, scale={self.get_slope_scale().item():.3f}"
            if self.learn_slope else "learn_slope=False"
        )
        return (
            f"PersistenceSilhouetteLayer("
            f"n_grid={self.n_grid}, weighting='{self.weighting}', "
            f"{slope_str}, {fitted_str})"
        )
