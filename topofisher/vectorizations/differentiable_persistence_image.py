"""
Differentiable Persistence Image vectorization in pure PyTorch.

Unlike the GUDHI-based PersistenceImageLayer, this implementation maintains
gradient flow from the persistence diagram (birth, death) values through
to the output image pixels. This enables end-to-end training of learnable
filtrations with persistence image summaries.

Reference: Adams et al. 2017 (https://jmlr.org/papers/v18/16-337.html)
"""
from typing import List, Optional
import math
import torch
import torch.nn as nn
import warnings


class DifferentiablePersistenceImageLayer(nn.Module):
    """
    Differentiable Persistence Image for a single homology dimension.

    Converts persistence diagrams to persistence images by placing
    Gaussian kernels at each (birth, persistence) point on a fixed grid.
    All operations are in PyTorch, preserving autograd gradients.

    Interface matches TopKLayer so it can be used with CombinedVectorization.
    """

    def __init__(
        self,
        n_pixels: int = 8,
        bandwidth: float = 1.0,
        weighting: str = "persistence",
        learn_bandwidth: bool = False,
        bandwidth_clamp_factor: float = 3.0,
        auto_scale_fixed_bandwidth: bool = True,
    ):
        """
        Initialize differentiable persistence image layer.

        Args:
            n_pixels: Resolution per axis (output is n_pixels x n_pixels = n_pixels² features)
            bandwidth: Gaussian kernel bandwidth (sigma). Used as fallback when
                      learn_bandwidth=True and fit() hasn't been called yet.
                      After fit(), bandwidth is auto-initialized from grid spacing.
            weighting: "persistence" weights by sqrt(persistence), "uniform" uses 1.0
            learn_bandwidth: If True, bandwidth is a trainable parameter optimized
                            jointly with the CNN. Parameterized in log-space to ensure
                            positivity: sigma = exp(log_sigma). Auto-initialized from
                            data scale in fit() and clamped to prevent drift.
            bandwidth_clamp_factor: When learn_bandwidth=True, sigma is clamped to
                            [optimal/factor, optimal*factor]. Default 3.0 means
                            sigma can vary ~3× around the data-optimal value.
            auto_scale_fixed_bandwidth: If True (default), fixed bandwidth is auto-scaled
                            once from diagram scale during fit(). Set False to preserve
                            exact user-provided fixed bandwidth (legacy behavior).
        """
        super().__init__()
        self.n_pixels = n_pixels
        self.weighting = weighting
        self.n_features = n_pixels * n_pixels
        self.learn_bandwidth = learn_bandwidth
        self.bandwidth_clamp_factor = bandwidth_clamp_factor
        self.auto_scale_fixed_bandwidth = auto_scale_fixed_bandwidth

        if learn_bandwidth:
            # Learnable log-bandwidth: sigma = exp(log_sigma), ensures sigma > 0
            # This is a provisional value; fit() will reinitialize from data scale.
            self.log_bandwidth = nn.Parameter(torch.tensor(float(bandwidth)).log())
            # Clamp bounds (will be set properly in fit())
            self.register_buffer('log_bw_min', torch.tensor(float('-inf')))
            self.register_buffer('log_bw_max', torch.tensor(float('inf')))
        else:
            self.bandwidth = bandwidth

        # Bounds will be set during fit()
        self.register_buffer('birth_min', torch.tensor(0.0))
        self.register_buffer('birth_max', torch.tensor(1.0))
        self.register_buffer('pers_min', torch.tensor(0.0))
        self.register_buffer('pers_max', torch.tensor(1.0))
        self.fitted = False

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """
        Update running min/max bounds from a batch of diagrams.

        Call this multiple times (e.g., once per dataset), then call
        finalize_fit() to set bounds and bandwidth. This avoids needing
        all diagrams in memory at once.

        Args:
            diagrams: List of diagrams (each shape (n_points, 2) with birth, death)
        """
        if not hasattr(self, '_running_b_min'):
            self._running_b_min = float('inf')
            self._running_b_max = float('-inf')
            self._running_p_min = float('inf')
            self._running_p_max = float('-inf')
            self._running_has_data = False

        for dgm in diagrams:
            if dgm.shape[0] > 0:
                dgm_detached = dgm.detach()
                births = dgm_detached[:, 0]
                pers = dgm_detached[:, 1] - dgm_detached[:, 0]
                valid = pers > 0
                if valid.any():
                    self._running_has_data = True
                    b = births[valid]
                    p = pers[valid]
                    self._running_b_min = min(self._running_b_min, b.min().item())
                    self._running_b_max = max(self._running_b_max, b.max().item())
                    self._running_p_min = min(self._running_p_min, p.min().item())
                    self._running_p_max = max(self._running_p_max, p.max().item())

    def finalize_fit(self):
        """
        Finalize bounds after all partial_fit() calls and set bandwidth.

        Must be called after all partial_fit() calls and before forward().
        """
        if not hasattr(self, '_running_b_min') or not self._running_has_data:
            warnings.warn(
                "No valid persistence points found. Using default bounds [0, 1].",
                RuntimeWarning
            )
            self.birth_min.fill_(0.0)
            self.birth_max.fill_(1.0)
            self.pers_min.fill_(0.0)
            self.pers_max.fill_(1.0)
        else:
            b_min, b_max = self._running_b_min, self._running_b_max
            p_min, p_max = self._running_p_min, self._running_p_max

            self._apply_bounds_with_margin(b_min, b_max, p_min, p_max)

        # Clean up running state
        for attr in ('_running_b_min', '_running_b_max',
                     '_running_p_min', '_running_p_max', '_running_has_data'):
            if hasattr(self, attr):
                delattr(self, attr)

        self._set_bandwidth_from_bounds()
        self.fitted = True

    def _apply_bounds_with_margin(self, b_min, b_max, p_min, p_max):
        """Apply birth/persistence bounds with 10% margin."""
        b_range = b_max - b_min
        p_range = p_max - p_min

        if b_range < 1e-6:
            b_min, b_max = 0.0, 1.0
        else:
            margin = 0.1 * b_range
            b_min -= margin
            b_max += margin

        if p_range < 1e-6:
            p_min, p_max = 0.0, 1.0
        else:
            margin = 0.1 * p_range
            p_min = max(0.0, p_min - margin)
            p_max += margin

        self.birth_min.fill_(b_min)
        self.birth_max.fill_(b_max)
        self.pers_min.fill_(p_min)
        self.pers_max.fill_(p_max)

    def fit(self, diagrams: List[torch.Tensor]):
        """
        Compute pixel grid bounds from ALL diagrams for this homology dimension.

        Args:
            diagrams: Flat list of diagrams (each shape (n_points, 2) with birth, death)
        """
        all_births = []
        all_pers = []

        for dgm in diagrams:
            if dgm.shape[0] > 0:
                dgm_detached = dgm.detach()
                births = dgm_detached[:, 0]
                pers = dgm_detached[:, 1] - dgm_detached[:, 0]
                valid = pers > 0
                if valid.any():
                    all_births.append(births[valid])
                    all_pers.append(pers[valid])

        if not all_births:
            warnings.warn(
                "No valid persistence points found. Using default bounds [0, 1].",
                RuntimeWarning
            )
            self.birth_min.fill_(0.0)
            self.birth_max.fill_(1.0)
            self.pers_min.fill_(0.0)
            self.pers_max.fill_(1.0)
        else:
            births_cat = torch.cat(all_births)
            pers_cat = torch.cat(all_pers)

            b_min, b_max = births_cat.min().item(), births_cat.max().item()
            p_min, p_max = pers_cat.min().item(), pers_cat.max().item()

            self._apply_bounds_with_margin(b_min, b_max, p_min, p_max)

        self._set_bandwidth_from_bounds()
        self.fitted = True

    def _set_bandwidth_from_bounds(self):
        """Set bandwidth based on current birth/pers bounds. Called by fit() and finalize_fit()."""
        if not self.fitted:
            b_range = self.birth_max.item() - self.birth_min.item()
            p_range = self.pers_max.item() - self.pers_min.item()
            # Optimal sigma ≈ half the grid spacing (each Gaussian covers ~1 pixel)
            grid_spacing = max(b_range, p_range) / self.n_pixels
            optimal_sigma = grid_spacing / 2.0
            optimal_sigma = max(optimal_sigma, 1e-4)  # safety floor

            if self.learn_bandwidth:
                # Reinitialize log_bandwidth to data-optimal value
                with torch.no_grad():
                    self.log_bandwidth.fill_(math.log(optimal_sigma))

                # Set clamp bounds: allow bandwidth_clamp_factor× variation
                log_factor = math.log(self.bandwidth_clamp_factor)
                self.log_bw_min.fill_(math.log(optimal_sigma) - log_factor)
                self.log_bw_max.fill_(math.log(optimal_sigma) + log_factor)

                warnings.warn(
                    f"DiffPI: auto-initialized learnable σ={optimal_sigma:.4f} "
                    f"from grid spacing {grid_spacing:.4f} "
                    f"(range: [{optimal_sigma/self.bandwidth_clamp_factor:.4f}, "
                    f"{optimal_sigma*self.bandwidth_clamp_factor:.4f}])",
                    stacklevel=2,
                )
            else:
                if self.auto_scale_fixed_bandwidth:
                    # Auto-scale fixed bandwidth from persistence diagram scale.
                    # Computed once on first fit(), then frozen for all training.
                    # This ensures σ matches the data regardless of B value.
                    old_bw = self.bandwidth
                    self.bandwidth = optimal_sigma
                    warnings.warn(
                        f"DiffPI: auto-scaled fixed σ: {old_bw:.4f} → {optimal_sigma:.4f} "
                        f"(grid_spacing={grid_spacing:.4f}, "
                        f"birth=[{self.birth_min.item():.4f}, {self.birth_max.item():.4f}], "
                        f"pers=[{self.pers_min.item():.4f}, {self.pers_max.item():.4f}])",
                        stacklevel=2,
                    )

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert persistence diagrams to differentiable persistence images.

        Uses batched GPU operations: pad all diagrams to the same length,
        then compute all Gaussian kernels in a single (B, M, P) tensor op.
        This is ~10× faster than the per-sample Python loop for large batches.

        Args:
            diagrams: List of diagrams for ONE homology dimension,
                      each of shape (n_points, 2) with (birth, death) pairs.

        Returns:
            Tensor of shape (n_samples, n_pixels * n_pixels)
        """
        if not self.fitted:
            raise RuntimeError(
                "DifferentiablePersistenceImageLayer must be fitted before use. "
                "Call .fit() first."
            )

        batch_size = len(diagrams)
        if batch_size == 0:
            return torch.empty((0, self.n_features))

        device = diagrams[0].device

        # Build pixel center grid — (P,) vectors
        birth_centers = torch.linspace(
            self.birth_min.item(), self.birth_max.item(), self.n_pixels, device=device
        )
        pers_centers = torch.linspace(
            self.pers_min.item(), self.pers_max.item(), self.n_pixels, device=device
        )
        grid_b, grid_p = torch.meshgrid(birth_centers, pers_centers, indexing='ij')
        grid_b_flat = grid_b.reshape(-1)  # (P,)
        grid_p_flat = grid_p.reshape(-1)  # (P,)
        P = self.n_features

        # Compute 2σ²
        if self.learn_bandwidth:
            with torch.no_grad():
                self.log_bandwidth.clamp_(
                    self.log_bw_min.item(), self.log_bw_max.item()
                )
            sigma = self.log_bandwidth.exp()
            sigma_sq_2 = 2.0 * sigma ** 2
        else:
            sigma_sq_2 = 2.0 * self.bandwidth ** 2

        # ── Fully batched path: pad raw diagrams first, then all ops in batch ──
        # Step 1: Pad raw diagrams to (B, max_K, 2) — single pad_sequence call
        # This avoids iterating per-sample for births/pers computation.
        n_pairs_list = [dgm.shape[0] for dgm in diagrams]  # fast attribute access, no GPU ops
        max_K = max(n_pairs_list) if n_pairs_list else 0

        if max_K == 0:
            return torch.zeros((batch_size, P), device=device)

        # Handle empty diagrams: replace (0,2) with (1,2) zeros so pad_sequence works,
        # then use n_pairs to mask them out.
        diagrams_safe = [
            dgm if dgm.shape[0] > 0 else torch.zeros((1, 2), device=device, dtype=dgm.dtype)
            for dgm in diagrams
        ]
        n_pairs = torch.tensor(n_pairs_list, device=device)  # (B,)

        # Pad all diagrams into a single (B, max_K_safe, 2) tensor
        dgm_pad = torch.nn.utils.rnn.pad_sequence(
            diagrams_safe, batch_first=True, padding_value=0.0
        )  # (B, max_K_safe, 2)
        max_K_safe = dgm_pad.shape[1]

        # Step 2: Compute births and persistence in batch
        births_all = dgm_pad[:, :, 0]                    # (B, max_K_safe)
        pers_all = dgm_pad[:, :, 1] - dgm_pad[:, :, 0]  # (B, max_K_safe)

        # Step 3: Build validity mask (original pairs AND pers > 0)
        idx_range = torch.arange(max_K_safe, device=device).unsqueeze(0)  # (1, max_K_safe)
        original_mask = idx_range < n_pairs.unsqueeze(1)  # (B, max_K) — True for non-padded
        valid_mask = original_mask & (pers_all > 0)       # (B, max_K) — also pers > 0
        mask = valid_mask.float()                         # (B, max_K)

        # Zero out invalid entries so they don't contribute to Gaussian
        births_pad = births_all * mask  # (B, max_K)
        pers_pad = pers_all * mask      # (B, max_K)

        # Weights: (B, max_K)
        if self.weighting == "persistence":
            weights = torch.sqrt(pers_pad.clamp(min=1e-8)) * mask
        else:
            weights = mask.clone()

        # Gaussian kernel — fully batched: (B, max_K, P)
        # births_pad: (B, K) → (B, K, 1), grid_b_flat: (P,) → (1, 1, P)
        diff_b = births_pad.unsqueeze(2) - grid_b_flat.unsqueeze(0).unsqueeze(0)  # (B, K, P)
        diff_p = pers_pad.unsqueeze(2) - grid_p_flat.unsqueeze(0).unsqueeze(0)    # (B, K, P)

        gauss = torch.exp(-(diff_b ** 2 + diff_p ** 2) / sigma_sq_2)  # (B, K, P)

        # Weighted sum: (B, K, 1) * (B, K, P) → sum over K → (B, P)
        images = (weights.unsqueeze(2) * gauss).sum(dim=1)  # (B, P)

        return images

    def get_bandwidth(self) -> float:
        """Return current bandwidth value (whether learnable or fixed)."""
        if self.learn_bandwidth:
            return self.log_bandwidth.exp().item()
        return self.bandwidth

    def __repr__(self):
        fitted_str = "fitted" if self.fitted else "not fitted"
        bw = self.get_bandwidth()
        if self.learn_bandwidth:
            bw_lo = math.exp(self.log_bw_min.item()) if self.log_bw_min.item() > -1e6 else 0.0
            bw_hi = math.exp(self.log_bw_max.item()) if self.log_bw_max.item() < 1e6 else float('inf')
            bw_str = f"bandwidth={bw:.4f} (learnable, range=[{bw_lo:.4f}, {bw_hi:.4f}])"
        else:
            bw_str = f"bandwidth={bw}"
        return (f"DifferentiablePersistenceImageLayer("
                f"n_pixels={self.n_pixels}, {bw_str}, "
                f"weighting='{self.weighting}', {fitted_str})")
