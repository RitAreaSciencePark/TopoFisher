"""
Differentiable Persistence Curve vectorization (Empirical Distribution Functions).

Differentiable version of PersistenceCurveLayer that enables gradient flow from
the Fisher loss back through the curve features to the persistence diagram values
(and thus to the CNN that produced the filtration).

The key idea: replace hard counting  count(birth > ν) / N  with soft sigmoid
counting  sum(sigmoid((birth - ν) / τ)) / N.  With a small temperature τ the
sigmoid approximates a step function, but has non-zero gradients everywhere —
enabling end-to-end training of CNN + Cubical + Curves + MOPED pipelines.

Reference: Biagetti, Cole & Shiu (arXiv:2009.04819) for the curve definitions.
"""
from typing import List, Optional
import math
import torch
import torch.nn as nn
import numpy as np
import warnings


class DifferentiablePersistenceCurveLayer(nn.Module):
    """
    Differentiable EDF curves from persistence diagrams.

    For a single homology dimension, computes 1D curves by soft-counting
    persistence diagram features, evaluated on a fixed grid.  All operations
    are in pure PyTorch, preserving autograd gradients.

    Same interface as PersistenceCurveLayer (partial_fit / finalize_fit /
    forward) so it can be used with CombinedVectorization.
    """

    def __init__(
        self,
        n_grid: int = 100,
        curves: str = "B,D,P",
        min_persistence: float = 0.0,
        percentile_clip: float = 99.9,
        temperature: Optional[float] = None,
        learn_temperature: bool = False,
        temperature_clamp_factor: float = 5.0,
        tau_scale: float = 2.0,
        freeze_after_fit: bool = False,
    ):
        """
        Initialise differentiable persistence curve layer.

        Args:
            n_grid: Number of evaluation points per curve.
            curves: Comma-separated list of curves (B, D, P, b).
            min_persistence: Minimum persistence to include a feature.
            percentile_clip: Upper percentile for grid range determination.
            temperature: Sigmoid temperature τ.  If None (default), auto-set
                from grid spacing in finalize_fit():  τ = grid_step * tau_scale.
            learn_temperature: If True, τ is a trainable parameter
                (parameterised in log-space to ensure positivity).
            temperature_clamp_factor: If learn_temperature, clamp τ to
                [τ_opt / factor, τ_opt × factor].
            tau_scale: Multiplier for auto temperature relative to grid step.
                τ = grid_step * tau_scale.  Larger = smoother sigmoid = more
                stable gradients but less sharp counting.  Default 2.0
                (transition spans ~2 grid cells).  Previous default was 0.25.
            freeze_after_fit: If True, the grid and temperature are locked
                after the first successful fit().  Subsequent fit() calls
                become no-ops.  Essential for learnable pipelines where the
                filtration changes each epoch — prevents the feature space
                from shifting every step.
        """
        super().__init__()
        self.n_grid = n_grid
        self.curve_names = [c.strip() for c in curves.split(",")]
        self.min_persistence = min_persistence
        self.percentile_clip = percentile_clip
        self.learn_temperature = learn_temperature
        self.temperature_clamp_factor = temperature_clamp_factor

        # Validate curve names
        valid = {"B", "D", "P", "b"}
        for c in self.curve_names:
            if c not in valid:
                raise ValueError(f"Unknown curve '{c}'. Valid: {valid}")

        self.n_curves = len(self.curve_names)
        self.n_features = self.n_curves * n_grid
        self.tau_scale = tau_scale
        self.freeze_after_fit = freeze_after_fit
        self._frozen = False  # set True after first fit when freeze_after_fit=True

        # Grid will be set during fit
        self.register_buffer("grid", torch.zeros(n_grid))
        self.fitted = False

        if learn_temperature:
            # Learnable log-temperature: τ = exp(log_τ), ensures τ > 0
            init_val = temperature if temperature is not None else 0.01
            self.log_temperature = nn.Parameter(
                torch.tensor(math.log(init_val))
            )
            self.register_buffer(
                "log_temp_min", torch.tensor(float("-inf"))
            )
            self.register_buffer(
                "log_temp_max", torch.tensor(float("inf"))
            )
        else:
            # Fixed temperature (set in finalize_fit if None)
            self._user_temperature = temperature

    # ------------------------------------------------------------------
    # Fitting (grid range determination — same logic as non-diff version)
    # ------------------------------------------------------------------

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """Accumulate filtration value ranges from a batch of diagrams."""
        if not hasattr(self, "_all_vals"):
            self._all_vals = []

        for dgm in diagrams:
            if dgm.shape[0] > 0:
                dgm_np = dgm.detach().cpu().numpy()
                births = dgm_np[:, 0]
                deaths = dgm_np[:, 1]
                pers = deaths - births
                valid = pers > self.min_persistence
                if valid.any():
                    self._all_vals.append(births[valid])
                    self._all_vals.append(deaths[valid])

    def finalize_fit(self):
        """Compute the evaluation grid and auto-set temperature."""
        if not hasattr(self, "_all_vals") or len(self._all_vals) == 0:
            warnings.warn(
                "No data accumulated for DifferentiablePersistenceCurveLayer. "
                "Using default [0, 1] grid."
            )
            self.grid = torch.linspace(0, 1, self.n_grid)
            self.fitted = True
            return

        all_vals = np.concatenate(self._all_vals)
        lo_pct = 100.0 - self.percentile_clip
        hi_pct = self.percentile_clip
        v_lo = float(np.percentile(all_vals, lo_pct))
        v_hi = float(np.percentile(all_vals, hi_pct))

        if v_hi - v_lo < 1e-8:
            v_lo, v_hi = v_lo - 0.5, v_lo + 0.5

        self.grid = torch.linspace(v_lo, v_hi, self.n_grid)

        # Auto-set temperature from grid spacing
        grid_step = (v_hi - v_lo) / max(self.n_grid - 1, 1)
        auto_temp = grid_step * self.tau_scale  # smooth transition

        if self.learn_temperature:
            # Reinitialise learnable τ from data scale
            with torch.no_grad():
                self.log_temperature.fill_(math.log(auto_temp))
            self.log_temp_min.fill_(
                math.log(auto_temp / self.temperature_clamp_factor)
            )
            self.log_temp_max.fill_(
                math.log(auto_temp * self.temperature_clamp_factor)
            )
        else:
            if self._user_temperature is not None:
                self._temperature = self._user_temperature
            else:
                self._temperature = auto_temp

        if not getattr(self, 'fitted', False):
            print(
                f"  DiffPersistenceCurveLayer fitted: "
                f"grid [{v_lo:.4f}, {v_hi:.4f}], {self.n_grid} pts, "
                f"curves={self.curve_names}, τ={self.get_temperature():.6f}, "
                f"{len(all_vals)} pooled values"
            )

        self.fitted = True
        if self.freeze_after_fit:
            self._frozen = True
        if hasattr(self, "_all_vals"):
            del self._all_vals

    def fit(self, diagrams: List[torch.Tensor]):
        """One-shot fit.  No-op if grid is frozen."""
        if self._frozen:
            return
        self._all_vals = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def get_temperature(self) -> float:
        """Return the current temperature value."""
        if self.learn_temperature:
            log_t = self.log_temperature.clamp(
                self.log_temp_min, self.log_temp_max
            )
            return log_t.exp().item()
        else:
            return getattr(self, "_temperature", 0.01)

    # ------------------------------------------------------------------
    # Forward (differentiable)
    # ------------------------------------------------------------------

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute differentiable persistence curve features.

        Uses soft sigmoid counting so gradients flow from the output
        back through to the diagram birth/death values.

        Args:
            diagrams: List of n_samples diagrams, each (n_points, 2).

        Returns:
            Tensor (n_samples, n_curves * n_grid).
        """
        if not self.fitted:
            raise RuntimeError(
                "DifferentiablePersistenceCurveLayer must be fitted "
                "before forward()."
            )

        batch_size = len(diagrams)
        # Determine device from input diagrams, not self.grid (which may be CPU)
        device = self.grid.device
        for dgm in diagrams:
            if dgm.numel() > 0:
                device = dgm.device
                break

        # Get temperature
        if self.learn_temperature:
            log_t = self.log_temperature.clamp(
                self.log_temp_min, self.log_temp_max
            )
            tau = log_t.exp()
        else:
            tau = torch.tensor(
                self._temperature, device=device, dtype=self.grid.dtype
            )

        grid = self.grid.to(device)  # (n_grid,)

        out = torch.zeros(batch_size, self.n_features, device=device)

        for i, dgm in enumerate(diagrams):
            if dgm.shape[0] == 0:
                continue

            dgm = dgm.to(device)
            births = dgm[:, 0]        # (n_pts,)
            deaths = dgm[:, 1]        # (n_pts,)
            pers = deaths - births     # (n_pts,)

            # Filter by min_persistence (differentiable soft gate)
            if self.min_persistence > 0:
                # Soft mask: sigmoid with steep slope around threshold
                mask = torch.sigmoid(
                    (pers - self.min_persistence) / (tau * 0.1)
                )  # (n_pts,)
            else:
                mask = torch.ones_like(pers)

            # Effective count (soft)
            n_eff = mask.sum().clamp(min=1.0)

            # Build requested curves, each shape (n_grid,)
            # For each grid point ν:
            #   B(ν) = sum( mask * sigmoid((birth - ν) / τ) ) / n_eff
            #   D(ν) = sum( mask * sigmoid((death - ν) / τ) ) / n_eff
            #   P(ν) = sum( mask * sigmoid((pers  - ν) / τ) ) / n_eff
            #   b(ν) = sum( mask * sigmoid((ν - birth) / τ) * sigmoid((death - ν) / τ) ) / n_eff
            #
            # Vectorised: values (n_pts,) vs grid (n_grid,) → (n_pts, n_grid)

            curves_list = []
            for cname in self.curve_names:
                if cname == "B":
                    # B(ν) = P(birth > ν) ≈ sum sigmoid((birth - ν) / τ) / N
                    diff = births.unsqueeze(1) - grid.unsqueeze(0)  # (n_pts, n_grid)
                    soft_counts = torch.sigmoid(diff / tau)          # (n_pts, n_grid)
                    curve = (mask.unsqueeze(1) * soft_counts).sum(0) / n_eff

                elif cname == "D":
                    diff = deaths.unsqueeze(1) - grid.unsqueeze(0)
                    soft_counts = torch.sigmoid(diff / tau)
                    curve = (mask.unsqueeze(1) * soft_counts).sum(0) / n_eff

                elif cname == "P":
                    diff = pers.unsqueeze(1) - grid.unsqueeze(0)
                    soft_counts = torch.sigmoid(diff / tau)
                    curve = (mask.unsqueeze(1) * soft_counts).sum(0) / n_eff

                elif cname == "b":
                    # Betti: b(ν) = P(birth ≤ ν AND death > ν)
                    #       ≈ sum sigmoid((ν - birth)/τ) * sigmoid((death - ν)/τ) / N
                    born = torch.sigmoid(
                        (grid.unsqueeze(0) - births.unsqueeze(1)) / tau
                    )  # P(birth ≤ ν)
                    alive = torch.sigmoid(
                        (deaths.unsqueeze(1) - grid.unsqueeze(0)) / tau
                    )  # P(death > ν)
                    curve = (mask.unsqueeze(1) * born * alive).sum(0) / n_eff

                curves_list.append(curve)

            out[i] = torch.cat(curves_list, dim=0)

        return out

    def __repr__(self):
        temp_str = (
            f"τ(learnable)={self.get_temperature():.4f}"
            if self.learn_temperature
            else f"τ={self.get_temperature():.6f}"
        )
        return (
            f"DifferentiablePersistenceCurveLayer("
            f"n_grid={self.n_grid}, curves={self.curve_names}, "
            f"features={self.n_features}, {temp_str})"
        )
