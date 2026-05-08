"""
Persistence Curve vectorization (Empirical Distribution Functions).

Converts persistence diagrams to 1D curves following Biagetti, Cole & Shiu
(arXiv:2009.04819). For each persistence diagram, we compute:

  B_p(ν) = count_p(ν_birth > ν) / N_p        — complementary CDF of births
  D_p(ν) = count_p(ν_death > ν) / N_p         — complementary CDF of deaths
  P_p(ν) = count_p(ν_persist > ν) / N_p       — complementary CDF of persistences
  b_p(ν) = (-B_p(ν) + D_p(ν))                 — (normalized) Betti curve

Each curve is evaluated at n_grid linearly spaced filtration values. After
normalization, these are empirical distribution functions that interpolate
between 1 and 0, avoiding the sparsity/Gaussianity issues of bin-count
histograms.

The full data vector for a single (tomo_bin, hom_dim) is the concatenation
of the selected curves.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
import warnings


class PersistenceCurveLayer(nn.Module):
    """
    Empirical distribution function curves from persistence diagrams.

    For a single homology dimension, computes 1D curves by counting persistence
    diagram features in specific regions, evaluated on a fixed grid.
    Output is the concatenated selected curves.

    Follows the partial_fit / finalize_fit / forward interface so it can be
    used with CombinedVectorization.
    """

    def __init__(
        self,
        n_grid: int = 100,
        curves: str = "B,D,P",
        min_persistence: float = 0.0,
        percentile_clip: float = 99.9,
    ):
        """
        Initialize persistence curve layer.

        Args:
            n_grid: Number of evaluation points per curve. Paper used 1000 for
                    final analysis, 50 for visualization. For NN compression
                    100–200 is usually sufficient.
            curves: Comma-separated list of curves to compute.
                    B = birth complementary-CDF
                    D = death complementary-CDF
                    P = persistence complementary-CDF
                    b = Betti curve (normalized)
                    Default: "B,D,P" (3 curves, output dim = 3 * n_grid)
            min_persistence: Minimum persistence to include a feature.
            percentile_clip: Upper percentile for grid range determination.
                            Grid spans from (100-percentile_clip)-th to
                            percentile_clip-th percentile of ALL filtration
                            values (births and deaths pooled).
        """
        super().__init__()
        self.n_grid = n_grid
        self.curve_names = [c.strip() for c in curves.split(",")]
        self.min_persistence = min_persistence
        self.percentile_clip = percentile_clip

        # Validate curve names
        valid = {"B", "D", "P", "b"}
        for c in self.curve_names:
            if c not in valid:
                raise ValueError(
                    f"Unknown curve '{c}'. Valid: {valid}"
                )

        self.n_curves = len(self.curve_names)
        self.n_features = self.n_curves * n_grid

        # Grid will be set during fit
        self.register_buffer("grid", torch.zeros(n_grid))
        self.fitted = False

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """
        Accumulate filtration value ranges from a batch of diagrams.

        Args:
            diagrams: List of diagrams, each (n_points, 2) with (birth, death).
        """
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
                    # Pool births, deaths, and persistence for range estimation
                    # Grid covers filtration scale, so use births + deaths
                    self._all_vals.append(births[valid])
                    self._all_vals.append(deaths[valid])

    def finalize_fit(self):
        """
        Compute the evaluation grid from accumulated data.
        """
        if not hasattr(self, "_all_vals") or len(self._all_vals) == 0:
            warnings.warn(
                "No data accumulated for PersistenceCurveLayer. "
                "Using default [0, 1] grid."
            )
            self.grid = torch.linspace(0, 1, self.n_grid)
            self.fitted = True
            return

        all_vals = np.concatenate(self._all_vals)

        lo_pct = 100.0 - self.percentile_clip
        hi_pct = self.percentile_clip

        v_lo = np.percentile(all_vals, lo_pct)
        v_hi = np.percentile(all_vals, hi_pct)

        # Ensure non-degenerate range
        if v_hi - v_lo < 1e-8:
            v_lo, v_hi = v_lo - 0.5, v_lo + 0.5

        self.grid = torch.linspace(float(v_lo), float(v_hi), self.n_grid)

        print(
            f"  PersistenceCurveLayer fitted: "
            f"grid [{v_lo:.4f}, {v_hi:.4f}], {self.n_grid} points, "
            f"curves={self.curve_names}, "
            f"{len(all_vals)} pooled filtration values"
        )

        self.fitted = True
        del self._all_vals

    def fit(self, diagrams: List[torch.Tensor]):
        """One-shot fit."""
        self._all_vals = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute persistence curve features for a batch of diagrams.

        For each diagram and each selected curve, evaluates the (normalised)
        complementary CDF at every grid point.

        Args:
            diagrams: List of n_samples diagrams, each (n_points, 2).

        Returns:
            Tensor (n_samples, n_curves * n_grid).
        """
        if not self.fitted:
            raise RuntimeError(
                "PersistenceCurveLayer must be fitted before forward()."
            )

        batch_size = len(diagrams)
        device = self.grid.device
        grid_np = self.grid.cpu().numpy()  # shape (n_grid,)
        out = torch.zeros(batch_size, self.n_features, device=device)

        for i, dgm in enumerate(diagrams):
            if dgm.shape[0] == 0:
                continue

            dgm_np = dgm.detach().cpu().numpy()
            births = dgm_np[:, 0]
            deaths = dgm_np[:, 1]
            pers = deaths - births

            valid = pers > self.min_persistence
            if not valid.any():
                continue

            b = births[valid]
            d = deaths[valid]
            p = d - b
            n_total = float(len(b))

            # Build requested curves, each shape (n_grid,)
            curves_list = []
            for cname in self.curve_names:
                if cname == "B":
                    # B(ν) = count(birth > ν) / N
                    curve = np.array(
                        [(b > nu).sum() for nu in grid_np], dtype=np.float32
                    ) / n_total
                elif cname == "D":
                    # D(ν) = count(death > ν) / N
                    curve = np.array(
                        [(d > nu).sum() for nu in grid_np], dtype=np.float32
                    ) / n_total
                elif cname == "P":
                    # P(ν) = count(persist > ν) / N
                    curve = np.array(
                        [(p > nu).sum() for nu in grid_np], dtype=np.float32
                    ) / n_total
                elif cname == "b":
                    # Betti: b(ν) = count(birth ≤ ν AND death > ν) / N
                    curve = np.array(
                        [((b <= nu) & (d > nu)).sum() for nu in grid_np],
                        dtype=np.float32,
                    ) / n_total
                curves_list.append(curve)

            vec = np.concatenate(curves_list)
            out[i] = torch.from_numpy(vec)

        return out

    def __repr__(self):
        return (
            f"PersistenceCurveLayer(n_grid={self.n_grid}, "
            f"curves={self.curve_names}, features={self.n_features})"
        )
