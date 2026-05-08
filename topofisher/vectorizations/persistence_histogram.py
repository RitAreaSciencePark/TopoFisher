"""
Persistence Histogram vectorization.

Converts persistence diagrams to histogram data vectors by binning
birth and death (or persistence) values. This is the vectorization
scheme used in Yip et al. JCAP09(2024)034, which was shown to yield
tight Fisher constraints on cosmological parameters.

For each homological dimension, we produce:
  - B_p: histogram of birth values
  - D_p: histogram of death values

The full data vector is the concatenation [B0, D0, B1, D1, ...].
"""
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
import warnings


class PersistenceHistogramLayer(nn.Module):
    """
    Persistence Histogram for a single homology dimension.

    Converts persistence diagrams to two concatenated histograms:
    one for birth values and one for death values (or persistence values).

    Follows the interface of DifferentiablePersistenceImageLayer so it
    can be used directly with CombinedVectorization.
    """

    def __init__(
        self,
        n_bins: int = 32,
        death_type: str = "death",
        percentile_clip: float = 99.9,
        min_persistence: float = 0.0,
    ):
        """
        Initialize persistence histogram layer.

        Args:
            n_bins: Number of histogram bins for EACH of birth and death/persistence.
                    Total output dimension = 2 * n_bins.
            death_type: What to histogram for the second component:
                       "death" = death values (nu_death)
                       "persistence" = persistence values (nu_death - nu_birth)
            percentile_clip: Upper percentile for range clipping (removes outliers).
                            Uses percentile_clip for upper bound, (100 - percentile_clip)
                            for lower bound of death values.
            min_persistence: Minimum persistence to include a feature (filters noise).
        """
        super().__init__()
        self.n_bins = n_bins
        self.death_type = death_type
        self.percentile_clip = percentile_clip
        self.min_persistence = min_persistence
        self.n_features = 2 * n_bins  # birth histogram + death/persistence histogram

        # Bin edges will be set during fit
        self.register_buffer('birth_edges', torch.zeros(n_bins + 1))
        self.register_buffer('death_edges', torch.zeros(n_bins + 1))
        self.fitted = False

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """
        Accumulate birth and death values from a batch of diagrams.

        Call this multiple times, then finalize_fit() to compute bin edges.

        Args:
            diagrams: List of diagrams (each shape (n_points, 2) with birth, death)
        """
        if not hasattr(self, '_all_births'):
            self._all_births = []
            self._all_deaths = []

        for dgm in diagrams:
            if dgm.shape[0] > 0:
                dgm_np = dgm.detach().cpu().numpy()
                births = dgm_np[:, 0]
                deaths = dgm_np[:, 1]
                pers = deaths - births

                # Filter by minimum persistence
                valid = pers > self.min_persistence
                if valid.any():
                    self._all_births.append(births[valid])
                    if self.death_type == "death":
                        self._all_deaths.append(deaths[valid])
                    elif self.death_type == "persistence":
                        self._all_deaths.append(pers[valid])
                    else:
                        raise ValueError(f"Unknown death_type: {self.death_type}")

    def finalize_fit(self):
        """
        Compute bin edges from accumulated data, then clean up.

        Must be called after all partial_fit() calls and before forward().
        """
        if not hasattr(self, '_all_births') or len(self._all_births) == 0:
            warnings.warn("No data accumulated for PersistenceHistogramLayer. Using default [0, 1] range.")
            self.birth_edges = torch.linspace(0, 1, self.n_bins + 1)
            self.death_edges = torch.linspace(0, 1, self.n_bins + 1)
            self.fitted = True
            return

        all_births = np.concatenate(self._all_births)
        all_deaths = np.concatenate(self._all_deaths)

        # Compute bin edges using percentile clipping
        lo_pct = 100.0 - self.percentile_clip  # e.g. 0.1
        hi_pct = self.percentile_clip            # e.g. 99.9

        birth_lo = np.percentile(all_births, lo_pct)
        birth_hi = np.percentile(all_births, hi_pct)
        death_lo = np.percentile(all_deaths, lo_pct)
        death_hi = np.percentile(all_deaths, hi_pct)

        # Ensure non-degenerate ranges
        if birth_hi - birth_lo < 1e-8:
            birth_lo, birth_hi = birth_lo - 0.5, birth_lo + 0.5
        if death_hi - death_lo < 1e-8:
            death_lo, death_hi = death_lo - 0.5, death_lo + 0.5

        self.birth_edges = torch.linspace(float(birth_lo), float(birth_hi), self.n_bins + 1)
        self.death_edges = torch.linspace(float(death_lo), float(death_hi), self.n_bins + 1)

        print(f"  PersistenceHistogramLayer fitted: "
              f"birth [{birth_lo:.4f}, {birth_hi:.4f}], "
              f"{self.death_type} [{death_lo:.4f}, {death_hi:.4f}], "
              f"{len(all_births)} total features, {self.n_bins} bins each")

        self.fitted = True

        # Clean up accumulated data
        del self._all_births, self._all_deaths

    def fit(self, diagrams: List[torch.Tensor]):
        """
        One-shot fit: compute bin edges from a list of diagrams.

        Args:
            diagrams: List of diagrams (each shape (n_points, 2) with birth, death)
        """
        self._all_births = []
        self._all_deaths = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert persistence diagrams to histogram feature vectors.

        Args:
            diagrams: List of n_samples diagrams, each shape (n_points, 2) with (birth, death).

        Returns:
            Tensor of shape (n_samples, 2 * n_bins) — concatenated birth and death histograms.
        """
        if not self.fitted:
            raise RuntimeError("PersistenceHistogramLayer must be fitted before forward(). "
                              "Call fit() or partial_fit() + finalize_fit() first.")

        batch_size = len(diagrams)
        device = self.birth_edges.device

        # Pre-allocate output
        out = torch.zeros(batch_size, self.n_features, device=device)

        birth_lo = self.birth_edges[0].item()
        birth_hi = self.birth_edges[-1].item()
        death_lo = self.death_edges[0].item()
        death_hi = self.death_edges[-1].item()

        for i, dgm in enumerate(diagrams):
            if dgm.shape[0] == 0:
                continue

            dgm_detached = dgm.detach().cpu()
            births = dgm_detached[:, 0]
            deaths = dgm_detached[:, 1]
            pers = deaths - births

            # Filter by minimum persistence
            valid = pers > self.min_persistence
            if not valid.any():
                continue

            births = births[valid].numpy()
            if self.death_type == "death":
                d_vals = deaths[valid].numpy()
            else:
                d_vals = pers[valid].numpy()

            # Histogram births
            b_counts, _ = np.histogram(births, bins=self.birth_edges.cpu().numpy())
            # Histogram deaths/persistence
            d_counts, _ = np.histogram(d_vals, bins=self.death_edges.cpu().numpy())

            out[i, :self.n_bins] = torch.from_numpy(b_counts.astype(np.float32))
            out[i, self.n_bins:] = torch.from_numpy(d_counts.astype(np.float32))

        return out

    def __repr__(self):
        return (f"PersistenceHistogramLayer(n_bins={self.n_bins}, "
                f"death_type='{self.death_type}', "
                f"features={self.n_features})")
