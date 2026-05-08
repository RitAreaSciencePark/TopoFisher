"""
Top-K persistence vectorization.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import warnings


class TopKBaseLayer(nn.Module):
    """
    Base class for Top-K vectorization with automatic k selection.

    Provides common functionality for all Top-K variants including
    automatic k determination during fit().
    """

    def __init__(self, k: Optional[int] = None, pad_value: float = 0.0, verbose: bool = False):
        """
        Initialize base Top-K layer.

        Args:
            k: Number of top elements to keep. If None, will be
               automatically determined during fit() as 95% of minimum diagram size.
            pad_value: Value to use for padding if fewer than k points exist
            verbose: If True, print auto-selection messages
        """
        super().__init__()
        self.k = k
        self.k_provided = k is not None  # Track if k was explicitly provided
        self.pad_value = pad_value
        self.verbose = verbose
        # n_features will be set after k is determined
        # Both subclasses use 2k features (either k births + k deaths, or k points × 2 coords)
        self.n_features = k * 2 if k is not None else None

    def fit(self, diagrams: List[torch.Tensor]):
        """
        Automatically determine k if not provided.

        Sets k to 95% of minimum number of points across all diagrams.

        Args:
            diagrams: List of all diagrams (from all simulation sets)
        """
        if self.k_provided:
            # k was explicitly provided, skip auto-detection
            return

        # Count points in each non-empty diagram
        point_counts = []
        for dgm in diagrams:
            if dgm.shape[0] > 0:
                point_counts.append(dgm.shape[0])

        if not point_counts:
            # All diagrams are empty - set k=1 with warning
            warnings.warn(
                "All diagrams are empty. Setting k=1 as fallback.",
                RuntimeWarning
            )
            self.k = 1
        else:
            # Set k to 95% of minimum, but at least 1
            min_points = min(point_counts)
            self.k = max(1, int(0.95 * min_points))

            if self.verbose:
                print(f"{self.__class__.__name__}: Auto-selected k={self.k} "
                      f"(95% of min={min_points} points)")

        # Update n_features now that k is determined
        self.n_features = self.k * 2

        # Mark k as provided so subsequent fit() calls don't override it
        self.k_provided = True

    def _validate_k(self):
        """Check if k is set before using it."""
        if self.k is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: k is not set. "
                "Either provide k during initialization or call fit() first."
            )

    def __repr__(self):
        """String representation showing configuration."""
        if self.k is None:
            k_str = "None (not yet fitted)"
        elif not self.k_provided:
            k_str = f"{self.k} (auto-selected)"
        else:
            k_str = str(self.k)

        return f"{self.__class__.__name__}(k={k_str}, pad_value={self.pad_value})"


class TopKBirthsDeathsLayer(TopKBaseLayer):
    """
    Top-K births and deaths vectorization (independent selection).

    Selects the K largest births and K largest deaths independently,
    ignoring the persistence pairing structure.

    If k is not provided (None), it will be automatically determined during fit() as
    95% of the minimum number of points across all diagrams.
    """

    def __init__(self, k: Optional[int] = None, pad_value: float = 0.0, verbose: bool = False):
        """
        Initialize Top-K births/deaths layer.

        Args:
            k: Number of top births and top deaths to keep. If None, will be
               automatically determined during fit() as 95% of minimum diagram size.
            pad_value: Value to use for padding if fewer than k points exist
            verbose: If True, print auto-selection messages
        """
        super().__init__(k=k, pad_value=pad_value, verbose=verbose)

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Vectorize by taking top-k births and top-k deaths independently.

        Args:
            diagrams: List of diagrams, each of shape (n_points, 2)
                     where [:, 0] are births and [:, 1] are deaths

        Returns:
            Tensor of shape (len(diagrams), k * 2)
            Format: [birth₁, birth₂, ..., birthₖ, death₁, death₂, ..., deathₖ]
            Both births and deaths sorted in descending order
        """
        self._validate_k()
        features = []

        for dgm in diagrams:
            if dgm.shape[0] == 0:
                # Empty diagram - pad with pad_value (preserve device!)
                vec = torch.full((self.k * 2,), self.pad_value, device=dgm.device)
            else:
                births = dgm[:, 0]
                deaths = dgm[:, 1]

                # Get top k births (descending order)
                if births.shape[0] >= self.k:
                    top_births, _ = torch.topk(births, self.k, largest=True, sorted=True)
                else:
                    # Pad if fewer than k points
                    top_births = torch.cat([
                        births.sort(descending=True)[0],  # All births sorted
                        torch.full((self.k - births.shape[0],), self.pad_value, device=dgm.device)
                    ])

                # Get top k deaths (descending order)
                if deaths.shape[0] >= self.k:
                    top_deaths, _ = torch.topk(deaths, self.k, largest=True, sorted=True)
                else:
                    # Pad if fewer than k points
                    top_deaths = torch.cat([
                        deaths.sort(descending=True)[0],  # All deaths sorted
                        torch.full((self.k - deaths.shape[0],), self.pad_value, device=dgm.device)
                    ])

                # Concatenate: [births, deaths]
                vec = torch.cat([top_births, top_deaths])

            features.append(vec)

        return torch.stack(features)


class TopKLayer(TopKBaseLayer):
    """
    Top-K persistence vectorization.

    Selects the K most persistent points and flattens them.

    If k is not provided (None), it will be automatically determined during fit() as
    95% of the minimum number of points across all diagrams.
    """

    def __init__(self, k: Optional[int] = None, pad_value: float = 0.0, verbose: bool = False):
        """
        Initialize Top-K layer.

        Args:
            k: Number of top persistent points to keep. If None, will be
               automatically determined during fit() as 95% of minimum diagram size.
            pad_value: Value to use for padding if fewer than k points exist
            verbose: If True, print auto-selection messages
        """
        super().__init__(k=k, pad_value=pad_value, verbose=verbose)

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Vectorize by taking top-k persistent points.

        Args:
            diagrams: List of diagrams, each of shape (n_points, 2)

        Returns:
            Tensor of shape (len(diagrams), k * 2)
        """
        self._validate_k()
        features = []

        for dgm in diagrams:
            if dgm.shape[0] == 0:
                # Empty diagram - pad with zeros (preserve device!)
                vec = torch.full((self.k, 2), self.pad_value, device=dgm.device)
            else:
                # Compute persistence
                persistence = dgm[:, 1] - dgm[:, 0]

                # Get top k indices
                if dgm.shape[0] >= self.k:
                    _, top_idx = torch.topk(persistence, self.k)
                    vec = dgm[top_idx]
                else:
                    # Pad if fewer than k points
                    vec = torch.cat([
                        dgm,
                        torch.full((self.k - dgm.shape[0], 2), self.pad_value, device=dgm.device)
                    ])

            features.append(vec.flatten())

        return torch.stack(features)