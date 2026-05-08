"""
ATOL vectorization layer (Royer, Chazal, Levrard, Ike, Umeda 2021).

Set-based persistence vectorization: each diagram is mapped to a K-dim
vector by pooling Gaussian responses centred at K learnable centres
in the (birth, death) plane:

    f(D)_k  =  pool_{p∈D}  w(p) · exp( -‖p − c_k‖² / (2 σ²) ),   k = 1..K

where pool is either sum (original ATOL) or weighted mean.  The default
is mean-pooling, which normalises by Σ_p w(p) and removes the dependence
on diagram cardinality.  This is critical for MOPED compatibility: with
sum-pooling the feature magnitude scales with the number of diagram
points, causing heavy-tailed (non-Gaussian) marginals that violate the
MOPED Gaussianity assumption.

Reference: Royer et al. 2021 "ATOL: Measure Vectorization for Automatic
Topologically-Oriented Learning" (https://arxiv.org/abs/1909.13472).
"""
from typing import List, Optional
import math
import warnings
import torch
import torch.nn as nn
import numpy as np


def _torch_kmeans(points: torch.Tensor, k: int, n_iter: int = 20,
                  seed: int = 0) -> torch.Tensor:
    """k-means++ init + Lloyd updates, torch-native fallback for sklearn.

    Args:
        points: (N, 2) tensor.
        k: number of clusters.
        n_iter: Lloyd iterations.
        seed: rng seed.

    Returns:
        (k, 2) tensor of centres.
    """
    g = torch.Generator(device='cpu').manual_seed(int(seed))
    pts = points.detach().cpu()
    n = pts.shape[0]
    if n == 0:
        return torch.zeros(k, 2)
    if n <= k:
        # Pad with first-point copies if fewer points than centres.
        pad = pts[0:1].expand(k - n, -1)
        return torch.cat([pts, pad], dim=0)

    # k-means++ seeding
    idx0 = torch.randint(0, n, (1,), generator=g).item()
    centres = [pts[idx0]]
    for _ in range(k - 1):
        c = torch.stack(centres, dim=0)                       # (m, 2)
        d2 = ((pts.unsqueeze(1) - c.unsqueeze(0)) ** 2).sum(-1).min(dim=1).values
        probs = d2 / d2.sum().clamp(min=1e-12)
        idx = torch.multinomial(probs, 1, generator=g).item()
        centres.append(pts[idx])
    centres = torch.stack(centres, dim=0)                     # (k, 2)

    # Lloyd updates
    for _ in range(n_iter):
        d2 = ((pts.unsqueeze(1) - centres.unsqueeze(0)) ** 2).sum(-1)
        labels = d2.argmin(dim=1)
        new_centres = centres.clone()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centres[j] = pts[mask].mean(dim=0)
        if torch.allclose(new_centres, centres, atol=1e-6):
            break
        centres = new_centres

    return centres


class ATOLLayer(nn.Module):
    """ATOL vectorization for a single homology dimension.

    Each diagram → K-dim vector via sum of Gaussian responses at K centres
    in (birth, death) space. Optionally weighted by sqrt(persistence).
    Same interface (partial_fit / finalize_fit / forward) as the other
    vectorization layers, so plays nicely with CombinedVectorization.

    Param budget for K=8 with shared σ: 16 (centres) + 1 (σ) = 17 params.
    """

    def __init__(
        self,
        n_centers: int = 8,
        weighting: str = "persistence",
        pooling: str = "mean",
        learn_centers: bool = False,
        learn_bandwidth: bool = False,
        bandwidth_clamp_factor: float = 3.0,
        per_center_bandwidth: bool = False,
        init: str = "kmeans",
    ):
        """Initialise ATOL layer.

        Args:
            n_centers: K, number of Gaussian centres.
            weighting: "persistence" (sqrt persistence) or "uniform".
            pooling: "mean" (weighted average over points, cardinality-invariant,
                recommended for MOPED) or "sum" (original ATOL formulation).
            learn_centers: If True, centres become nn.Parameter after fit().
            learn_bandwidth: If True, σ becomes nn.Parameter (log-parameterised),
                clamped to [opt/factor, opt·factor].
            bandwidth_clamp_factor: Multiplicative range for learnable σ.
            per_center_bandwidth: If True, σ is a (K,) vector (one per centre).
                Default False (single scalar σ shared across centres) keeps the
                param budget tiny.
            init: Centre initialisation strategy: "kmeans" (sklearn or torch
                fallback) or "percentile_grid" (regular grid bracketed by
                percentile bounds).
        """
        super().__init__()
        if weighting not in ("persistence", "uniform"):
            raise ValueError(f"Unknown weighting '{weighting}'.")
        if pooling not in ("mean", "sum"):
            raise ValueError(f"Unknown pooling '{pooling}'. Use 'mean' or 'sum'.")
        if init not in ("kmeans", "percentile_grid"):
            raise ValueError(f"Unknown init '{init}'.")

        self.n_centers = int(n_centers)
        self.weighting = weighting
        self.pooling = pooling
        self.learn_centers = bool(learn_centers)
        self.learn_bandwidth = bool(learn_bandwidth)
        self.per_center_bandwidth = bool(per_center_bandwidth)
        self.bandwidth_clamp_factor = float(bandwidth_clamp_factor)
        self.init = init
        self.n_features = self.n_centers
        self.fitted = False

        # Centres start as a buffer; promoted to Parameter in fit() if learn_centers.
        self.register_buffer("centers", torch.zeros(self.n_centers, 2))

        # Bandwidth: scalar by default; (K,) if per-center.
        bw_shape = (self.n_centers,) if self.per_center_bandwidth else ()
        if self.learn_bandwidth:
            self.log_bandwidth = nn.Parameter(torch.zeros(bw_shape))
            self.register_buffer("log_bw_min", torch.tensor(float("-inf")))
            self.register_buffer("log_bw_max", torch.tensor(float("inf")))
        else:
            self.register_buffer("bandwidth_buf", torch.ones(bw_shape))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """Accumulate diagram points across batches before finalize_fit()."""
        if not hasattr(self, "_pool"):
            self._pool = []
        for dgm in diagrams:
            if dgm.shape[0] > 0:
                pts = dgm.detach().cpu()
                pers = pts[:, 1] - pts[:, 0]
                valid = pers > 0
                if valid.any():
                    self._pool.append(pts[valid])

    def finalize_fit(self):
        """Compute centres + bandwidth from accumulated points."""
        if not hasattr(self, "_pool") or len(self._pool) == 0:
            warnings.warn(
                "ATOLLayer: no valid diagram points; using default unit-square "
                "centres and σ=0.1.", RuntimeWarning,
            )
            grid_lin = torch.linspace(0.1, 0.9, max(self.n_centers, 1))
            centers = torch.stack([grid_lin, grid_lin + 0.1], dim=1)[: self.n_centers]
            self._set_centers(centers)
            self._set_bandwidth(0.1)
            self.fitted = True
            self._cleanup_pool()
            return

        pool = torch.cat(self._pool, dim=0)  # (N_total, 2)
        if self.init == "kmeans":
            centers = self._init_kmeans(pool)
        else:
            centers = self._init_percentile_grid(pool)

        # Bandwidth from median nearest-centre distance (analogous to
        # DiffPI's grid_spacing/2).
        d2 = ((centers.unsqueeze(0) - centers.unsqueeze(1)) ** 2).sum(-1)
        d2 = d2 + torch.eye(self.n_centers) * 1e9
        nn_dists = d2.min(dim=1).values.clamp(min=1e-8).sqrt()
        sigma_opt = float(nn_dists.median().item()) / 2.0
        sigma_opt = max(sigma_opt, 1e-4)

        self._set_centers(centers)
        self._set_bandwidth(sigma_opt)
        self.fitted = True
        self._cleanup_pool()

        print(
            f"  ATOLLayer fitted: K={self.n_centers}, "
            f"σ={sigma_opt:.4f} ({'learnable' if self.learn_bandwidth else 'fixed'}), "
            f"centers ∈ [{centers.min().item():.3f}, {centers.max().item():.3f}], "
            f"{pool.shape[0]} pooled points"
        )

    def fit(self, diagrams: List[torch.Tensor]):
        """One-shot fit on a flat list of diagrams."""
        self._pool = []
        self.partial_fit(diagrams)
        self.finalize_fit()

    def _cleanup_pool(self):
        if hasattr(self, "_pool"):
            del self._pool

    def _init_kmeans(self, pool: torch.Tensor) -> torch.Tensor:
        try:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(
                n_clusters=self.n_centers,
                init="k-means++",
                n_init=3,
                random_state=0,
                batch_size=min(4096, pool.shape[0]),
            )
            km.fit(pool.numpy())
            return torch.from_numpy(km.cluster_centers_).float()
        except Exception:
            return _torch_kmeans(pool, self.n_centers).float()

    def _init_percentile_grid(self, pool: torch.Tensor) -> torch.Tensor:
        """Place centres on a regular grid in the (birth, death) box."""
        b_lo, b_hi = float(np.percentile(pool[:, 0].numpy(), 1)), float(
            np.percentile(pool[:, 0].numpy(), 99)
        )
        d_lo, d_hi = float(np.percentile(pool[:, 1].numpy(), 1)), float(
            np.percentile(pool[:, 1].numpy(), 99)
        )
        # Pick (a, b) ≈ √K for grid; pad with edge-points if a*b > K.
        a = int(round(math.sqrt(self.n_centers)))
        b = int(math.ceil(self.n_centers / max(a, 1)))
        bs = torch.linspace(b_lo, b_hi, max(a, 1))
        ds = torch.linspace(d_lo, d_hi, max(b, 1))
        bb, dd = torch.meshgrid(bs, ds, indexing="ij")
        pts = torch.stack([bb.reshape(-1), dd.reshape(-1)], dim=1)
        return pts[: self.n_centers]

    def _set_centers(self, centers: torch.Tensor):
        if self.learn_centers:
            # Replace buffer with a Parameter
            del self.centers
            self.centers = nn.Parameter(centers.clone().float())
        else:
            self.centers.copy_(centers.float())

    def _set_bandwidth(self, sigma_opt: float):
        target = torch.full(
            (self.n_centers,) if self.per_center_bandwidth else (),
            float(sigma_opt),
        )
        if self.learn_bandwidth:
            with torch.no_grad():
                self.log_bandwidth.fill_(math.log(sigma_opt))
            log_factor = math.log(self.bandwidth_clamp_factor)
            self.log_bw_min = torch.tensor(math.log(sigma_opt) - log_factor)
            self.log_bw_max = torch.tensor(math.log(sigma_opt) + log_factor)
        else:
            self.bandwidth_buf.copy_(target)

    def get_bandwidth(self):
        if self.learn_bandwidth:
            log_sigma = self.log_bandwidth.clamp(self.log_bw_min, self.log_bw_max)
            return log_sigma.exp()
        return self.bandwidth_buf

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """Vectorize a batch of diagrams to (n_samples, K).

        Uses padded-batch tensor + mask, same as DiffPI.
        """
        if not self.fitted:
            raise RuntimeError(
                "ATOLLayer must be fitted before forward(). Call .fit() first."
            )

        batch_size = len(diagrams)
        if batch_size == 0:
            return torch.empty((0, self.n_features))

        # Device from first non-empty diagram, fallback to centers.
        device = self.centers.device
        for dgm in diagrams:
            if dgm.numel() > 0:
                device = dgm.device
                break

        n_pairs_list = [int(dgm.shape[0]) for dgm in diagrams]
        max_K = max(n_pairs_list) if n_pairs_list else 0
        if max_K == 0:
            return torch.zeros((batch_size, self.n_features), device=device)

        # Replace empty diagrams with a (1, 2) zero so pad_sequence works.
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
        pers_all = (dgm_pad[:, :, 1] - dgm_pad[:, :, 0])                # (B, M)
        valid_mask = original_mask & (pers_all > 0)                     # (B, M)
        mask = valid_mask.float()                                       # (B, M)

        # Centres on the right device
        centers = self.centers.to(device)                               # (K, 2)
        K = self.n_centers

        # Pairwise squared distances (B, M, K)
        diff = dgm_pad.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)  # (B, M, K, 2)
        sq = (diff ** 2).sum(-1)                                         # (B, M, K)

        # Bandwidth tensor (K,) or scalar
        sigma = self.get_bandwidth().to(device)
        if sigma.dim() == 0:
            denom = 2.0 * sigma * sigma
            gauss = torch.exp(-sq / denom)                               # (B, M, K)
        else:
            denom = 2.0 * (sigma * sigma).unsqueeze(0).unsqueeze(0)      # (1, 1, K)
            gauss = torch.exp(-sq / denom)

        # Persistence weight (B, M)
        if self.weighting == "persistence":
            weights = (pers_all.clamp(min=0.0) + 1e-12).sqrt() * mask
        else:
            weights = mask

        # Pool over diagram points: (B, M, 1) * (B, M, K) → (B, K)
        out = (weights.unsqueeze(-1) * gauss).sum(dim=1)            # (B, K)
        if self.pooling == "mean":
            # Normalise by total weight to remove cardinality dependence.
            # This keeps features in [0, 1] and gives near-Gaussian marginals
            # under MOPED, since we're averaging rather than summing.
            total_w = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)  # (B, 1)
            out = out / total_w
        return out

    def __repr__(self):
        fitted_str = "fitted" if self.fitted else "not fitted"
        params_str = (
            f"learn_centers={self.learn_centers}, "
            f"learn_bandwidth={self.learn_bandwidth}, "
            f"per_center_bw={self.per_center_bandwidth}"
        )
        return (
            f"ATOLLayer(n_centers={self.n_centers}, weighting='{self.weighting}', "
            f"pooling='{self.pooling}', {params_str}, {fitted_str})"
        )
