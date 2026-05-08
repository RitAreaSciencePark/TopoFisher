"""
PersLay — Permutation-invariant neural network vectorization of persistence diagrams.

Reference: Carrière, Chazal, Ike, Lacombe, Royer, Umeda (AISTATS 2020)
            "PersLay: A Neural Network Layer for Persistence Diagrams and
             New Graph Topological Signatures."

Architecture:
    V(D) = ρ( Σ_p  w(p) · φ(x(p)) )

where for each point p = (b, d) with persistence π = d − b:
  - x(p) = (b, d, π, log(π + ε))             4-dim per-point feature
  - φ : R⁴ → R^h                              spectral-normalized 2-layer MLP
  - w(p) = π · sigmoid(u·b + v·d + b₀)        diagonal-vanishing weight
  - Σ_p                                       sum-pool (permutation-invariant)
  - ρ : R^h → R^m                             optional post-pool projection

Every existing vectorization (PI, DiffCurves, Silhouette, Landscape, ATOL) is a
special case under this parameterization, so the optimizer can recover any of
them or interpolate.

W₁-Lipschitz on finite diagrams when:
  (i)  φ is Lipschitz                       — enforced via spectral_norm
  (ii) w(p) → 0 as π → 0                    — guaranteed by the leading π factor
  (iii) sum pooling                         — chosen
"""
from typing import List, Optional
import math
import warnings

import torch
import torch.nn as nn


class PersLayLayer(nn.Module):
    """
    PersLay vectorization for a single homology dimension.

    Interface matches the rest of `topofisher.vectorizations`: takes a list of
    diagrams `[Tensor(N_i, 2), ...]` per sample and returns a `(B, output_dim)`
    feature tensor. Composable inside `CombinedVectorization` for multiple
    homology dimensions.
    """

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        post_pool_dim: int = 0,
        learn_features: bool = True,
        spectral_norm: bool = True,
        log_persistence_eps: float = 1e-6,
        outlier_quantile: float = 0.999,
    ):
        """
        Args:
            embed_dim: Output dimension of φ (h). Per-H-dim feature size when
                post_pool_dim=0. Default 16; spec recommends 16 to start.
            hidden_dim: Hidden width of the 2-layer φ MLP. Default 32.
            post_pool_dim: If > 0, append a Linear(embed_dim → post_pool_dim) (ρ).
                Default 0 → identity (recommended when MOPED follows).
            learn_features: If False, freeze all parameters (smoke-test baseline
                using random features). The factory routes this from `trainable`.
            spectral_norm: Wrap each Linear in φ with spectral_norm parametrization
                so ‖W‖₂ ≤ 1, giving a constructive Lipschitz bound on φ.
            log_persistence_eps: ε in log(π + ε) to stabilise the log-persistence
                input feature near the diagonal.
            outlier_quantile: Filter pairs with persistence above this quantile
                of the training-pool persistence distribution. Defaults to 0.999
                — drops the top 0.1 % of pairs, which catches "essential"
                topological classes (death = field max) emitted by the GPU
                gudhi backend but not by CPU gudhi. Setting to 1.0 disables
                filtering. The threshold is computed in `finalize_fit` and
                stored as a buffer; mean/std are also fitted excluding these
                outliers so warm-start across CPU↔GPU backends is consistent.
        """
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.post_pool_dim = int(post_pool_dim)
        self.learn_features = bool(learn_features)
        self.use_spectral_norm = bool(spectral_norm)
        self.eps = float(log_persistence_eps)
        self.outlier_quantile = float(outlier_quantile)

        # Input normalization buffers — fitted on the training pool.
        # Use register_buffer so they move with .to(device) but are not learnable.
        # finalize_fit() updates them via copy_() to preserve buffer registration
        # (lesson from the DiffCurves device bug fix).
        self.register_buffer('input_mean', torch.zeros(4))
        self.register_buffer('input_std',  torch.ones(4))
        # Persistence threshold above which pairs are treated as outliers and
        # masked out in forward. Set in finalize_fit; +inf disables filtering.
        self.register_buffer('pers_threshold', torch.tensor(float('inf')))

        # φ: 4 → hidden_dim → embed_dim, LeakyReLU(0.01) between.
        self.phi = nn.Sequential(
            self._linear(4, self.hidden_dim, self.use_spectral_norm),
            nn.LeakyReLU(0.01),
            self._linear(self.hidden_dim, self.embed_dim, self.use_spectral_norm),
        )

        # w(p) = π · sigmoid(u·b + v·d + b₀)
        # Bias initialised to 0 so sigmoid ≈ 0.5 at start → w ≈ π/2 (matches
        # the persistence-weighting prior used by DiffPI / ATOL).
        self.w_logit = nn.Linear(2, 1)
        nn.init.zeros_(self.w_logit.bias)

        # ρ — optional post-pool linear projection.
        if self.post_pool_dim > 0:
            self.rho = nn.Linear(self.embed_dim, self.post_pool_dim)
            self.output_dim = self.post_pool_dim
        else:
            self.rho = nn.Identity()
            self.output_dim = self.embed_dim

        # n_features mirrors the convention of the other layers (used by combined.py).
        self.n_features = self.output_dim

        if not self.learn_features:
            for p in self.parameters():
                p.requires_grad = False

        self.fitted = False

    @staticmethod
    def _linear(d_in: int, d_out: int, use_sn: bool) -> nn.Module:
        layer = nn.Linear(d_in, d_out)
        if use_sn:
            from torch.nn.utils.parametrizations import spectral_norm
            layer = spectral_norm(layer)
        return layer

    # ------------------------------------------------------------------
    # Fitting — accumulate input-feature stats for standardization.
    # ------------------------------------------------------------------

    def partial_fit(self, diagrams: List[torch.Tensor]):
        """
        Accumulate (b, d, π, log(π+ε)) statistics from a batch of diagrams.

        Call multiple times (one per dataset), then `finalize_fit()`.
        Same streaming API as DiffPI / Silhouette / Landscape.
        """
        if not hasattr(self, '_pool'):
            self._pool = []  # list of (N_i, 4) numpy-like tensors on CPU

        for dgm in diagrams:
            if dgm.shape[0] > 0:
                d_cpu = dgm.detach().cpu()
                b = d_cpu[:, 0]
                d = d_cpu[:, 1]
                pers = d - b
                valid = pers > 0
                if valid.any():
                    bv = b[valid]
                    dv = d[valid]
                    pv = pers[valid]
                    lpv = torch.log(pv + self.eps)
                    feats = torch.stack([bv, dv, pv, lpv], dim=1)  # (N, 4)
                    self._pool.append(feats)

    def finalize_fit(self):
        """Compute per-feature mean/std and write into the buffers in-place."""
        if not hasattr(self, '_pool') or len(self._pool) == 0:
            warnings.warn(
                "PersLayLayer: no valid points seen during fit; "
                "using default zero-mean / unit-std normalization.",
                RuntimeWarning, stacklevel=2,
            )
            self.input_mean.zero_()
            self.input_std.fill_(1.0)
            self.fitted = True
            self._cleanup_pool()
            return

        all_feats = torch.cat(self._pool, dim=0)  # (N_total, 4) — (b, d, π, logπ)

        # Compute persistence outlier threshold from the pooled distribution.
        # Pairs with persistence above this quantile are likely "essential"
        # topological classes (death = field max) returned by the GPU gudhi
        # backend but suppressed by the CPU gudhi backend used in
        # perslay_learnable_moped — they are extreme outliers (~50× typical
        # persistence) that inflate input_std and break warm-start.
        if 0.0 < self.outlier_quantile < 1.0:
            # numpy.quantile to avoid torch.quantile's ~16M-element size limit
            # (training pools at 512² are ~60M pairs).
            import numpy as np
            pers_all = all_feats[:, 2].numpy()  # column 2 is persistence π
            pers_thresh = float(np.quantile(pers_all, self.outlier_quantile))
        else:
            pers_thresh = float('inf')
        self.pers_threshold.fill_(pers_thresh)

        # Refit mean/std EXCLUDING outlier pairs so the stats represent the
        # bulk distribution PersLay's φ will actually see at forward time.
        keep = all_feats[:, 2] <= pers_thresh  # (N_total,)
        kept_feats = all_feats[keep]
        n_dropped = all_feats.shape[0] - kept_feats.shape[0]
        if kept_feats.shape[0] >= 2:
            mean = kept_feats.mean(dim=0)
            std = kept_feats.std(dim=0).clamp_min(1e-6)
        else:
            # Pathological — fall back to the unfiltered statistics.
            mean = all_feats.mean(dim=0)
            std = all_feats.std(dim=0).clamp_min(1e-6)

        # In-place update preserves register_buffer + device tracking.
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
        self.fitted = True

        warnings.warn(
            f"PersLayLayer fitted on {all_feats.shape[0]} points "
            f"(dropped {n_dropped} outliers above π={pers_thresh:.4g}, "
            f"q={self.outlier_quantile}): "
            f"mean={mean.tolist()}, std={std.tolist()}",
            stacklevel=2,
        )
        self._cleanup_pool()

    def fit(self, diagrams: List[torch.Tensor]):
        """One-shot wrapper: partial_fit + finalize_fit."""
        self.partial_fit(diagrams)
        self.finalize_fit()

    def renormalize(self, diagrams: List[torch.Tensor]):
        """Update input_mean/input_std from the current diagram distribution.

        Called periodically during joint CNN+PersLay training so normalization
        tracks the evolving filtration output. pers_threshold is NOT updated —
        the outlier filter is fixed at fit time and should stay stable.
        """
        all_feats = []
        for dgm in diagrams:
            if dgm is None or dgm.numel() == 0:
                continue
            b, d = dgm[:, 0], dgm[:, 1]
            pi = (d - b).clamp_min(0.0)
            valid = (pi > 0) & (pi <= self.pers_threshold.item())
            if valid.sum() < 2:
                continue
            b, d, pi = b[valid], d[valid], pi[valid]
            log_pi = torch.log(pi + self.eps)
            all_feats.append(
                torch.stack([b, d, pi, log_pi], dim=-1).detach().cpu()
            )

        if len(all_feats) < 10:
            return

        all_feats = torch.cat(all_feats, dim=0)
        mean = all_feats.mean(dim=0)
        std = all_feats.std(dim=0).clamp_min(1e-8)
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_std.copy_(std.to(self.input_std.device))

    def _cleanup_pool(self):
        if hasattr(self, '_pool'):
            del self._pool

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, diagrams: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            diagrams: List of (N_i, 2) tensors, one per sample, possibly empty.

        Returns:
            Tensor (B, output_dim) — sum-pooled per-diagram features.
        """
        if not self.fitted:
            raise RuntimeError(
                "PersLayLayer must be fitted before forward(). "
                "Call .fit() or use partial_fit/finalize_fit."
            )

        batch_size = len(diagrams)
        if batch_size == 0:
            return torch.empty((0, self.output_dim), device=self.input_mean.device)

        # Chunk large batches: the padded tensor [B, max_K, hidden_dim] is
        # O(B × max_K × hidden_dim). For B=20000, max_K~5000, hidden_dim=64
        # this reaches ~25 GB per set × 5 sets = OOM on 384 GB nodes.
        # Chunk to keep peak memory ≈ chunk_size × max_K × hidden_dim ≈ 500 MB.
        _CHUNK = 500
        if batch_size > _CHUNK:
            return torch.cat(
                [self.forward(diagrams[i: i + _CHUNK])
                 for i in range(0, batch_size, _CHUNK)],
                dim=0,
            )

        # Device from first non-empty diagram, fallback to input_mean buffer.
        device = self.input_mean.device
        for dgm in diagrams:
            if dgm.numel() > 0:
                device = dgm.device
                break

        # Padded batch + mask (mirrors DiffPI / ATOL pattern).
        n_pairs_list = [int(dgm.shape[0]) for dgm in diagrams]
        max_K = max(n_pairs_list) if n_pairs_list else 0
        if max_K == 0:
            return torch.zeros((batch_size, self.output_dim), device=device)

        diagrams_safe = [
            dgm if dgm.shape[0] > 0
            else torch.zeros((1, 2), device=device, dtype=torch.float32)
            for dgm in diagrams
        ]
        n_pairs = torch.tensor(n_pairs_list, device=device)
        dgm_pad = torch.nn.utils.rnn.pad_sequence(
            diagrams_safe, batch_first=True, padding_value=0.0
        )  # (B, M, 2)
        M = dgm_pad.shape[1]

        idx_range = torch.arange(M, device=device).unsqueeze(0)        # (1, M)
        original_mask = idx_range < n_pairs.unsqueeze(1)               # (B, M)
        births = dgm_pad[..., 0]                                       # (B, M)
        deaths = dgm_pad[..., 1]                                       # (B, M)
        pers = deaths - births                                         # (B, M)
        # Outlier filter: drop pairs whose persistence exceeds the fitted
        # threshold (catches "essential" pairs from the GPU gudhi backend).
        # +inf threshold (default before fit) is a no-op.
        not_outlier = pers <= self.pers_threshold.to(device)            # (B, M)
        valid_mask = original_mask & (pers > 0) & not_outlier          # (B, M)
        mask = valid_mask.float()                                      # (B, M)

        # Build per-point feature x(p) = (b, d, π, log(π + ε)).
        # clamp_min(0) on pers for the log to avoid -inf on padded zeros (which
        # are masked out anyway); the eps additionally stabilises near-diagonal.
        log_pers = torch.log(pers.clamp_min(0.0) + self.eps)           # (B, M)
        features = torch.stack([births, deaths, pers, log_pers], dim=-1)  # (B, M, 4)

        # Standardize using fitted stats (broadcasts over B, M).
        # .to(device) is a no-op when already on the right device.
        features = (features - self.input_mean.to(device)) / self.input_std.to(device)

        # φ embedding.
        # phi_device may differ from `device` when the module was loaded from
        # a CPU checkpoint via load_state_dict(map_location='cpu') and
        # spectral_norm parametrizations were not moved back to GPU by to().
        # Moving features→phi_device for the MLP, then output back to `device`,
        # handles both the matched (no-op .to()) and mismatched cases.
        phi_device = next(self.phi.parameters()).device
        phi_out = self.phi(features.to(phi_device)).to(device)         # (B, M, h)

        # w(p) = π · sigmoid(u·b + v·d + b₀); use raw (un-standardized) (b, d).
        w_logits = self.w_logit(dgm_pad.to(phi_device)).squeeze(-1).to(device)  # (B, M)
        w = pers.clamp_min(0.0) * torch.sigmoid(w_logits)              # (B, M)
        w = w * mask                                                   # zero invalid points

        # Sum pool over diagram points.
        pooled = (phi_out * w.unsqueeze(-1)).sum(dim=1)                # (B, h)

        # ρ (Linear or Identity). If Linear, apply same device-routing as phi.
        rho_param = next(self.rho.parameters(), None)
        if rho_param is not None and rho_param.device != pooled.device:
            return self.rho(pooled.to(rho_param.device)).to(device)
        return self.rho(pooled)                                        # (B, output_dim)

    def get_num_parameters(self) -> int:
        """Total number of learnable parameters (matches existing convention)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        fitted_str = "fitted" if self.fitted else "not fitted"
        return (
            f"PersLayLayer(embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}, "
            f"post_pool_dim={self.post_pool_dim}, learn_features={self.learn_features}, "
            f"spectral_norm={self.use_spectral_norm}, "
            f"params={self.get_num_parameters()}, {fitted_str})"
        )
