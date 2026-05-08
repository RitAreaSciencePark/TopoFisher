"""
Learnable dense-input filtration: distance matrix → MLP → learned distances → Rips.

The MLP receives the **full pairwise distance row** for each vertex
(n features) and outputs a new distance row of the same size.  The
collection of output rows forms a learned distance matrix D', which is
symmetrised, made non-negative, and fed to a Rips filtration.

Pipeline:
    Point Cloud → full distance matrix  D(n,n)
                  → MLP(D[i,:]) per vertex → D'_raw(n, n)
                  → symmetrise + softplus + zero diagonal → D'(n, n)
                  → kNN sparsify (k_persistence) → sparse D'
                  → Ripser → persistence diagrams

Differentiability:
    Ripser computes the combinatorial pairing (which edge kills which
    cycle).  We then *gather* the differentiable learned distances D'[i,j]
    at the generator edge indices → birth/death values carry gradients
    back through the MLP.

Key differences from LearnableFastEdgeFiltration:
    ┌───────────────┬──────────────────────┬──────────────────────────┐
    │               │ FastEdge             │ DensePoint (this)        │
    ├───────────────┼──────────────────────┼──────────────────────────┤
    │ MLP input     │ kNN dists (n, k)     │ full dist row (n, n)     │
    │ MLP output    │ embedding → ||·||    │ learned dist row (n, n)  │
    │ Edge weight   │ ||emb_i − emb_j||    │ D'[i,j] directly        │
    │ kNN graph     │ fixed (original pts) │ recomputed on D' each fwd│
    │ Topology cache│ yes (pts unchanged)  │ no (D' changes)          │
    └───────────────┴──────────────────────┴──────────────────────────┘
"""

import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module-level worker functions for multiprocessing.Pool
# ---------------------------------------------------------------------------

def _mp_ripser_worker_init():
    """Import ripser once per worker process."""
    import ripser
    global _mp_ripser_mod
    _mp_ripser_mod = ripser


def _mp_ripser_call(args):
    """Compute Ripser persistence for one sparse matrix.

    Parameters packed as tuple for pool.map:
        (sparse_matrix, maxdim, thresh)
    """
    sparse_matrix, maxdim, thresh = args
    return _mp_ripser_mod.ripser(
        sparse_matrix, maxdim=maxdim, thresh=thresh, distance_matrix=True
    )


# ---------------------------------------------------------------------------
# Distance MLP
# ---------------------------------------------------------------------------

class DenseDistanceMLP(nn.Module):
    """MLP that maps full distance rows → learned distance rows.

    Input : (n_points, n_points) — one full distance row per vertex
    Output: (n_points, n_points) — learned distance row per vertex

    Uses LazyLinear for the first and last layers so that the network
    auto-adapts to the number of points on first forward pass.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        self._output_layer_built = False
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers: list = []
        if not self.hidden_dims:
            # Single lazy layer: n → n (resolved on first forward)
            layers.append(nn.LazyLinear(1))  # placeholder output dim
        else:
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            layers.append(nn.LeakyReLU(0.01))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))
            for i in range(len(self.hidden_dims) - 1):
                layers.append(
                    nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
                )
                layers.append(nn.LeakyReLU(0.01))
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))
            # Final layer: last hidden → n_points (placeholder, fixed on first call)
            layers.append(nn.LazyLinear(1))  # placeholder
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (*, n_points) distance row
        Returns: (*, n_points) learned distance row (raw, before softplus)
        """
        n_points = x.shape[-1]

        # On first call, replace the last LazyLinear placeholder with correct output dim
        if not self._output_layer_built:
            # Trigger lazy init of first layer
            with torch.no_grad():
                _ = self.network(x[:1])

            # Now replace last layer with correct output dim
            last_idx = len(self.network) - 1
            last_layer = self.network[last_idx]
            if hasattr(last_layer, 'in_features'):
                in_f = last_layer.in_features
            else:
                # LazyLinear was materialised — get its in_features
                in_f = last_layer.weight.shape[1]
            new_layer = nn.Linear(in_f, n_points).to(x.device)
            self.network[last_idx] = new_layer
            self._output_layer_built = True

        return self.network(x)


# ---------------------------------------------------------------------------
# Main filtration class
# ---------------------------------------------------------------------------

class LearnableDensePointFiltration(nn.Module):
    """
    Learnable filtration: full distance matrix → MLP → learned D' → Rips.

    The MLP transforms each distance row independently, then we symmetrise,
    apply softplus for non-negativity, zero the diagonal, sparsify via kNN,
    and run Ripser.

    Parameters
    ----------
    hidden_dims : list[int] | None
        Hidden layer sizes for the distance MLP.
    dropout : float
        Dropout probability inside the MLP.
    homology_dimensions : list[int]
        Which Betti numbers to compute (default [0, 1]).
    k_persistence : int
        Number of nearest neighbors to keep in the learned distance matrix
        for sparse Rips.  Default 20.
    max_edge : float
        Maximum edge length for Ripser.  Default inf.
    n_workers : int
        Process-pool size for parallel Ripser calls (mp.Pool).
        0 = auto (SLURM_CPUS_PER_TASK or os.cpu_count()).
    show_progress : bool
        Show tqdm progress bar.
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        homology_dimensions: Optional[List[int]] = None,
        k_persistence: int = 20,
        max_edge: float = float("inf"),
        n_workers: int = 0,
        show_progress: bool = False,
        # Kept for config compat — unused
        embedding_dim: int = 3,
        p: float = 0.0,
        n_gudhi_workers: int = 0,
    ):
        super().__init__()

        self.homology_dimensions = homology_dimensions or [0, 1]
        self.max_hom_dim = max(self.homology_dimensions)
        self.k_persistence = k_persistence
        self.max_edge = max_edge
        self.show_progress = show_progress
        self.pre_compute = True  # flag expected by pipeline

        # Worker count
        if n_workers <= 0:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                self.n_workers = max(1, int(slurm_cpus))
            else:
                self.n_workers = max(1, os.cpu_count() or 1)
        else:
            self.n_workers = n_workers

        # Persistent mp.Pool for Ripser (holds GIL → need processes)
        self._mp_pool = None
        if self.n_workers > 1:
            self._mp_pool = mp.Pool(
                processes=self.n_workers,
                initializer=_mp_ripser_worker_init,
                maxtasksperchild=500,
            )
            # Warmup
            self._mp_pool.map(
                _mp_ripser_call,
                [(sp.coo_matrix((1, 1)), 0, float("inf"))] * self.n_workers,
            )

        # Learnable MLP: distance row → learned distance row
        self.mlp = DenseDistanceMLP(
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        # Softplus for non-negative distances
        self.softplus = nn.Softplus(beta=5.0)

    def __del__(self):
        """Clean up multiprocessing pool."""
        if getattr(self, "_mp_pool", None) is not None:
            try:
                self._mp_pool.terminate()
                self._mp_pool.join()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_learned_distances(
        self, pts: torch.Tensor
    ) -> torch.Tensor:
        """Compute learned distance matrix D' from a single point cloud.

        Returns: (n, n) — symmetric, non-negative, zero diagonal.
        """
        # Full pairwise distance matrix as MLP input
        D = torch.cdist(pts, pts)  # (n, n)

        # MLP: each row independently → learned distance row
        D_raw = self.mlp(D)  # (n, n)

        # Symmetrise
        D_sym = (D_raw + D_raw.T) / 2.0

        # Non-negative via softplus
        D_pos = self.softplus(D_sym)

        # Zero diagonal
        D_learned = D_pos - torch.diag(torch.diag(D_pos))

        return D_learned

    def _sparsify_knn(
        self, D_learned_np: np.ndarray, k: int
    ):
        """Keep only k nearest neighbors per vertex → sparse matrix + edge list.

        Returns: (sparse_mat, edges, edge_weights_np, rows, cols)
        """
        n = D_learned_np.shape[0]
        k_use = min(k, n - 1)

        # kNN indices (exclude self via inf diagonal)
        D_tmp = D_learned_np.copy()
        np.fill_diagonal(D_tmp, np.inf)
        knn_idx = np.argpartition(D_tmp, k_use, axis=1)[:, :k_use]

        # Build unique undirected edge set
        edge_set = set()
        for i in range(n):
            for j in knn_idx[i]:
                edge_set.add((min(i, j), max(i, j)))

        edges = np.array(sorted(edge_set), dtype=np.int64)
        if len(edges) == 0:
            return (
                sp.coo_matrix((n, n)),
                np.empty((0, 2), dtype=np.int64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        edge_weights_np = D_learned_np[edges[:, 0], edges[:, 1]]

        # Symmetric sparse matrix
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([edge_weights_np, edge_weights_np])
        sparse_mat = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

        return sparse_mat, edges, edge_weights_np, rows, cols

    def _extract_diagrams(
        self,
        D_learned: torch.Tensor,
        edges: np.ndarray,
        ripser_result: dict,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Match Ripser birth/death values to differentiable D'[i,j].

        Gathers differentiable learned distances at generator edge indices.
        """
        # Differentiable edge weights: D'[i,j] for each edge
        edge_idx = torch.from_numpy(edges).long().to(device)
        edge_weights = D_learned[edge_idx[:, 0], edge_idx[:, 1]]

        # Sort for searchsorted matching
        ew_np = edge_weights.detach().cpu().numpy().astype(np.float64)
        sort_order = np.argsort(ew_np)
        sorted_weights = ew_np[sort_order]

        def _match_values(values: np.ndarray) -> torch.Tensor:
            """Map Ripser distance values → differentiable edge weights."""
            idxs = np.searchsorted(sorted_weights, values)
            idxs = np.clip(idxs, 0, len(sorted_weights) - 1)
            left = np.maximum(idxs - 1, 0)
            err_cur = np.abs(sorted_weights[idxs] - values)
            err_left = np.abs(sorted_weights[left] - values)
            use_left = (err_left < err_cur) & (idxs > 0)
            idxs = np.where(use_left, left, idxs)
            orig = sort_order[idxs]
            return edge_weights[torch.from_numpy(orig).long().to(device)]

        dgms_out: List[torch.Tensor] = []
        ripser_dgms = ripser_result["dgms"]

        for hom_dim in self.homology_dimensions:
            if hom_dim >= len(ripser_dgms):
                dgms_out.append(torch.zeros((0, 2), device=device))
                continue

            raw = ripser_dgms[hom_dim]
            finite_mask = np.isfinite(raw[:, 1])
            raw = raw[finite_mask]

            if len(raw) == 0:
                dgms_out.append(torch.zeros((0, 2), device=device))
                continue

            if hom_dim == 0:
                births = torch.zeros(len(raw), device=device)
                deaths = _match_values(raw[:, 1])
                dgms_out.append(torch.stack([births, deaths], dim=-1))
            else:
                births = _match_values(raw[:, 0])
                deaths = _match_values(raw[:, 1])
                dgms_out.append(torch.stack([births, deaths], dim=-1))

        return dgms_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_single(self, data) -> List[torch.Tensor]:
        """Process one sample."""
        if isinstance(data, torch.Tensor):
            pts = data
        else:
            pts = data["pts"]

        device = pts.device
        empty = [
            torch.zeros((0, 2), device=device)
            for _ in self.homology_dimensions
        ]

        # 1. Learned distance matrix
        D_learned = self._compute_learned_distances(pts)

        # 2. Sparsify via kNN
        D_np = D_learned.detach().cpu().numpy().astype(np.float64)
        sparse_mat, edges, ew_np, rows, cols = self._sparsify_knn(
            D_np, self.k_persistence
        )

        if len(edges) == 0:
            return empty

        # 3. Ripser
        import ripser
        rips = ripser.ripser(
            sparse_mat, maxdim=self.max_hom_dim,
            thresh=self.max_edge, distance_matrix=True,
        )

        # 4. Differentiable diagram extraction
        return self._extract_diagrams(D_learned, edges, rips, device)

    def forward(self, X) -> List[List[torch.Tensor]]:
        """
        Batch forward with 3-phase parallelism:
          A. GPU – batched MLP → learned distance matrices D'
          B. CPU processes – parallel Ripser (mp.Pool)
          C. GPU – differentiable diagram extraction
        """
        # --- Prepare samples ---
        if isinstance(X, torch.Tensor):
            if X.ndim == 2:
                X = X.unsqueeze(0)
            samples = [{"pts": X[i]} for i in range(X.shape[0])]
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            samples = X
        else:
            samples = [{"pts": x} for x in X]

        n_samples = len(samples)
        all_diagrams: List[List[torch.Tensor]] = [
            [] for _ in self.homology_dimensions
        ]
        if n_samples == 0:
            return all_diagrams

        device = samples[0]["pts"].device

        # ====== Phase A: GPU – batched distance matrix + MLP ===========
        pts_list = [s["pts"] for s in samples]
        stacked = torch.stack(pts_list)  # (B, n, d_in)
        B, n_pts, _ = stacked.shape

        # Batched pairwise distances
        D_batch = torch.cdist(stacked, stacked)  # (B, n, n)

        # Batched MLP
        D_flat = D_batch.reshape(B * n_pts, n_pts)  # (B*n, n)
        D_raw_flat = self.mlp(D_flat)                # (B*n, n)
        D_raw_all = D_raw_flat.reshape(B, n_pts, n_pts)  # (B, n, n)

        # Symmetrise + softplus + zero diagonal (per sample)
        D_learned_all = []
        for b in range(B):
            D_sym = (D_raw_all[b] + D_raw_all[b].T) / 2.0
            D_pos = self.softplus(D_sym)
            D_clean = D_pos - torch.diag(torch.diag(D_pos))
            D_learned_all.append(D_clean)

        # Transfer to CPU for kNN + Ripser
        D_learned_np_all = [
            d.detach().cpu().numpy().astype(np.float64) for d in D_learned_all
        ]

        # Free intermediates
        del D_batch, D_flat, D_raw_flat, D_raw_all, stacked

        # ====== Phase B: sparsify + parallel Ripser ====================
        sparse_data = []
        for b in range(B):
            result = self._sparsify_knn(D_learned_np_all[b], self.k_persistence)
            sparse_data.append(result)

        del D_learned_np_all

        # Parallel Ripser
        non_empty_indices = [
            b for b in range(B) if len(sparse_data[b][1]) > 0
        ]

        rips_results = [None] * B
        if self._mp_pool is not None and len(non_empty_indices) > 1:
            args_list = [
                (sparse_data[b][0], self.max_hom_dim, self.max_edge)
                for b in non_empty_indices
            ]
            results = self._mp_pool.map(
                _mp_ripser_call, args_list, chunksize=1
            )
            for j, b in enumerate(non_empty_indices):
                rips_results[b] = results[j]
        else:
            import ripser
            for b in non_empty_indices:
                rips_results[b] = ripser.ripser(
                    sparse_data[b][0],
                    maxdim=self.max_hom_dim,
                    thresh=self.max_edge,
                    distance_matrix=True,
                )

        # ====== Phase C: GPU – differentiable diagram extraction =======
        for b in range(B):
            if len(sparse_data[b][1]) == 0 or rips_results[b] is None:
                for dim_idx in range(len(self.homology_dimensions)):
                    all_diagrams[dim_idx].append(
                        torch.zeros((0, 2), device=device)
                    )
            else:
                edges = sparse_data[b][1]
                dgms = self._extract_diagrams(
                    D_learned_all[b], edges, rips_results[b], device
                )
                for dim_idx, dgm in enumerate(dgms):
                    all_diagrams[dim_idx].append(dgm)

        return all_diagrams

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_parameters(self) -> int:
        total = 0
        for p in self.parameters():
            if p.requires_grad and not isinstance(p, nn.UninitializedParameter):
                total += p.numel()
        return total

    def get_topologies_cached_batch(self, X: torch.Tensor) -> List[Dict[str, Any]]:
        """Compatibility: no topology caching (D' changes every epoch)."""
        if X.ndim == 2:
            X = X.unsqueeze(0)
        return [{"pts": X[i]} for i in range(X.shape[0])]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Compatibility: no cache."""
        return {
            "size": 0,
            "max_size": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate": 0.0,
            "namespace": "dense_point_no_cache",
        }

    def clear_cache(self) -> None:
        """No-op: no cache."""
        pass
