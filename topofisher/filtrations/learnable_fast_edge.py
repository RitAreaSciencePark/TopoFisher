"""
Fast learnable edge filtration using Ripser for persistence computation.

Speed-optimised pipeline that replaces GUDHI with Ripser (C++ Vietoris-Rips)
and parallelises per-sample persistence via multiprocessing.Pool.

Pipeline:
    Point Cloud → kNN distances → MLP → Embedding Space → Pairwise Distances
                                                                     ↓
                     Sparse Distance Matrix → Ripser (parallel, C++)
                                                                     ↓
                              Value-matched Differentiable Diagrams

Speed advantages over the GUDHI-based LearnableEdgeFiltration:
  - Ripser sparse Rips: ~5 ms/sample vs ~15+ ms for GUDHI simplex tree
  - No simplex tree construction or flag expansion
  - GPU-native kNN graph computation (torch.cdist)
  - Parallel Ripser via mp.Pool (~15x speedup; Ripser holds the GIL,
    so ThreadPoolExecutor gives zero parallelism)
  - Zero GUDHI dependency

Differentiability:
  Ripser is used only for the combinatorial *pairing* (which simplices
  cause births/deaths).  The actual birth/death *values* are recovered by
  matching ripser's output back to entries of the differentiable edge-weight
  tensor, so gradients flow:
      Fisher → diagram → edge_weights → embeddings → MLP
"""

import hashlib
import multiprocessing as mp
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import ripser as _ripser_module

    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False


# ---------------------------------------------------------------------------
# Module-level worker functions for multiprocessing.Pool
# (must be picklable – cannot be methods or closures)
# ---------------------------------------------------------------------------

def _mp_ripser_worker_init():
    """Initialiser run once per worker process – imports ripser."""
    import ripser
    global _mp_ripser_mod
    _mp_ripser_mod = ripser


def _mp_ripser_call(args):
    """Compute Ripser persistence for one sparse matrix.

    Parameters are packed as a tuple to work with ``pool.map``:
        (sparse_matrix, maxdim, thresh)
    """
    sparse_matrix, maxdim, thresh = args
    return _mp_ripser_mod.ripser(
        sparse_matrix, maxdim=maxdim, thresh=thresh, distance_matrix=True
    )


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class FastEdgeEmbeddingMLP(nn.Module):
    """MLP that maps per-vertex features (kNN distances or coords) to R^d."""

    def __init__(
        self,
        embedding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        layers: list = []
        if not self.hidden_dims:
            layers.append(nn.LazyLinear(self.embedding_dim))
        else:
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            layers.append(nn.LeakyReLU(0.01))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                layers.append(nn.LeakyReLU(0.01))
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))
            layers.append(nn.Linear(self.hidden_dims[-1], self.embedding_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Main filtration class
# ---------------------------------------------------------------------------

class LearnableFastEdgeFiltration(nn.Module):
    """
    Speed-optimised learned edge filtration (Ripser backend).

    Parameters
    ----------
    k : int
        Number of nearest neighbours for the kNN skeleton.
        Controls the MLP input dimensionality (each point gets k distances).
    k_persistence : int | None
        Number of nearest neighbours used to build the edge set for
        Ripser persistence.  When ``None`` (default) it equals ``k``.
        Set this *lower* than ``k`` to decouple MLP expressiveness from
        persistence cost — the MLP still sees all ``k`` neighbour
        distances, but only the ``k_persistence``-nearest edges enter
        the filtration complex, keeping Ripser fast.
    embedding_dim : int
        Dimension of the learned embedding space.
    hidden_dims : list[int] | None
        Hidden layer sizes for the embedding MLP.
    dropout : float
        Dropout probability inside the MLP.
    homology_dimensions : list[int]
        Which Betti numbers to compute (default [0, 1]).
    max_edge : float
        Ripser ``thresh`` – ignore edges above this learned weight.
        Use ``inf`` (default) for no threshold.
    use_coords : bool
        If True the MLP receives raw point coordinates; otherwise kNN
        distances (default False).
    edge_scale : float
        Multiplicative factor applied to embedding-space distances.
    n_ripser_workers : int
        Process pool size for parallel Ripser calls (multiprocessing.Pool).
        0 = ``SLURM_CPUS_PER_TASK`` or ``os.cpu_count()``, 1 = sequential.
    enable_topology_cache : bool
        Cache the kNN graph across epochs (recommended).
    topology_cache_size : int
        Maximum number of cached entries.
    topology_cache_namespace : str | None
        Namespace tag for cache isolation.
    show_progress : bool
        Show tqdm progress bar.
    """

    def __init__(
        self,
        k: int = 20,
        k_persistence: Optional[int] = None,
        embedding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        homology_dimensions: Optional[List[int]] = None,
        max_edge: float = float("inf"),
        use_coords: bool = False,
        edge_scale: float = 1.0,
        n_ripser_workers: int = 0,
        enable_topology_cache: bool = True,
        topology_cache_size: int = 50000,
        topology_cache_namespace: Optional[str] = None,
        show_progress: bool = False,
    ):
        super().__init__()
        if not HAS_RIPSER:
            raise ImportError(
                "ripser is required for LearnableFastEdgeFiltration. "
                "Install with: pip install ripser"
            )

        self.k = k
        self.k_persistence = k_persistence if k_persistence is not None else k
        self.embedding_dim = embedding_dim
        self.homology_dimensions = homology_dimensions or [0, 1]
        self.max_hom_dim = max(self.homology_dimensions)
        self.max_edge = max_edge
        self.use_coords = use_coords
        self.edge_scale = edge_scale
        self.show_progress = show_progress
        self.pre_compute = True  # flag expected by pipeline

        # Threading
        if n_ripser_workers == 0:
            # Use SLURM_CPUS_PER_TASK when available (os.cpu_count() reports ALL
            # node CPUs, not the allocated subset, causing thread over-subscription).
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus is not None:
                n_ripser_workers = max(1, int(slurm_cpus))
            else:
                n_ripser_workers = os.cpu_count() or 4
        self.n_ripser_workers = n_ripser_workers

        # Persistent multiprocessing pool for Ripser calls.
        # Ripser's C++ code holds the GIL, so ThreadPoolExecutor gives
        # zero parallelism.  mp.Pool bypasses the GIL entirely and
        # provides ~15x speedup with 16 workers.
        self._mp_pool: Optional[mp.Pool] = None
        if self.n_ripser_workers > 1:
            self._mp_pool = mp.Pool(
                processes=self.n_ripser_workers,
                initializer=_mp_ripser_worker_init,
            )
            # Warmup: force each worker to import ripser
            self._mp_pool.map(_mp_ripser_call, [
                (sp.coo_matrix((1, 1)), 0, float('inf'))
            ] * self.n_ripser_workers)

        # Topology cache  (placed after pool init)
        self.enable_topology_cache = enable_topology_cache
        self.topology_cache_size = max(1, int(topology_cache_size))
        self.topology_cache_namespace = topology_cache_namespace or "default"
        self._topology_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Learnable MLP
        self.mlp = FastEdgeEmbeddingMLP(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def __del__(self):
        """Clean up multiprocessing pool on garbage collection."""
        if getattr(self, '_mp_pool', None) is not None:
            try:
                self._mp_pool.terminate()
                self._mp_pool.join()
            except Exception:
                pass

    def clear_cache(self) -> None:
        self._topology_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "size": len(self._topology_cache),
            "max_size": self.topology_cache_size,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total) if total > 0 else 0.0,
            "namespace": self.topology_cache_namespace,
        }

    def _sample_cache_key(self, pts: torch.Tensor) -> str:
        pts_cpu = pts.detach().to(device="cpu", dtype=torch.float32, copy=False).contiguous()
        digest = hashlib.blake2b(pts_cpu.numpy().tobytes(), digest_size=16).hexdigest()
        shape_key = "x".join(str(int(v)) for v in pts_cpu.shape)
        return f"{self.topology_cache_namespace}:{shape_key}:{digest}"

    # ------------------------------------------------------------------
    # kNN graph (GPU-native, cached)
    # ------------------------------------------------------------------

    def _compute_knn_graph(self, pts: torch.Tensor):
        """Return (edges_np, knn_dists_cpu).

        edges_np : ndarray (n_edges, 2)  int64, i < j, unique
            Built from the ``k_persistence``-nearest neighbours (sparser
            graph for Ripser).
        knn_dists_cpu : Tensor (n, k)  float32 on CPU
            Full ``k``-nearest-neighbour distances (MLP input).
        """
        n = pts.shape[0]
        k_use = min(self.k, n - 1)

        # Pairwise distances on same device as pts
        dists = torch.cdist(pts, pts)  # (n, n)
        knn_dists, knn_idx = torch.topk(dists, k_use + 1, dim=-1, largest=False)
        knn_idx = knn_idx[:, 1:]       # drop self
        knn_dists = knn_dists[:, 1:]   # drop self-distance (0)

        # --- Edge set: use k_persistence (may be < k) -----------------
        k_edge = min(self.k_persistence, n - 1)
        if k_edge < k_use:
            edge_idx = knn_idx[:, :k_edge]  # only closest k_persistence
        else:
            edge_idx = knn_idx

        src = torch.arange(n, device=pts.device).unsqueeze(1).expand(-1, edge_idx.shape[1])
        pairs = torch.stack([src.reshape(-1), edge_idx.reshape(-1)], dim=1)
        pairs = torch.sort(pairs, dim=1).values          # i < j
        pairs = torch.unique(pairs, dim=0)                # deduplicate

        edges_np = pairs.cpu().numpy().astype(np.int64)
        knn_cpu = knn_dists.detach().cpu()                # full k distances

        return edges_np, knn_cpu

    def _build_topology_entry(self, pts: torch.Tensor) -> Dict[str, Any]:
        edges_np, knn_cpu = self._compute_knn_graph(pts)
        edge_index_cpu = torch.from_numpy(edges_np).long()
        n_vertices = int(pts.shape[0])

        # Pre-compute sparse matrix index arrays (reused every epoch)
        rows = np.concatenate([edges_np[:, 0], edges_np[:, 1]]).astype(np.int32)
        cols = np.concatenate([edges_np[:, 1], edges_np[:, 0]]).astype(np.int32)

        return {
            "edges": edges_np,
            "edge_index_cpu": edge_index_cpu,
            "knn": knn_cpu,
            "sparse_rows": rows,
            "sparse_cols": cols,
            "n_vertices": n_vertices,
        }

    def _get_or_build_topology(self, pts: torch.Tensor) -> Dict[str, Any]:
        if not self.enable_topology_cache:
            return self._build_topology_entry(pts)

        key = self._sample_cache_key(pts)
        cached = self._topology_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            self._topology_cache.move_to_end(key)
            return cached

        self._cache_misses += 1
        entry = self._build_topology_entry(pts)
        self._topology_cache[key] = entry
        self._topology_cache.move_to_end(key)
        if len(self._topology_cache) > self.topology_cache_size:
            self._topology_cache.popitem(last=False)
        return entry

    def get_topologies_cached_batch(self, X: torch.Tensor) -> List[Dict[str, Any]]:
        """Build / retrieve cached topology for every sample in X."""
        if X.ndim == 2:
            X = X.unsqueeze(0)
        samples: List[Dict[str, Any]] = []
        for i in range(X.shape[0]):
            pts = X[i]
            entry = self._get_or_build_topology(pts)
            sample: Dict[str, Any] = {"pts": pts}
            sample["edges"] = entry["edges"]
            sample["edge_index_tensor"] = entry["edge_index_cpu"].to(
                pts.device, non_blocking=True
            )
            sample["sparse_rows"] = entry["sparse_rows"]
            sample["sparse_cols"] = entry["sparse_cols"]
            sample["n_vertices"] = entry["n_vertices"]
            if entry["knn"] is not None:
                sample["knn_distances"] = entry["knn"].to(
                    pts.device, non_blocking=True
                )
            samples.append(sample)
        return samples

    # ------------------------------------------------------------------
    # Edge weight computation (differentiable)
    # ------------------------------------------------------------------

    def compute_knn_distances(self, pts: torch.Tensor) -> torch.Tensor:
        """Compute kNN distances (n_points, k) on the same device as pts."""
        dists = torch.cdist(pts, pts)
        k_use = min(self.k + 1, pts.shape[0])
        knn_dists, _ = torch.topk(dists, k_use, dim=-1, largest=False)
        knn_dists = knn_dists[:, 1 : self.k + 1]
        if knn_dists.shape[1] < self.k:
            pad = self.k - knn_dists.shape[1]
            knn_dists = torch.nn.functional.pad(knn_dists, (0, pad), value=0.0)
        return knn_dists

    def compute_edge_filtration(
        self, embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """||emb_i - emb_j||_2  for each edge (differentiable)."""
        diff = embeddings[edge_index[:, 0]] - embeddings[edge_index[:, 1]]
        return self.edge_scale * torch.linalg.norm(diff, dim=-1)

    # ------------------------------------------------------------------
    # Ripser helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sparse_matrix(
        edge_weights_np: np.ndarray, edges: np.ndarray, n_vertices: int,
        rows: Optional[np.ndarray] = None, cols: Optional[np.ndarray] = None,
    ) -> sp.coo_matrix:
        """Build a symmetric sparse distance matrix from edge list + weights.

        If pre-computed ``rows`` / ``cols`` index arrays are supplied (from the
        topology cache), skip the concatenation step.
        """
        if rows is None or cols is None:
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([edge_weights_np, edge_weights_np])
        return sp.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

    @staticmethod
    def _ripser_worker(sparse_matrix: sp.coo_matrix, maxdim: int, thresh: float):
        """Single ripser call (runs in thread – C++ releases GIL)."""
        return _ripser_module.ripser(
            sparse_matrix, maxdim=maxdim, thresh=thresh, distance_matrix=True
        )

    # ------------------------------------------------------------------
    # Differentiable diagram extraction
    # ------------------------------------------------------------------

    def _extract_diagrams(
        self,
        edge_weights: torch.Tensor,
        ripser_result: dict,
        sort_order: np.ndarray,
        sorted_weights: np.ndarray,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Match ripser birth/death values to edge indices and gather
        differentiable values from the edge_weights tensor.

        For Rips persistence:
          H0 – birth = 0 (vertex), death = edge weight
          H1 – birth = edge weight, death = edge weight
        """

        def _match_values(values: np.ndarray) -> torch.Tensor:
            """Map an array of distance-matrix values → differentiable tensor.

            Fully vectorised: no Python loop over features.
            """
            idxs = np.searchsorted(sorted_weights, values)
            idxs = np.clip(idxs, 0, len(sorted_weights) - 1)
            # Vectorised neighbour check: pick the closer of idx and idx-1
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
            # Keep only finite pairs
            finite_mask = np.isfinite(raw[:, 1])
            raw = raw[finite_mask]

            if len(raw) == 0:
                dgms_out.append(torch.zeros((0, 2), device=device))
                continue

            if hom_dim == 0:
                # H0: birth = 0 (vertex), death = matched edge
                births = torch.zeros(len(raw), device=device)
                deaths = _match_values(raw[:, 1])
                dgms_out.append(torch.stack([births, deaths], dim=-1))
            else:
                # H_k, k≥1: both birth and death are edge weights
                births = _match_values(raw[:, 0])
                deaths = _match_values(raw[:, 1])
                dgms_out.append(torch.stack([births, deaths], dim=-1))

        return dgms_out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_single(self, data) -> List[torch.Tensor]:
        """Process one sample (used by topology prewarm and fallback)."""
        # --- parse input ----
        if isinstance(data, torch.Tensor):
            pts = data
            device = pts.device
            topo = self._get_or_build_topology(pts)
            edges = topo["edges"]
            edge_index = topo["edge_index_cpu"].to(device, non_blocking=True)
            knn_dist = (
                topo["knn"].to(device, non_blocking=True)
                if topo["knn"] is not None
                else None
            )
        else:
            pts = data["pts"]
            device = pts.device
            edges = data.get("edges")
            edge_index = data.get("edge_index_tensor")
            knn_dist = data.get("knn_distances")
            if edges is None or edge_index is None:
                topo = self._get_or_build_topology(pts)
                edges = topo["edges"]
                edge_index = topo["edge_index_cpu"].to(device, non_blocking=True)
            if knn_dist is None and not self.use_coords:
                topo = self._get_or_build_topology(pts)
                if topo["knn"] is not None:
                    knn_dist = topo["knn"].to(device, non_blocking=True)
                else:
                    knn_dist = self.compute_knn_distances(pts)

        n_vertices = int(pts.shape[0])
        empty = [torch.zeros((0, 2), device=device) for _ in self.homology_dimensions]

        if len(edges) == 0:
            return empty

        # --- MLP → embeddings → edge weights (differentiable) ----------
        mlp_input = pts if self.use_coords else knn_dist
        embeddings = self.mlp(mlp_input)
        edge_weights = self.compute_edge_filtration(embeddings, edge_index)

        # --- Ripser (non-differentiable pairing) -----------------------
        ew_np = edge_weights.detach().cpu().numpy().astype(np.float64)
        sort_order = np.argsort(ew_np)
        sorted_weights = ew_np[sort_order]

        sparse_mat = self._build_sparse_matrix(ew_np, edges, n_vertices)
        rips = self._ripser_worker(sparse_mat, self.max_hom_dim, self.max_edge)

        # --- Differentiable extraction ---------------------------------
        return self._extract_diagrams(
            edge_weights, rips, sort_order, sorted_weights, device
        )

    def forward(self, X) -> List[List[torch.Tensor]]:
        """
        Batch forward with 3-phase parallelism:
          A. GPU – batched MLP + batched edge weights → single CPU transfer
          B. CPU processes – parallel Ripser calls (persistent mp.Pool)
          C. GPU – differentiable diagram extraction
        """
        # --- Prepare samples -------------------------------------------
        if isinstance(X, torch.Tensor):
            samples = self.get_topologies_cached_batch(X)
        else:
            samples = X

        n_samples = len(samples)
        all_diagrams: List[List[torch.Tensor]] = [[] for _ in self.homology_dimensions]
        if n_samples == 0:
            return all_diagrams

        device = samples[0]["pts"].device

        # ====== Phase A: GPU – batched MLP + edge weights ==============
        #
        # 1. Gather cached topology metadata (edges, indices, knn dists)
        # 2. Batch MLP: one big forward for all samples
        # 3. Batch edge weights: one big GPU gather, single .cpu() call
        # 4. Vectorised argsort on CPU (no Python loop)
        # -----------------------------------------------------------------

        # Gather MLP inputs and edge metadata
        mlp_inputs: List[Optional[torch.Tensor]] = []
        edge_meta: List[dict] = []
        for i in range(n_samples):
            s = samples[i]
            pts = s["pts"]
            edges = s.get("edges")
            edge_index = s.get("edge_index_tensor")
            knn_dist = s.get("knn_distances")
            sp_rows = s.get("sparse_rows")
            sp_cols = s.get("sparse_cols")
            n_verts_cached = s.get("n_vertices")

            if edges is None or edge_index is None:
                topo = self._get_or_build_topology(pts)
                edges = topo["edges"]
                edge_index = topo["edge_index_cpu"].to(device, non_blocking=True)
                sp_rows = topo["sparse_rows"]
                sp_cols = topo["sparse_cols"]
                n_verts_cached = topo["n_vertices"]
            if knn_dist is None and not self.use_coords:
                topo = self._get_or_build_topology(pts)
                if topo["knn"] is not None:
                    knn_dist = topo["knn"].to(device, non_blocking=True)
                else:
                    knn_dist = self.compute_knn_distances(pts)

            n_verts = n_verts_cached if n_verts_cached is not None else int(pts.shape[0])

            if len(edges) == 0:
                mlp_inputs.append(None)
                edge_meta.append({"empty": True})
                continue

            mlp_in = pts if self.use_coords else knn_dist
            mlp_inputs.append(mlp_in)
            edge_meta.append({
                "empty": False,
                "edges": edges,
                "edge_index": edge_index,
                "sp_rows": sp_rows,
                "sp_cols": sp_cols,
                "n_verts": n_verts,
            })

        non_empty = [i for i, m in enumerate(edge_meta) if not m["empty"]]
        if not non_empty:
            # All samples empty
            for _ in range(n_samples):
                for dim_idx in range(len(self.homology_dimensions)):
                    all_diagrams[dim_idx].append(torch.zeros((0, 2), device=device))
            return all_diagrams

        # --- Batched MLP: single GPU forward ---------------------------
        stacked = torch.stack([mlp_inputs[i] for i in non_empty])  # (B, n_pts, d_in)
        B, n_pts, _ = stacked.shape
        emb_flat = self.mlp(stacked.reshape(B * n_pts, -1))        # (B*n_pts, emb_dim)
        emb_all = emb_flat.reshape(B, n_pts, -1)                    # (B, n_pts, emb_dim)

        # --- Batched edge weights on GPU, single CPU transfer ----------
        #
        # Instead of 200 individual .cpu() calls, compute all edge weights
        # on GPU, concatenate, transfer once, then slice on CPU.
        n_edges_list: List[int] = []    # number of edges per non-empty sample
        all_edge_weights: List[torch.Tensor] = []  # differentiable, on device

        for idx_in_batch, i in enumerate(non_empty):
            meta = edge_meta[i]
            embeddings = emb_all[idx_in_batch]
            edge_weights = self.compute_edge_filtration(embeddings, meta["edge_index"])
            all_edge_weights.append(edge_weights)
            n_edges_list.append(edge_weights.shape[0])

        # Single GPU → CPU transfer for all edge weights
        all_ew_cat = torch.cat(all_edge_weights)                    # (total_edges,) on GPU
        all_ew_np = all_ew_cat.detach().cpu().numpy().astype(np.float64)  # single transfer

        # --- Vectorised per-sample argsort + sparse matrix construction
        per_sample: List[dict] = [{"empty": True}] * n_samples
        offset = 0
        for idx_in_batch, i in enumerate(non_empty):
            meta = edge_meta[i]
            ne = n_edges_list[idx_in_batch]
            ew_np = all_ew_np[offset : offset + ne]
            offset += ne

            sort_order = np.argsort(ew_np)
            sorted_weights = ew_np[sort_order]
            sparse_mat = self._build_sparse_matrix(
                ew_np, meta["edges"], meta["n_verts"],
                rows=meta["sp_rows"], cols=meta["sp_cols"],
            )

            per_sample[i] = {
                "empty": False,
                "edge_weights": all_edge_weights[idx_in_batch],
                "sparse_mat": sparse_mat,
                "sort_order": sort_order,
                "sorted_weights": sorted_weights,
            }

        # ====== Phase B: parallel Ripser via multiprocessing.Pool =====
        #
        # Ripser's C++ holds the GIL, so ThreadPoolExecutor gives zero
        # parallelism.  mp.Pool runs each call in a separate process,
        # bypassing the GIL and achieving ~15x speedup with 16 workers.
        rips_results: List[Optional[dict]] = [None] * n_samples
        non_empty_indices = [i for i in non_empty]  # already computed

        if self._mp_pool is not None and len(non_empty_indices) > 1:
            args_list = [
                (per_sample[i]["sparse_mat"], self.max_hom_dim, self.max_edge)
                for i in non_empty_indices
            ]
            results = self._mp_pool.map(_mp_ripser_call, args_list, chunksize=1)
            for j, i in enumerate(non_empty_indices):
                rips_results[i] = results[j]
        else:
            for i in non_empty_indices:
                rips_results[i] = self._ripser_worker(
                    per_sample[i]["sparse_mat"], self.max_hom_dim, self.max_edge
                )

        # ====== Phase C: GPU – differentiable diagram extraction =======
        for i in range(n_samples):
            if per_sample[i]["empty"]:
                for dim_idx in range(len(self.homology_dimensions)):
                    all_diagrams[dim_idx].append(
                        torch.zeros((0, 2), device=device)
                    )
            else:
                dgms = self._extract_diagrams(
                    per_sample[i]["edge_weights"],
                    rips_results[i],
                    per_sample[i]["sort_order"],
                    per_sample[i]["sorted_weights"],
                    device,
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
