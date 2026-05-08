"""
Learnable edge filtration for point clouds using flag complex.

This module implements a learnable filtration that transforms point clouds
to an embedding space via a neural network, then uses pairwise distances
in that embedding space as edge filtration values (Rips-type filtration).

Pipeline:
    Point Cloud → MLP → Embedding Space → Pairwise Distances → Edge Filtration
                                                                      ↓
                        Flag Complex (kNN skeleton) with learned metric
                                                                      ↓
                                      Persistence Diagrams (differentiable)

The key idea is to learn a deformation of the metric that maximizes Fisher
information: "what notion of distance is most informative about the parameter?"

Unlike the vertex-based filtration (learnable_point) which assigns scalar
values to vertices and uses lower-star filtration, this approach:
  - Assigns each vertex an embedding vector via MLP
  - Edge weight = ||f(x_i) - f(x_j)|| in embedding space
  - Simplex filtration = max edge weight (standard Rips convention)
  - Gradients flow: Fisher → diagram → edge weights → embeddings → MLP

This is conceptually a "learned Rips" filtration where the MLP learns
what metric space makes the topology most informative about the model
parameters being estimated.
"""
import hashlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Literal
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm


class EdgeEmbeddingMLP(nn.Module):
    """
    MLP that maps point coordinates (or kNN distances) to an embedding
    vector in R^d. Pairwise distances in the embedding space define
    the learned edge filtration.

    Uses lazy initialization - input dimension is automatically inferred
    on the first forward pass.

    Args:
        embedding_dim: Dimension of the output embedding space.
        hidden_dims: List of hidden layer dimensions.
                    None or [] for linear (no hidden layers)
        dropout: Dropout probability (default 0.0)
    """

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
        """Build MLP network with lazy first layer."""
        layers = []

        if not self.hidden_dims:
            # Linear: input → embedding_dim
            layers.append(nn.LazyLinear(self.embedding_dim))
        else:
            # First layer (lazy)
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

            # Middle layers
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))

            # Final layer → embedding_dim (no activation — free embedding)
            layers.append(nn.Linear(self.hidden_dims[-1], self.embedding_dim))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embedding vectors.

        Args:
            x: Input features of shape (n_points, d_in)

        Returns:
            Embeddings of shape (n_points, embedding_dim)
        """
        return self.network(x)


class LearnableEdgeFiltration(nn.Module):
    """
    Learnable edge filtration for point clouds.

    This layer:
    1. Applies an MLP to map each vertex to an embedding in R^d
    2. Computes edge weights as pairwise distances in embedding space
    3. Uses standard Rips convention: simplex value = max edge weight
    4. Builds simplex tree and computes persistence
    5. Extracts diagrams differentiably via torch.gather

    The kNN graph (1-skeleton) is determined by the original point cloud
    coordinates and cached. Only the filtration values change during
    training — the graph topology is fixed.

    Args:
        k: Number of nearest neighbors for kNN graph skeleton
        embedding_dim: Dimension of the learned embedding space
        hidden_dims: List of hidden layer dimensions for MLP
        dropout: Dropout probability for MLP (default 0.0)
        homology_dimensions: List of homology dimensions to compute
        complex_type: 'alpha' or 'knn' — graph skeleton source
        max_edge: Maximum edge length for rips complex
        show_progress: Show progress bar during computation
        use_coords: If True, MLP input is raw coordinates; if False, kNN distances
        edge_scale: Scaling factor for edge weights (default 1.0)
        enable_topology_cache: Cache graph topology across epochs
        topology_cache_size: Max entries in topology cache
        topology_cache_namespace: Namespace for cache isolation
    """

    def __init__(
        self,
        k: int = 10,
        embedding_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        homology_dimensions: List[int] = [0, 1],
        complex_type: Literal['alpha', 'knn'] = 'alpha',
        max_edge: float = np.inf,
        show_progress: bool = False,
        use_coords: bool = False,
        edge_scale: float = 1.0,
        enable_topology_cache: bool = True,
        topology_cache_size: int = 20000,
        topology_cache_namespace: Optional[str] = None,
    ):
        super().__init__()

        self.k = k
        self.embedding_dim = embedding_dim
        self.homology_dimensions = homology_dimensions
        self.max_hom_dim = max(homology_dimensions)
        self.complex_type = complex_type
        self.max_edge = max_edge
        self.show_progress = show_progress
        self.use_coords = use_coords
        self.edge_scale = edge_scale
        self.pre_compute = True
        self.enable_topology_cache = enable_topology_cache
        self.topology_cache_size = max(1, int(topology_cache_size))
        self.topology_cache_namespace = topology_cache_namespace or "default"

        # LRU cache: sample_key -> topology data
        self._topology_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Learnable MLP for embedding
        self.mlp = EdgeEmbeddingMLP(
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def clear_cache(self) -> None:
        """Clear cached sample topologies."""
        self._topology_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache usage statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            'size': len(self._topology_cache),
            'max_size': self.topology_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': (self._cache_hits / total) if total > 0 else 0.0,
            'namespace': self.topology_cache_namespace,
        }

    def _sample_cache_key(self, pts: torch.Tensor) -> str:
        """Stable cache key for a point cloud sample."""
        pts_cpu = pts.detach().to(device='cpu', dtype=torch.float32, copy=False).contiguous()
        digest = hashlib.blake2b(pts_cpu.numpy().tobytes(), digest_size=16).hexdigest()
        shape_key = 'x'.join(str(int(v)) for v in pts_cpu.shape)
        return f"{self.topology_cache_namespace}:{shape_key}:{digest}"

    def compute_knn_distances(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Compute k-nearest neighbor distances for each point.

        Args:
            pts: Point cloud of shape (n_points, d)

        Returns:
            kNN distances of shape (n_points, k)
        """
        dists = torch.cdist(pts, pts)  # (n_points, n_points)
        k_plus_1 = min(self.k + 1, pts.shape[0])
        knn_dists, _ = torch.topk(dists, k_plus_1, dim=-1, largest=False)
        knn_dists = knn_dists[:, 1:self.k + 1]
        if knn_dists.shape[1] < self.k:
            pad_size = self.k - knn_dists.shape[1]
            knn_dists = torch.nn.functional.pad(knn_dists, (0, pad_size), value=0.0)
        return knn_dists

    def _extract_edges(self, pts_np: np.ndarray) -> np.ndarray:
        """Extract 1-skeleton edges from alpha or kNN complex."""
        if self.complex_type == 'alpha':
            alpha = gudhi.AlphaComplex(points=pts_np)
            st = alpha.create_simplex_tree(default_filtration_value=False)
            edges = np.array(
                [s[0] for s in st.get_skeleton(1) if len(s[0]) == 2],
                dtype=np.int64,
            )
        elif self.complex_type == 'knn':
            # Build kNN graph directly
            from scipy.spatial import KDTree
            tree = KDTree(pts_np)
            _, idxs = tree.query(pts_np, k=self.k + 1)
            edge_set = set()
            for v in range(len(pts_np)):
                for j in range(1, idxs.shape[1]):
                    u = idxs[v, j]
                    edge_set.add((min(v, u), max(v, u)))
            edges = np.array(sorted(edge_set), dtype=np.int64)
        else:
            raise ValueError(f"Unknown complex_type: {self.complex_type}")

        if edges.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return edges

    def _build_topology_entry(self, pts: torch.Tensor) -> Dict[str, Any]:
        """Build and store precomputed topology data for one sample."""
        pts_np = pts.detach().cpu().numpy()
        edges = self._extract_edges(pts_np)

        knn_cpu = None
        if not self.use_coords:
            knn = self.compute_knn_distances(pts)
            knn_cpu = knn.detach().cpu()

        if edges.size > 0:
            edge_index_cpu = torch.from_numpy(edges).long()
        else:
            edge_index_cpu = torch.empty((0, 2), dtype=torch.long)

        return {
            'edges': edges,
            'knn': knn_cpu,
            'edge_index_cpu': edge_index_cpu,
        }

    def _get_or_build_topology(self, pts: torch.Tensor) -> Dict[str, Any]:
        """Fetch topology entry from cache or compute and insert it."""
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
        """Convert a batch tensor to precomputed sample dicts using topology cache."""
        if X.ndim == 2:
            X = X.unsqueeze(0)

        samples: List[Dict[str, Any]] = []
        for i in range(X.shape[0]):
            pts = X[i]
            entry = self._get_or_build_topology(pts)
            sample = {
                'pts': pts,
                'alpha_graph': entry['edges'],
            }
            if entry.get('edge_index_cpu') is not None:
                sample['edge_index_tensor'] = entry['edge_index_cpu'].to(pts.device, non_blocking=True)
            if entry['knn'] is not None:
                sample['knn_distances'] = entry['knn'].to(pts.device, non_blocking=True)
            samples.append(sample)

        return samples

    def compute_edge_filtration(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute edge filtration values as distances in embedding space.

        edge_weight_{ij} = edge_scale * ||f(x_i) - f(x_j)||_2

        Args:
            embeddings: Vertex embeddings of shape (n_points, embedding_dim)
            edge_index: Edge indices of shape (n_edges, 2)

        Returns:
            Edge filtration values of shape (n_edges,)
        """
        emb_u = embeddings[edge_index[:, 0]]
        emb_v = embeddings[edge_index[:, 1]]
        # L2 distance in embedding space
        edge_weights = torch.linalg.norm(emb_u - emb_v, dim=-1)
        return self.edge_scale * edge_weights

    def update_filtration(
        self,
        edge_filts_np: np.ndarray,
        edges: np.ndarray,
        n_vertices: int,
    ) -> gudhi.SimplexTree:
        """
        Build simplex tree with edge-based (Rips-type) filtration.

        Vertices get filtration value 0 (all born at time 0).
        Edges get learned filtration values.
        Higher simplices get max-edge filtration via expansion.

        Uses GUDHI's insert_batch API for vectorised C++ insertion.
        """
        new_st = gudhi.SimplexTree()

        # Batch-insert vertices with filtration 0
        vert_array = np.arange(n_vertices, dtype=np.int32).reshape(1, -1)
        vert_filts = np.zeros(n_vertices, dtype=np.float64)
        new_st.insert_batch(vert_array, vert_filts)

        # Batch-insert edges with learned filtration values
        if len(edges) > 0:
            edge_array = edges.T.astype(np.int32)  # (2, n_edges)
            new_st.insert_batch(edge_array, edge_filts_np.astype(np.float64))

        # Expand to higher simplices (flag complex from 1-skeleton)
        new_st.expansion(self.max_hom_dim + 1)

        # For Rips-type: make_filtration_non_decreasing ensures
        # simplex value = max of face values (which = max edge weight)
        new_st.make_filtration_non_decreasing()

        return new_st

    def extract_diagrams_single(
        self,
        edge_filts: torch.Tensor,
        edges: np.ndarray,
        n_vertices: int,
        pers_generators: tuple,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Extract persistence diagrams differentiably for a single sample.

        For edge-based filtration:
        - H0: birth = 0 (vertex, non-differentiable constant),
               death = edge filtration value
        - H1: birth = edge filtration, death = edge filtration

        Gradient flows through edge_filts → embedding distances → MLP.

        Args:
            edge_filts: Edge filtration values (n_edges,), differentiable
            edges: Edge indices (n_edges, 2)
            n_vertices: Number of vertices
            pers_generators: Output from st.flag_persistence_generators()
            device: torch device

        Returns:
            List of diagrams, one per homology dimension
        """
        diagrams = []

        # Build edge lookup: (u, v) → edge index
        edge_lookup = np.full((n_vertices, n_vertices), -1, dtype=np.int64)
        if len(edges) > 0:
            edge_lookup[edges[:, 0], edges[:, 1]] = np.arange(len(edges), dtype=np.int64)
            edge_lookup[edges[:, 1], edges[:, 0]] = np.arange(len(edges), dtype=np.int64)

        # H0 generators: birth at vertex (filt=0), death at edge
        # pers_generators[0] has shape (n_pairs, 3): [birth_v, death_v1, death_v2]
        if 0 in self.homology_dimensions:
            h0_gens = pers_generators[0]
            if len(h0_gens) > 0:
                h0_gens = np.array(h0_gens)

                # Birth: vertex filtration = 0 for all vertices in Rips
                births = torch.zeros(len(h0_gens), device=device)

                # Death: edge filtration (differentiable)
                death_edges = h0_gens[:, 1:]
                death_idx_np = edge_lookup[death_edges[:, 0], death_edges[:, 1]]
                death_indices = torch.from_numpy(death_idx_np).long().to(device)
                deaths = edge_filts[death_indices]

                dgm_h0 = torch.stack([births, deaths], dim=-1)
            else:
                dgm_h0 = torch.zeros((0, 2), device=device)
            diagrams.append(dgm_h0)

        # H1 generators: birth at edge, death at edge
        # pers_generators[1][0] has shape (n_pairs, 4): [birth_v1, birth_v2, death_v1, death_v2]
        if 1 in self.homology_dimensions:
            if len(pers_generators) > 1 and len(pers_generators[1]) > 0:
                h1_gens = (
                    np.array(pers_generators[1][0])
                    if len(pers_generators[1][0]) > 0
                    else np.array([]).reshape(0, 4)
                )
                if len(h1_gens) > 0:
                    # Birth: edge filtration
                    birth_edges = h1_gens[:, :2]
                    birth_idx_np = edge_lookup[birth_edges[:, 0], birth_edges[:, 1]]
                    birth_indices = torch.from_numpy(birth_idx_np).long().to(device)
                    births = edge_filts[birth_indices]

                    # Death: edge filtration
                    death_edges = h1_gens[:, 2:]
                    death_idx_np = edge_lookup[death_edges[:, 0], death_edges[:, 1]]
                    death_indices = torch.from_numpy(death_idx_np).long().to(device)
                    deaths = edge_filts[death_indices]

                    dgm_h1 = torch.stack([births, deaths], dim=-1)
                else:
                    dgm_h1 = torch.zeros((0, 2), device=device)
            else:
                dgm_h1 = torch.zeros((0, 2), device=device)
            diagrams.append(dgm_h1)

        return diagrams

    def forward_single(self, data) -> List[torch.Tensor]:
        """
        Process a single point cloud through the learned edge filtration.
        """
        # 1. Parse input
        if isinstance(data, torch.Tensor):
            pts = data
            device = pts.device
            topo = self._get_or_build_topology(pts)
            edges = topo['edges']
            edge_index_tensor = topo['edge_index_cpu'].to(device, non_blocking=True)
            knn_dist = topo['knn'].to(device, non_blocking=True) if topo['knn'] is not None else None
        else:
            pts = data['pts']
            device = pts.device
            edges = data.get('alpha_graph')
            edge_index_tensor = data.get('edge_index_tensor')
            if edges is None:
                topo = self._get_or_build_topology(pts)
                edges = topo['edges']
            if edge_index_tensor is None:
                topo = self._get_or_build_topology(pts)
                edge_index_tensor = topo['edge_index_cpu'].to(device, non_blocking=True)
            knn_dist = data.get('knn_distances')
            if knn_dist is None and not self.use_coords:
                topo = self._get_or_build_topology(pts)
                if topo['knn'] is not None:
                    knn_dist = topo['knn'].to(device, non_blocking=True)
                else:
                    knn_dist = self.compute_knn_distances(pts)

        n_vertices = int(pts.shape[0])

        # 2. Compute embeddings via MLP
        mlp_input = pts if self.use_coords else knn_dist
        embeddings = self.mlp(mlp_input)  # (n_points, embedding_dim)

        # 3. Check edges
        if len(edges) == 0:
            return [torch.zeros((0, 2), device=device) for _ in self.homology_dimensions]

        # 4. Compute edge filtration = distances in embedding space (differentiable)
        edge_filts = self.compute_edge_filtration(embeddings, edge_index_tensor)

        # 5. Build simplex tree with edge filtration (non-differentiable)
        edge_filts_np = edge_filts.detach().cpu().numpy()
        new_st = self.update_filtration(edge_filts_np, edges, n_vertices)

        # 6. Compute persistence and get flag generators
        new_st.compute_persistence()
        pers_generators = new_st.flag_persistence_generators()

        # 7. Extract diagrams differentiably via torch.gather
        diagrams = self.extract_diagrams_single(
            edge_filts, edges, n_vertices, pers_generators, device
        )

        return diagrams

    def forward(self, X) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams for a batch of point clouds.

        Args:
            X: torch.Tensor (batch, n_points, d) or List[dict] with
               pre-computed topology data.

        Returns:
            List of diagram lists, one per homology dimension.
            all_diagrams[hom_dim][sample_idx] is a (n_pairs, 2) tensor.
        """
        if isinstance(X, torch.Tensor):
            samples = self.get_topologies_cached_batch(X)
        else:
            samples = X

        n_samples = len(samples)
        all_diagrams = [[] for _ in self.homology_dimensions]

        if n_samples == 0:
            return all_diagrams

        iterator = (
            tqdm(range(n_samples), desc="Computing Edge Filtration")
            if self.show_progress
            else range(n_samples)
        )

        for i in iterator:
            dgms = self.forward_single(samples[i])
            for dim_idx, dgm in enumerate(dgms):
                all_diagrams[dim_idx].append(dgm)

        return all_diagrams

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters (0 if lazy layers not yet initialized)."""
        total = 0
        for p in self.parameters():
            if p.requires_grad and not isinstance(p, nn.UninitializedParameter):
                total += p.numel()
        return total
