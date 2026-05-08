"""
Learnable filtration for point clouds using flag complex.

This module implements a learnable filtration that transforms point clouds
using a neural network to compute vertex filtration values. The network
takes k-nearest neighbor distances as input and outputs a filtration value
for each vertex.

Pipeline:
    Point Cloud → kNN distances → MLP → Vertex Filtration
                                              ↓
                    Flag Complex (alpha/rips) with learned filtration
                                              ↓
                              Persistence Diagrams (differentiable)

The key insight is that while persistence pairing is computed via GUDHI
(non-differentiable), the actual birth/death values are extracted via
torch.gather (differentiable), enabling gradient flow through the MLP.

Based on TensorFlow implementation in:
    external/topofisher_tensorflow/topofisher/filtrations/tensorflow/flag_layer.py
    external/topofisher_tensorflow/topofisher/filtrations/tensorflow/dtm_layer.py

TODO: Clean up and refactor this code:
    - Consolidate extract_diagrams_single logic
    - Add batched kNN computation for efficiency
    - Better error handling for edge cases
    - Add support for higher homology dimensions (H2+)
"""
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Literal
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm


class VertexFiltrationMLP(nn.Module):
    """
    MLP that maps k-nearest neighbor distances to a single filtration value.

    Uses lazy initialization - input dimension is automatically inferred
    on the first forward pass based on k-nearest neighbors.

    Args:
        hidden_dims: List of hidden layer dimensions.
                    None or [] for linear (no hidden layers)
                    [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
        dropout: Dropout probability (default 0.0, no dropout)
    """

    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.dropout_prob = dropout
        self.output_dim = 1  # Always output 1 filtration value per vertex

        # Build network
        self.network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Build MLP network with lazy first layer."""
        layers = []

        if not self.hidden_dims:
            # Linear (no hidden layers): input → 1
            layers.append(nn.LazyLinear(self.output_dim))
            layers.append(nn.Tanh())
        else:
            # First layer is lazy (input dimension inferred on first forward)
            layers.append(nn.LazyLinear(self.hidden_dims[0]))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))

            # Middle layers (dimensions known)
            for i in range(len(self.hidden_dims) - 1):
                layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
                if self.dropout_prob > 0:
                    layers.append(nn.Dropout(self.dropout_prob))

            # Final layer → 1 output
            layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, knn_distances: torch.Tensor) -> torch.Tensor:
        """
        Compute vertex filtration values from kNN distances.

        Args:
            knn_distances: Tensor of shape (n_points, k) or (batch, n_points, k)

        Returns:
            Vertex filtration values of shape (n_points,) or (batch, n_points)
        """
        # 5 * tanh(final_linear)
        return 5.0 * self.network(knn_distances).squeeze(-1)



class LearnablePointFiltration(nn.Module):
    """
    Learnable filtration for point clouds using flag complex.

    This layer:
    1. Computes k-nearest neighbor distances for each point
    2. Applies an MLP to get vertex filtration values
    3. Computes edge filtrations as: (d^p + fmax^p)^(1/p)
       where d = edge length, fmax = max(f(u), f(v))
    4. Builds simplex tree and computes persistence
    5. Extracts diagrams differentiably via torch.gather

    Args:
        k: Number of nearest neighbors for vertex filtration
        hidden_dims: List of hidden layer dimensions for MLP.
                    None or [] for linear (no hidden layers)
                    [h1] for 1 hidden layer, [h1, h2] for 2 hidden layers, etc.
        dropout: Dropout probability for MLP (default 0.0)
        homology_dimensions: List of homology dimensions to compute
        complex_type: 'alpha' or 'rips'
        max_edge: Maximum edge length for rips complex
        p: Parameter for edge filtration formula (default: 1.0)
        show_progress: Show progress bar during computation
    """

    def __init__(
        self,
        k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
        homology_dimensions: List[int] = [0, 1],
        complex_type: Literal['alpha', 'rips'] = 'alpha',
        max_edge: float = np.inf,
        p: float = 1.0,
        show_progress: bool = False,
        use_coords: bool = False,
        enable_topology_cache: bool = True,
        topology_cache_size: int = 50000,
        topology_cache_namespace: Optional[str] = None,
        n_gudhi_workers: int = 0,
    ):
        super().__init__()

        self.k = k
        self.homology_dimensions = homology_dimensions
        self.max_hom_dim = max(homology_dimensions)
        self.complex_type = complex_type
        self.max_edge = max_edge
        self.p = p
        self.show_progress = show_progress
        self.use_coords = use_coords
        self.pre_compute = True
        self.enable_topology_cache = enable_topology_cache
        self.topology_cache_size = max(1, int(topology_cache_size))
        self.topology_cache_namespace = topology_cache_namespace or "default"

        # Number of threads for parallel GUDHI persistence.
        # 0 = auto, 1 = sequential (no threading).
        # Use SLURM_CPUS_PER_TASK when available (os.cpu_count() reports ALL
        # node CPUs, not the allocated subset, causing thread over-subscription).
        import os
        if n_gudhi_workers <= 0:
            slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            if slurm_cpus is not None:
                self.n_gudhi_workers = max(1, int(slurm_cpus))
            else:
                self.n_gudhi_workers = max(1, os.cpu_count() or 1)
        else:
            self.n_gudhi_workers = n_gudhi_workers

        # LRU cache: sample_key -> {'edges': np.ndarray, 'knn': Optional[torch.Tensor]}
        self._topology_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0

        # Learnable MLP for vertex filtration
        self.mlp = VertexFiltrationMLP(hidden_dims=hidden_dims, dropout=dropout)

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

    def _extract_alpha_edges(self, pts_np: np.ndarray) -> np.ndarray:
        """Extract 1-skeleton edges from alpha/rips simplex tree."""
        st_temp = self.get_simplex_tree(pts_np)
        edges = np.array([s[0] for s in st_temp.get_skeleton(1) if len(s[0]) == 2], dtype=np.int64)
        if edges.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return edges

    def _build_topology_entry_cpu(self, pts_np: np.ndarray) -> Dict[str, Any]:
        """Build topology entry from numpy array (CPU-only, thread-safe).

        This is the expensive part: alpha complex construction + edge extraction.
        Since GUDHI C++ releases the GIL, this can run in parallel threads.
        """
        edges = self._extract_alpha_edges(pts_np)

        knn_cpu = None
        if not self.use_coords:
            # CPU kNN via torch on CPU tensor
            pts_t = torch.from_numpy(pts_np).float()
            knn = self.compute_knn_distances(pts_t)
            knn_cpu = knn.detach()

        if edges.size > 0:
            edge_index_cpu = torch.from_numpy(edges).long()
            pts_t = torch.from_numpy(pts_np).float()
            pts_u = pts_t[edge_index_cpu[:, 0]]
            pts_v = pts_t[edge_index_cpu[:, 1]]
            edge_distances_cpu = torch.linalg.norm(
                pts_u - pts_v, dim=-1
            ).detach()
        else:
            edge_index_cpu = torch.empty((0, 2), dtype=torch.long)
            edge_distances_cpu = torch.empty(0)

        return {
            'edges': edges,
            'knn': knn_cpu,
            'edge_index_cpu': edge_index_cpu,
            'edge_distances_cpu': edge_distances_cpu,
        }

    def _build_topology_entry(self, pts: torch.Tensor) -> Dict[str, Any]:
        """Build and store precomputed topology data for one sample."""
        pts_np = pts.detach().cpu().numpy()
        return self._build_topology_entry_cpu(pts_np)

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
        """Convert a batch tensor to precomputed sample dicts using topology cache.

        Cache misses are built in parallel threads (GUDHI C++ releases the GIL),
        then inserted into the cache before assembling the output list.
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)

        n = X.shape[0]
        entries: List[Optional[Dict[str, Any]]] = [None] * n
        miss_indices: List[int] = []

        # ── 1. Check cache for all samples ──
        keys: List[Optional[str]] = [None] * n
        for i in range(n):
            if not self.enable_topology_cache:
                miss_indices.append(i)
                continue
            key = self._sample_cache_key(X[i])
            keys[i] = key
            cached = self._topology_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                self._topology_cache.move_to_end(key)
                entries[i] = cached
            else:
                miss_indices.append(i)

        # ── 2. Build topologies for cache misses in parallel ──
        if miss_indices:
            # Move all miss samples to CPU numpy once
            miss_pts_np = [X[i].detach().cpu().numpy() for i in miss_indices]

            if self.n_gudhi_workers > 1 and len(miss_indices) > 1:
                with ThreadPoolExecutor(max_workers=self.n_gudhi_workers) as pool:
                    built = list(pool.map(self._build_topology_entry_cpu, miss_pts_np))
            else:
                built = [self._build_topology_entry_cpu(p) for p in miss_pts_np]

            # Insert into cache
            for j, i in enumerate(miss_indices):
                entry = built[j]
                entries[i] = entry
                self._cache_misses += 1
                if self.enable_topology_cache and keys[i] is not None:
                    self._topology_cache[keys[i]] = entry
                    self._topology_cache.move_to_end(keys[i])
                    if len(self._topology_cache) > self.topology_cache_size:
                        self._topology_cache.popitem(last=False)

        # ── 3. Assemble output samples ──
        device = X.device
        samples: List[Dict[str, Any]] = []
        for i in range(n):
            entry = entries[i]
            sample = {
                'pts': X[i],
                'alpha_graph': entry['edges'],
            }
            if entry.get('edge_index_cpu') is not None:
                sample['edge_index_tensor'] = entry['edge_index_cpu'].to(device, non_blocking=True)
            if entry.get('edge_distances_cpu') is not None:
                sample['edge_distances'] = entry['edge_distances_cpu'].to(device, non_blocking=True)
            if entry['knn'] is not None:
                sample['knn_distances'] = entry['knn'].to(device, non_blocking=True)
            samples.append(sample)

        return samples

    def compute_knn_distances(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Compute k-nearest neighbor distances for each point.

        Args:
            pts: Point cloud of shape (n_points, d)

        Returns:
            kNN distances of shape (n_points, k)
        """
        # Compute pairwise distances
        dists = torch.cdist(pts, pts)  # (n_points, n_points)

        # Get k+1 smallest (including self with distance 0)
        # Then exclude self (first column)
        k_plus_1 = min(self.k + 1, pts.shape[0])
        knn_dists, _ = torch.topk(dists, k_plus_1, dim=-1, largest=False)

        # Exclude self-distance (first column) and take k neighbors
        knn_dists = knn_dists[:, 1:self.k + 1]

        # Pad with zeros if fewer than k neighbors
        if knn_dists.shape[1] < self.k:
            pad_size = self.k - knn_dists.shape[1]
            knn_dists = torch.nn.functional.pad(knn_dists, (0, pad_size), value=0.0)

        return knn_dists

    def compute_edge_filtration(
        self,
        pts: torch.Tensor,
        vertex_filts: torch.Tensor,
        edges: np.ndarray,
        edge_index_tensor: Optional[torch.Tensor] = None,
        edge_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge filtration values.

        Edge filtration = (d^p + fmax^p)^(1/p)
        where d = edge length, fmax = max(f(u), f(v))

        When p=1: edge_filt = d + max(f(u), f(v))

        Args:
            pts: Point cloud of shape (n_points, d)
            vertex_filts: Vertex filtration values of shape (n_points,)
            edges: Edge indices of shape (n_edges, 2)
            edge_index_tensor: Pre-computed edge index tensor (optional).
            edge_distances: Pre-computed edge lengths (optional).
                If provided, skips the per-call point indexing + norm.

        Returns:
            Edge filtration values of shape (n_edges,)
        """
        p = self.p

        # Get vertex filtration values for edge endpoints
        if edge_index_tensor is None:
            edges_tensor = torch.from_numpy(edges).long().to(pts.device)
        else:
            edges_tensor = edge_index_tensor
        f_u = vertex_filts[edges_tensor[:, 0]]
        f_v = vertex_filts[edges_tensor[:, 1]]
        fmax = torch.maximum(f_u, f_v)

        if p <= 0:
            return fmax

        # Edge lengths — use cached distances if available
        if edge_distances is not None:
            d = edge_distances
        else:
            pts_u = pts[edges_tensor[:, 0]]
            pts_v = pts[edges_tensor[:, 1]]
            d = torch.linalg.norm(pts_u - pts_v, dim=-1)

        # Edge filtration formula
        if p == 1.0:
            return d + fmax
        else:
            return (d ** p + fmax ** p) ** (1 / p)

    def get_simplex_tree(self, pts_np: np.ndarray) -> gudhi.SimplexTree:
        """
        Create simplex tree from point cloud.

        Args:
            pts_np: Point cloud as numpy array of shape (n_points, d)

        Returns:
            GUDHI SimplexTree
        """
        if self.complex_type == 'alpha':
            alpha = gudhi.AlphaComplex(points=pts_np)
            st = alpha.create_simplex_tree(default_filtration_value=False)
        elif self.complex_type == 'rips':
            rips = gudhi.RipsComplex(points=pts_np, max_edge_length=self.max_edge)
            st = rips.create_simplex_tree(max_dimension=2)
        else:
            raise ValueError(f"Unknown complex_type: {self.complex_type}")

        return st

    def update_filtration(
        self,
        vertex_filts_np: np.ndarray,
        edge_filts_np: np.ndarray,
        edges: np.ndarray
    ) -> gudhi.SimplexTree:
        """
        Update simplex tree with learned filtration values.

        Uses GUDHI's insert_batch API for vectorised C++ insertion
        instead of per-element Python loops (~2.5x faster).
        """
        new_st = gudhi.SimplexTree()

        # Batch-insert vertices: vertex_array shape (1, n_vertices)
        n_vertices = len(vertex_filts_np)
        vert_array = np.arange(n_vertices, dtype=np.int32).reshape(1, -1)
        new_st.insert_batch(vert_array, vertex_filts_np.astype(np.float64))

        # Batch-insert edges: edge_array shape (2, n_edges)
        if len(edges) > 0:
            edge_array = edges.T.astype(np.int32)  # (2, n_edges)
            new_st.insert_batch(edge_array, edge_filts_np.astype(np.float64))

        # Expand to higher simplices (st-expansion)
        new_st.expansion(self.max_hom_dim + 1)

        # Ensure filtration is non-decreasing
        new_st.make_filtration_non_decreasing()

        return new_st

    def extract_diagrams_single(
        self,
        pts: torch.Tensor,
        vertex_filts: torch.Tensor,
        edge_filts: torch.Tensor,
        edges: np.ndarray,
        pers_generators: tuple
    ) -> List[torch.Tensor]:
        """
        Extract persistence diagrams differentiably for a single point cloud.

        Uses torch.gather to extract birth/death values from filtrations,
        enabling gradient flow.

        Args:
            pts: Point cloud of shape (n_points, d)
            vertex_filts: Vertex filtration values of shape (n_points,)
            edge_filts: Edge filtration values of shape (n_edges,)
            edges: Edge indices of shape (n_edges, 2)
            pers_generators: Output from st.flag_persistence_generators()

        Returns:
            List of diagrams, one per homology dimension
        """
        diagrams = []
        n_vertices = int(pts.shape[0])
        edge_lookup = np.full((n_vertices, n_vertices), -1, dtype=np.int64)
        if len(edges) > 0:
            edge_lookup[edges[:, 0], edges[:, 1]] = np.arange(len(edges), dtype=np.int64)
            edge_lookup[edges[:, 1], edges[:, 0]] = np.arange(len(edges), dtype=np.int64)

        # H0 generators: (birth_vertex, death_edge)
        # pers_generators[0] has shape (n_pairs, 3): [birth_v, death_v1, death_v2]
        if 0 in self.homology_dimensions:
            h0_gens = pers_generators[0]
            if len(h0_gens) > 0:
                h0_gens = np.array(h0_gens)
                # Birth: vertex filtration
                birth_indices = torch.from_numpy(h0_gens[:, 0]).long().to(pts.device)
                births = vertex_filts[birth_indices]

                # Death: edge filtration
                death_edges = h0_gens[:, 1:]
                death_idx_np = edge_lookup[death_edges[:, 0], death_edges[:, 1]]
                death_indices = torch.from_numpy(death_idx_np).long().to(pts.device)
                deaths = edge_filts[death_indices]

                dgm_h0 = torch.stack([births, deaths], dim=-1)
            else:
                dgm_h0 = torch.zeros((0, 2), device=pts.device)
            diagrams.append(dgm_h0)

        # H1 generators: (birth_edge, death_edge)
        # pers_generators[1][0] has shape (n_pairs, 4): [birth_v1, birth_v2, death_v1, death_v2]
        if 1 in self.homology_dimensions:
            if len(pers_generators) > 1 and len(pers_generators[1]) > 0:
                h1_gens = np.array(pers_generators[1][0]) if len(pers_generators[1][0]) > 0 else np.array([]).reshape(0, 4)
                if len(h1_gens) > 0:
                    # Birth: edge filtration
                    birth_edges = h1_gens[:, :2]
                    birth_idx_np = edge_lookup[birth_edges[:, 0], birth_edges[:, 1]]
                    birth_indices = torch.from_numpy(birth_idx_np).long().to(pts.device)
                    births = edge_filts[birth_indices]

                    # Death: edge filtration
                    death_edges = h1_gens[:, 2:]
                    death_idx_np = edge_lookup[death_edges[:, 0], death_edges[:, 1]]
                    death_indices = torch.from_numpy(death_idx_np).long().to(pts.device)
                    deaths = edge_filts[death_indices]

                    dgm_h1 = torch.stack([births, deaths], dim=-1)
                else:
                    dgm_h1 = torch.zeros((0, 2), device=pts.device)
            else:
                dgm_h1 = torch.zeros((0, 2), device=pts.device)
            diagrams.append(dgm_h1)

        return diagrams
 
  
    def forward_single(self, data) -> List[torch.Tensor]:
        """
        Process a single point cloud.
        Supporta sia un torch.Tensor (grezzo) che un dict (pre-calcolato).
        """
        # 1. Identificazione del tipo di input e setup dati base
        if isinstance(data, torch.Tensor):
            pts = data
            device = pts.device
            # Caso standard: dobbiamo calcolare i grafi ora
            topo = self._get_or_build_topology(pts)
            edges = topo['edges']
            knn_dist = topo['knn'].to(device, non_blocking=True) if topo['knn'] is not None else None
            edge_index_tensor = None
            edge_distances = None
        else:
            # Caso ottimizzato: i dati sono già nel dizionario
            pts = data['pts']
            device = pts.device
            edges = data.get('alpha_graph')
            edge_index_tensor = data.get('edge_index_tensor')
            edge_distances = data.get('edge_distances')
            if edges is None:
                topo = self._get_or_build_topology(pts)
                edges = topo['edges']
            # Se l'MLP usa le distanze ma non sono nel dict, le calcoliamo
            knn_dist = data.get('knn_distances')
            if knn_dist is None and not self.use_coords:
                topo = self._get_or_build_topology(pts)
                if topo['knn'] is not None:
                    knn_dist = topo['knn'].to(device, non_blocking=True)
                else:
                    knn_dist = self.compute_knn_distances(pts)

        # 2. Vertex Filtration (MLP)
        # Scegliamo se passare coordinate o distanze kNN
        mlp_input = pts if self.use_coords else knn_dist
        vertex_filts = self.mlp(mlp_input)

        # 3. Controllo archi
        if len(edges) == 0:
            return [torch.zeros((0, 2), device=device) for _ in self.homology_dimensions]

        if edge_index_tensor is None:
            edge_index_tensor = torch.from_numpy(edges).long().to(device)

        # 4. Compute edge filtrations (Differenziabile)
        edge_filts = self.compute_edge_filtration(
            pts, vertex_filts, edges,
            edge_index_tensor=edge_index_tensor,
            edge_distances=edge_distances,
        )

        # 5. Update Simplex Tree (Calcolo non differenziabile della persistenza)
        vertex_filts_np = vertex_filts.detach().cpu().numpy()
        edge_filts_np = edge_filts.detach().cpu().numpy()
        
        # Creiamo un nuovo Simplex Tree con le filtrazioni apprese
        # Assicurati che il tuo metodo update_filtration accetti questi argomenti
        new_st = self.update_filtration(vertex_filts_np, edge_filts_np, edges)

        # 6. Calcolo Persistenza e Generatori
        new_st.compute_persistence()
        pers_generators = new_st.flag_persistence_generators()

        # 7. Estrazione Diagrammi (Differenziabile tramite torch.gather)
        diagrams = self.extract_diagrams_single(
            pts, vertex_filts, edge_filts, edges, pers_generators
        )

        return diagrams

    @staticmethod
    def _gudhi_worker(update_fn, vf_np, ef_np, edges, max_hom_dim):
        """Run GUDHI persistence in a worker.

        Uses insert_batch for vectorised C++ insertion.
        """
        new_st = gudhi.SimplexTree()

        # Batch-insert vertices
        n_v = len(vf_np)
        vert_arr = np.arange(n_v, dtype=np.int32).reshape(1, -1)
        new_st.insert_batch(vert_arr, vf_np.astype(np.float64))

        # Batch-insert edges
        if len(edges) > 0:
            edge_arr = edges.T.astype(np.int32)
            new_st.insert_batch(edge_arr, ef_np.astype(np.float64))

        new_st.expansion(max_hom_dim + 1)
        new_st.make_filtration_non_decreasing()
        new_st.compute_persistence()
        return new_st.flag_persistence_generators()

    def forward(self, X) -> List[List[torch.Tensor]]:
        """
        Compute persistence diagrams for a batch of point clouds.

        Three-phase pipeline:
          Phase A (GPU, main thread): MLP + edge filtrations for all samples
          Phase B (CPU, thread pool): Parallel GUDHI persistence
          Phase C (GPU, main thread): Differentiable diagram extraction

        GUDHI C++ releases the GIL, so ThreadPoolExecutor gives real
        parallelism for Phase B (the dominant cost).

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

        # ── Phase A: GPU forward (MLP + edge filtrations) ──────────────
        per_sample = []  # list of dicts with torch tensors + numpy arrays
        iterator = (
            tqdm(range(n_samples), desc="Phase A: MLP + edge filts")
            if self.show_progress
            else range(n_samples)
        )

        for i in iterator:
            data = samples[i]

            # Parse input
            if isinstance(data, torch.Tensor):
                pts = data
                device = pts.device
                topo = self._get_or_build_topology(pts)
                edges = topo['edges']
                knn_dist = topo['knn'].to(device, non_blocking=True) if topo['knn'] is not None else None
                edge_index_tensor = None
                edge_distances = None
            else:
                pts = data['pts']
                device = pts.device
                edges = data.get('alpha_graph')
                edge_index_tensor = data.get('edge_index_tensor')
                edge_distances = data.get('edge_distances')
                if edges is None:
                    topo = self._get_or_build_topology(pts)
                    edges = topo['edges']
                knn_dist = data.get('knn_distances')
                if knn_dist is None and not self.use_coords:
                    topo = self._get_or_build_topology(pts)
                    if topo['knn'] is not None:
                        knn_dist = topo['knn'].to(device, non_blocking=True)
                    else:
                        knn_dist = self.compute_knn_distances(pts)

            # MLP → vertex filtrations
            mlp_input = pts if self.use_coords else knn_dist
            vertex_filts = self.mlp(mlp_input)

            if len(edges) == 0:
                per_sample.append(None)
                continue

            if edge_index_tensor is None:
                edge_index_tensor = torch.from_numpy(edges).long().to(device)

            # Edge filtrations (differentiable)
            edge_filts = self.compute_edge_filtration(
                pts, vertex_filts, edges,
                edge_index_tensor=edge_index_tensor,
                edge_distances=edge_distances,
            )

            # Detach to numpy for GUDHI
            vf_np = vertex_filts.detach().cpu().numpy()
            ef_np = edge_filts.detach().cpu().numpy()

            per_sample.append({
                'pts': pts,
                'vertex_filts': vertex_filts,
                'edge_filts': edge_filts,
                'edges': edges,
                'vf_np': vf_np,
                'ef_np': ef_np,
            })

        # ── Phase B: Parallel GUDHI persistence ───────────────────────
        # Submit all GUDHI jobs to thread pool (C++ releases GIL)
        generators = [None] * n_samples

        # Collect indices of samples that have edges (non-None)
        active_indices = [i for i, s in enumerate(per_sample) if s is not None]

        if self.n_gudhi_workers > 1 and len(active_indices) > 1:
            with ThreadPoolExecutor(max_workers=self.n_gudhi_workers) as pool:
                futures = {}
                for i in active_indices:
                    s = per_sample[i]
                    f = pool.submit(
                        self._gudhi_worker, None,
                        s['vf_np'], s['ef_np'], s['edges'],
                        self.max_hom_dim,
                    )
                    futures[i] = f
                for i, f in futures.items():
                    generators[i] = f.result()
        else:
            # Sequential fallback
            for i in active_indices:
                s = per_sample[i]
                generators[i] = self._gudhi_worker(
                    None, s['vf_np'], s['ef_np'], s['edges'],
                    self.max_hom_dim,
                )

        # ── Phase C: Differentiable diagram extraction ─────────────────
        for i in range(n_samples):
            if per_sample[i] is None:
                # No edges → empty diagrams
                device = samples[i].device if isinstance(samples[i], torch.Tensor) else samples[i]['pts'].device
                for dim_idx in range(len(self.homology_dimensions)):
                    all_diagrams[dim_idx].append(
                        torch.zeros((0, 2), device=device)
                    )
                continue

            s = per_sample[i]
            diagrams = self.extract_diagrams_single(
                s['pts'], s['vertex_filts'], s['edge_filts'],
                s['edges'], generators[i],
            )
            for dim_idx, dgm in enumerate(diagrams):
                all_diagrams[dim_idx].append(dgm)

        return all_diagrams

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)