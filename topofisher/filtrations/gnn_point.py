import string
from typing import List, Optional, Literal, Union
import torch
import torch.nn as nn
import numpy as np
import gudhi
from tqdm import tqdm
from torch_geometric.nn import GCNConv 

class VertexGNN(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=None, dropout_rate=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.hidden_dims = list(hidden_dims)
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        prev_dim = input_dim
        for h in self.hidden_dims:
            self.convs.append(GCNConv(prev_dim, h))
            self.lns.append(nn.LayerNorm(h))
            prev_dim = h

        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(prev_dim, 1)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x, edge_index):
        h = x
        for conv, ln in zip(self.convs, self.lns):
            out = self.activation(ln(conv(h, edge_index)))
            out = self.dropout(out)
            h = out
        return 5.0 * torch.tanh(self.head(h)).squeeze(-1)


class LearnableGNNPointFiltration(nn.Module):
    def __init__(
        self,
        k: int = 10,
        positional_encoding: str = "none",
        homology_dimensions: List[int] = [0, 1],
        hidden_dims: List[int] = [10, 20],
        edge_term: bool = True,
        p: float = 1.0,
        dropout_rate: float = 0.0,
        max_edge: float = 1.0,
        show_progress: bool = False,
        graph: str = 'knn',
        data_dim: int = 2
    ):
        super().__init__()
        self.k = k
        self.positional_encoding = positional_encoding
        self.homology_dimensions = homology_dimensions
        self.max_hom_dim = max(homology_dimensions)
        self.max_edge = max_edge
        self.p = p
        self.show_progress = show_progress
        self.graph = graph
        self.data_dim = data_dim
        self.hidden_dims = hidden_dims 
        self.edge_term = edge_term
        self.dropout_rate = dropout_rate
        self.pre_compute = True # Abilita l'ottimizzazione nella pipeline

        if self.positional_encoding == "none":
            input_dim = self.k
        elif self.positional_encoding == "also":
            input_dim = self.k + self.data_dim
        else:
            input_dim = self.data_dim

        self.gnn = VertexGNN(input_dim=input_dim, hidden_dims=self.hidden_dims, dropout_rate=self.dropout_rate)

    def compute_knn(self, pts: torch.Tensor):
        n = pts.size(0)
        neigh_k = min(self.k, max(n - 1, 0))
        k_plus_1 = min(self.k + 1, n)
        dists = torch.cdist(pts, pts)
        knn_vals, knn_idx = torch.topk(dists, k=k_plus_1, dim=-1, largest=False)

        knn_dists = knn_vals[:, 1:k_plus_1]
        if knn_dists.shape[1] < self.k:
            pad_size = self.k - knn_dists.shape[1]
            knn_dists = torch.nn.functional.pad(knn_dists, (0, pad_size), value=0.0)

        if neigh_k == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=pts.device)
        else:
            dst = knn_idx[:, 1:k_plus_1].reshape(-1)
            src = torch.arange(n, device=pts.device).repeat_interleave(neigh_k)
            edge_index = torch.stack([src, dst], dim=0)

        return knn_dists, edge_index

    def compute_edge_filtration(
        self, 
        vertex_filts: torch.Tensor, 
        edges: np.ndarray,
        device: torch.device,
        pts: torch.Tensor
    ) -> torch.Tensor:
        edges_tensor = torch.from_numpy(edges).long().to(device)
        f_u = vertex_filts[edges_tensor[:, 0]]
        f_v = vertex_filts[edges_tensor[:, 1]]
        fmax = torch.maximum(f_u, f_v)

        if not self.edge_term:
            return fmax
        
        pts_u = pts[edges_tensor[:, 0]]
        pts_v = pts[edges_tensor[:, 1]]
        d = torch.linalg.norm(pts_u - pts_v, dim=-1)

        if self.p == 1.0:
            return d + fmax
        else:
            return (d ** self.p + fmax ** self.p) ** (1 / self.p)

    def get_simplex_tree(self, pts_np: np.ndarray) -> gudhi.SimplexTree:
        alpha = gudhi.AlphaComplex(points=pts_np)
        return alpha.create_simplex_tree(default_filtration_value=False)

    def update_filtration(
        self,
        vertex_filts_np: np.ndarray,
        edge_filts_np: np.ndarray,
        edges: np.ndarray
    ) -> gudhi.SimplexTree:
        new_st = gudhi.SimplexTree()
        for i, filt in enumerate(vertex_filts_np):
            new_st.insert([i], filtration=float(filt))
        for (i, j), filt in zip(edges, edge_filts_np):
            new_st.insert([i, j], filtration=float(filt))
        
        new_st.expansion(self.max_hom_dim + 1)
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
        diagrams = []
        edge_to_idx = {(min(i, j), max(i, j)): idx for idx, (i, j) in enumerate(edges)}

        if 0 in self.homology_dimensions:
            h0_gens = pers_generators[0]
            if len(h0_gens) > 0:
                h0_gens = np.array(h0_gens)
                births = vertex_filts[torch.from_numpy(h0_gens[:, 0]).long().to(pts.device)]
                death_indices = torch.tensor([edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))] for e in h0_gens[:, 1:]], device=pts.device)
                deaths = edge_filts[death_indices]
                diagrams.append(torch.stack([births, deaths], dim=-1))
            else:
                diagrams.append(torch.zeros((0, 2), device=pts.device))

        if 1 in self.homology_dimensions:
            if len(pers_generators) > 1 and len(pers_generators[1]) > 0:
                h1_gens = np.array(pers_generators[1][0])
                if len(h1_gens) > 0:
                    birth_indices = torch.tensor([edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))] for e in h1_gens[:, :2]], device=pts.device)
                    death_indices = torch.tensor([edge_to_idx[(min(e[0], e[1]), max(e[0], e[1]))] for e in h1_gens[:, 2:]], device=pts.device)
                    diagrams.append(torch.stack([edge_filts[birth_indices], edge_filts[death_indices]], dim=-1))
                else:
                    diagrams.append(torch.zeros((0, 2), device=pts.device))
            else:
                diagrams.append(torch.zeros((0, 2), device=pts.device))
        return diagrams

    def forward_single(self, data) -> List[torch.Tensor]:
        if isinstance(data, torch.Tensor):
            pts = data
            device = pts.device
            pts_np = pts.detach().cpu().numpy()
            st_temp = self.get_simplex_tree(pts_np)
            edges = np.array([s[0] for s in st_temp.get_skeleton(1) if len(s[0]) == 2])
            knn_dists, knn_edge_index = self.compute_knn(pts)
        else:
            pts = data['pts']
            device = pts.device
            edges = data['alpha_graph']
            knn_dists = data.get('knn_distances')
            knn_edge_index = None
            if knn_dists is None:
                knn_dists, knn_edge_index = self.compute_knn(pts)

        # Selezione del grafo per la GNN
        if self.graph == 'knn':
            if knn_edge_index is None: # Se non calcolato sopra
                _, edge_index = self.compute_knn(pts)
            else:
                edge_index = knn_edge_index
        else: # Alpha graph
            edge_index = torch.from_numpy(edges).long().t().contiguous().to(device)

        # Simmetrizzazione per Message Passing
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Positional Encoding
        if self.positional_encoding == "none": x = knn_dists
        elif self.positional_encoding == "also": x = torch.cat([knn_dists, pts], dim=-1)
        else: x = pts

        vertex_filts = self.gnn(x, edge_index)
        edge_filts = self.compute_edge_filtration(vertex_filts, edges, device=device, pts=pts)

        vertex_filts_np = vertex_filts.detach().cpu().numpy()
        edge_filts_np = edge_filts.detach().cpu().numpy()
        
        new_st = self.update_filtration(vertex_filts_np, edge_filts_np, edges)
        new_st.compute_persistence()
        pers_generators = new_st.flag_persistence_generators()

        return self.extract_diagrams_single(pts, vertex_filts, edge_filts, edges, pers_generators)

    def forward(self, X) -> List[List[torch.Tensor]]:
        if isinstance(X, torch.Tensor):
            if X.ndim == 2: X = X.unsqueeze(0)
            samples = [{'pts': X[i]} for i in range(X.shape[0])]
        else:
            samples = X

        all_diagrams = [[] for _ in self.homology_dimensions]
        iterator = tqdm(range(len(samples)), desc="GNN Flag Complex") if self.show_progress else range(len(samples))
        
        for i in iterator:
            sample_diagrams = self.forward_single(samples[i])
            for dim_idx, dgm in enumerate(sample_diagrams):
                all_diagrams[dim_idx].append(dgm)
        return all_diagrams