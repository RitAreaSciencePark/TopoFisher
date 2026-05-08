"""
GNN with readout for point clouds.
Optimized with pre-computation support and PyG Batching.
"""
import torch
import torch.nn as nn
import numpy as np
import gudhi
from typing import List, Optional, Literal, Union
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, Batch

class VertexGNNReadout(nn.Module):
    """
    Graph neural network that outputs multiple features per node.
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim

        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        prev_dim = input_dim
        for h in self.hidden_dims:
            self.convs.append(GCNConv(prev_dim, h))
            self.lns.append(nn.LayerNorm(h))
            prev_dim = h

        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(prev_dim, output_dim)
        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, ln in zip(self.convs, self.lns):
            out = self.activation(ln(conv(h, edge_index)))
            out = self.dropout(out)
            h = out
        return self.head(h)


class GNNReadout(nn.Module):
    def __init__(
        self,
        k: int = 10,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout_rate: float = 0.0,
        positional_encoding: str = "none",
        data_dim: int = 2,
        graph: str = "knn",
        max_edge: float = 1.0
    ):
        super().__init__()
        self.k = k
        self.output_dim = output_dim
        self.positional_encoding = positional_encoding
        self.data_dim = data_dim
        self.graph = graph
        self.max_edge = max_edge
        self.pre_compute = True # Fondamentale per attivare l'ottimizzazione nella pipeline

        if self.positional_encoding == "none":
            input_dim = self.k
        elif self.positional_encoding == "also":
            input_dim = self.k + self.data_dim
        else:
            input_dim = self.data_dim

        self.gnn = VertexGNNReadout(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )

    def get_simplex_tree(self, pts_np: np.ndarray) -> gudhi.SimplexTree:
        alpha = gudhi.AlphaComplex(points=pts_np)
        return alpha.create_simplex_tree(default_filtration_value=False)

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

    def forward(self, data_input: Union[torch.Tensor, List[dict]]) -> torch.Tensor:
        """
        Gestisce sia Tensor batch (B, N, d) che liste di campioni pre-calcolati.
        """
        device = next(self.parameters()).device
        
        # 1. Normalizzazione Input: Trasformiamo tutto in List[dict] se è un Tensor
        if isinstance(data_input, torch.Tensor):
            B, N, _ = data_input.shape
            samples = [{'pts': data_input[b]} for b in range(B)]
        else:
            samples = data_input

        data_list = []

        # 2. Loop sui campioni (veloce se i grafi sono pre-calcolati)
        for sample in samples:
            pts_b = sample['pts'].to(device)
            
            # Recupero o calcolo del grafo
            if self.graph == "alpha":
                if 'alpha_graph' in sample:
                    edges = sample['alpha_graph']
                else:
                    pts_np = pts_b.detach().cpu().numpy()
                    st = self.get_simplex_tree(pts_np)
                    edges = np.array([s[0] for s in st.get_skeleton(1) if len(s[0]) == 2])

                if len(edges) > 0:
                    edge_index = torch.from_numpy(edges).long().t().contiguous().to(device)
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            
            else: # kNN Graph
                # Se abbiamo le distanze pre-calcolate ma non gli indici, li ricalcoliamo
                # (Solitamente per il readout preferiamo ricalcolare kNN se non salvato)
                _, edge_index = self.compute_knn(pts_b)

            # Feature Encoding (kNN distances + Positional)
            knn_dists = sample.get('knn_distances')
            if knn_dists is None and self.positional_encoding != "only":
                knn_dists, _ = self.compute_knn(pts_b)
            
            if self.positional_encoding == "none":
                x = knn_dists
            elif self.positional_encoding == "also":
                x = torch.cat([knn_dists, pts_b], dim=-1)
            else: # "only"
                x = pts_b

            # Rendiamo il grafo non orientato
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            data_list.append(Data(x=x, edge_index=edge_index))

        # 3. Batching PyG e Forward unico
        batch = Batch.from_data_list(data_list).to(device)
        node_features = self.gnn(batch.x, batch.edge_index)

        # 4. Global Readout (Sum)
        return global_add_pool(node_features, batch.batch)