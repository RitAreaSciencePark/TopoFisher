"""
Hybrid vectorization for mixed vector + diagram outputs.

Handles the output of CNNGAPTDAFiltration which produces:
    [[gap_vectors], [H0_diagrams], [H1_diagrams]]

Applies IdentityVectorization to slot 0 (GAP vectors) and a separate
vectorization layer (e.g., DiffCurves) to each diagram slot, then
concatenates everything into a single feature vector per sample.
"""
from typing import List
import torch
import torch.nn as nn


class HybridVectorization(nn.Module):
    """
    Vectorization for hybrid filtration outputs (vectors + diagrams).

    Slot 0: vectors → stacked directly (identity)
    Slots 1+: persistence diagrams → vectorized by diagram_layers

    Args:
        diagram_layers: List of vectorization modules for diagram slots
                       (one per homology dimension, e.g., [DiffCurves_H0, DiffCurves_H1])
    """

    def __init__(self, diagram_layers: List[nn.Module]):
        super().__init__()
        self.diagram_layers = nn.ModuleList(diagram_layers)

    def fit(self, all_diagrams_list: List[List[List[torch.Tensor]]]):
        """
        Fit diagram vectorization layers on ALL data.

        Args:
            all_diagrams_list: List of output sets from filtration.
                Each set: [[gap_vecs], [H0_diags], [H1_diags]]
        """
        # Fit each diagram layer on its corresponding slot across ALL sets
        for hom_idx, layer in enumerate(self.diagram_layers):
            if hasattr(layer, 'fit'):
                # Slot 0 is GAP vectors; diagrams start at slot 1
                diagram_slot = hom_idx + 1
                all_data_for_dim = []
                for diagrams_set in all_diagrams_list:
                    all_data_for_dim.extend(diagrams_set[diagram_slot])
                layer.fit(all_data_for_dim)

    def forward(self, all_diagrams: List[List[torch.Tensor]]) -> torch.Tensor:
        """
        Vectorize hybrid output: stack GAP vectors + vectorize diagrams, concat.

        Args:
            all_diagrams: [[gap_vec_0, ...], [H0_diag_0, ...], [H1_diag_0, ...]]

        Returns:
            Tensor of shape (n_samples, gap_dim + sum_of_diagram_features)
        """
        features = []

        # Slot 0: GAP vectors — stack directly
        gap_vectors = all_diagrams[0]
        gap_stacked = torch.stack(gap_vectors)  # (n_samples, gap_dim)
        features.append(gap_stacked)
        device = gap_stacked.device

        # Slots 1+: persistence diagrams — vectorize with diagram layers
        for hom_idx, layer in enumerate(self.diagram_layers):
            diagram_slot = hom_idx + 1
            diagrams_for_dim = all_diagrams[diagram_slot]
            feat = layer(diagrams_for_dim).to(device)  # (n_samples, feat_dim)
            features.append(feat)

        return torch.cat(features, dim=-1)

    def __repr__(self):
        layer_reprs = [repr(layer) for layer in self.diagram_layers]
        return f"HybridVectorization(diagram_layers=[{', '.join(layer_reprs)}])"
