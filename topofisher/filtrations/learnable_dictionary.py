"""
Learnable Dictionary Filtration: M learned views of κ → M × cubical persistence.

Instead of downsampling through a large CNN, this filtration applies M learned
pointwise transforms to the input field, each producing a different topological
"view." Each view is a linear combination of K physics-motivated feature maps
(identity, gradient magnitude, Laplacian, etc.), with only M×K learnable
parameters (~40 for M=4, K=10).

Architecture:
    κ (input_size × input_size)
        → AvgPool to (target_size × target_size)
        → Compute K feature maps: {identity, |∇κ|, ∇²κ, det(H), κ², ...}
        → M linear heads: f_m = Σ_k α_{m,k} · φ_k   (m = 1..M)
        → Per-sample standardize each f_m
        → M × cubical persistence → M × {H0, H1} diagrams
        → Return M×2 sets of diagrams (compatible with CombinedVectorization)

The key insight: topology depends only on pixel *ordering*, not values.
Different linear combinations of features create genuinely different orderings
(and thus different persistence diagrams), each capturing complementary
topological information about the field.

Physics motivation for the feature dictionary:
    identity      : κ             → peaks/voids of convergence
    gradient_mag  : |∇κ|          → filamentary structure
    laplacian     : ∇²κ           → curvature: concentrated vs diffuse mass
    hessian_det   : det H(κ)      → peak/void classifier
    squared       : κ²            → non-Gaussianity probe
    smooth_σ      : G_σ * κ       → structure at physical scale σ
    highpass_σ    : κ - G_σ * κ   → small-scale detail
    phase_only    : F⁻¹(F[κ]/|F[κ]|) → all non-Gaussian information
    local_std     : σ_local(κ)    → local variability

Advantages over CNN-based LTDA:
    - 40 params vs 100K+ → vastly more parameter-efficient
    - Fully interpretable: each α_m is a 1D weight vector you can inspect
    - No information bottleneck: M independent views, not one squeezed channel
    - No spatial convolutions: features are physics-derived, not learned filters
    - Faster: feature computation is microseconds, persistence dominates runtime
"""
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.dictionary import (
    FEATURE_REGISTRY, DEFAULT_FEATURES, _ensure_3d
)
from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class LearnableDictionaryFiltration(nn.Module):
    """
    M learned linear combinations of K dictionary features → M × cubical persistence.

    Each of the M heads produces a scalar field as a weighted sum of K feature maps.
    These M fields are fed independently through cubical persistence, yielding
    M × n_hom_dims sets of persistence diagrams.

    Learnable parameters: M × K weights (e.g., 4 × 10 = 40 params).

    Example:
        >>> filt = LearnableDictionaryFiltration(
        ...     n_heads=4,
        ...     feature_names=['identity', 'gradient_mag', 'laplacian', 'squared'],
        ...     target_size=64, input_size=512,
        ... )
        >>> x = torch.randn(10, 512, 512)
        >>> diagrams = filt(x)  # List of 4*2=8 sets of diagrams
        >>> len(diagrams)  # 8 (4 heads × 2 hom dims)
        8
    """

    def __init__(
        self,
        n_heads: int = 4,
        feature_names: Optional[List[str]] = None,
        target_size: int = 64,
        input_size: int = 512,
        homology_dimensions: List[int] = [0, 1],
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        persistence_backend: str = 'gudhi_gpu',
        persistence_construction: str = 'V',
        gpu_sub_batch_size: int = 100,
        alpha_init: Optional[List] = None,
        freeze_alpha: bool = False,
        periodic: bool = False,
        standardize: bool = True,
    ):
        """
        Initialize learnable dictionary filtration.

        Args:
            n_heads: Number of independent filtration heads (M). Each learns
                a different linear combination of features.
            feature_names: List of feature names from the dictionary.
                Defaults to all 10 features in DEFAULT_FEATURES.
            target_size: Spatial resolution for persistence computation.
                Input is downsampled to this size via AvgPool.
            input_size: Expected input spatial resolution.
            homology_dimensions: Homology dimensions to compute (e.g., [0, 1]).
            min_persistence: Minimum persistence threshold per dimension.
            superlevel: If True, compute superlevel filtration.
            persistence_backend: Backend for cubical persistence computation.
            persistence_construction: Cubical complex construction ('T' or 'V').
            gpu_sub_batch_size: Sub-batch size for GPU persistence.
        """
        super().__init__()

        self.feature_names = feature_names or list(DEFAULT_FEATURES)
        self.K = len(self.feature_names)
        self.n_heads = n_heads
        self.target_size = target_size
        self.input_size = input_size
        self.homology_dimensions = homology_dimensions
        self.n_hom = len(homology_dimensions)

        # Validate feature names
        for name in self.feature_names:
            if name not in FEATURE_REGISTRY:
                raise ValueError(
                    f"Unknown feature: {name}. "
                    f"Available: {list(FEATURE_REGISTRY.keys())}"
                )

        # AvgPool for downsampling (parameter-free)
        pool_size = input_size // target_size
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)

        # M learnable mixing heads: shape (M, K)
        if alpha_init is not None:
            # Accept list, list-of-lists, or tensor; broadcast/reshape to (M, K).
            alpha_tensor = torch.tensor(alpha_init, dtype=torch.float32)
            if alpha_tensor.ndim == 1:
                # Single alpha vector (K,) → use for all heads
                alpha_tensor = alpha_tensor.unsqueeze(0).expand(n_heads, -1).contiguous()
            if alpha_tensor.shape != (n_heads, self.K):
                raise ValueError(
                    f"alpha_init has shape {list(alpha_tensor.shape)}, "
                    f"expected ({n_heads}, {self.K})"
                )
            alpha = alpha_tensor
        else:
            # Default: initialize each head as a one-hot on a different feature,
            # so heads start at physically distinct topological views.
            alpha = torch.zeros(n_heads, self.K)
            for m in range(n_heads):
                feat_idx = m % self.K  # cycle through features if M > K
                alpha[m, feat_idx] = 1.0
        self.alpha = nn.Parameter(alpha, requires_grad=not freeze_alpha)

        self.standardize = standardize

        # Cubical persistence layer (shared across all heads)
        self.cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=min_persistence,
            superlevel=superlevel,
            backend=persistence_backend,
            construction=persistence_construction,
            gpu_sub_batch_size=gpu_sub_batch_size,
            periodic=periodic,
        )

    def _compute_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all K feature maps on (already downsampled) input.

        Args:
            x: (B, H, W) downsampled input fields.

        Returns:
            (B, K, H, W) tensor of feature maps.
        """
        maps = []
        for name in self.feature_names:
            fn = FEATURE_REGISTRY[name]
            maps.append(fn(x))
        return torch.stack(maps, dim=1)  # (B, K, H, W)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Apply M learned filtrations + cubical persistence to input fields.

        Args:
            x: Input tensor in one of two layouts:
               - (B, H, W) or (H, W): raw maps; the K dictionary features are
                 computed on-the-fly via _compute_feature_maps.
               - (B, K, H, W): precomputed feature maps where the K axis matches
                 self.K. Skips _compute_feature_maps (used with cached features
                 produced by experiments/.../precompute_dict_features.py).

        Returns:
            List[List[Tensor]] with M × n_hom entries (virtual homology dims).
            Ordered as [head0_H0, head0_H1, head1_H0, head1_H1, ...].
            Each inner list has B diagrams, each of shape (n_pairs, 2).
        """
        # Detect precomputed-features layout: (B, K, H, W) with matching K.
        if x.ndim == 4 and x.shape[1] == self.K:
            feat_maps = x.float() if x.dtype != torch.float32 else x
            B = feat_maps.shape[0]
        else:
            x = _ensure_3d(x)
            B = x.shape[0]

            # 1. Downsample: (B, H, W) → (B, target_size, target_size)
            x_down = self.pool(x.unsqueeze(1)).squeeze(1)

            # 2. Compute K feature maps: (B, K, H', W')
            feat_maps = self._compute_feature_maps(x_down)

        # 3. Mix: M heads × K features → M scalar fields
        #    alpha: (M, K), feat_maps: (B, K, H', W') → mixed: (B, M, H', W')
        mixed = torch.einsum('mk,bkhw->bmhw', self.alpha, feat_maps)

        # 4. Reshape to (B*M, H', W') for batch persistence
        BM = B * self.n_heads
        mixed_flat = mixed.reshape(BM, *mixed.shape[2:])

        # 5. Per-sample standardize (optional; normalize value range for stable
        #    persistence, but KILLS amplitude-based cosmological information such
        #    as Omega_m / sigma_8 → set standardize=False for PI+MOPED pipelines)
        if self.standardize:
            m = mixed_flat.mean(dim=(-2, -1), keepdim=True)
            s = mixed_flat.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            mixed_flat = (mixed_flat - m) / s

        # 6. Cubical persistence on all B*M fields
        #    Returns List[List[Tensor]]: [hom_dim][sample_in_batch]
        all_diagrams = self.cubical(mixed_flat)  # n_hom lists, each BM entries

        # 7. Rearrange into M × n_hom virtual homology dimensions
        #    mixed was (B, M, H, W) reshaped to (B*M, H, W) in row-major order:
        #    index i*M + m corresponds to sample i, head m
        result = []
        for m_idx in range(self.n_heads):
            for h_idx in range(self.n_hom):
                head_diagrams = [
                    all_diagrams[h_idx][i * self.n_heads + m_idx]
                    for i in range(B)
                ]
                result.append(head_diagrams)

        return result

    def get_alpha_summary(self) -> str:
        """Return human-readable summary of learned head weights."""
        lines = []
        for m in range(self.n_heads):
            weights = self.alpha[m].detach().cpu()
            sorted_idx = torch.argsort(weights.abs(), descending=True)
            top3 = [(self.feature_names[i], weights[i].item()) for i in sorted_idx[:3]]
            top_str = ', '.join(f'{name}={w:.3f}' for name, w in top3)
            lines.append(f"  Head {m}: [{top_str}]")
        return '\n'.join(lines)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"LearnableDictionaryFiltration("
            f"n_heads={self.n_heads}, K={self.K}, "
            f"target_size={self.target_size}, "
            f"params={self.get_num_parameters()})"
        )
