"""
Wavelet Persistence Filtration: Morlet wavelets → learned attention → cubical persistence.

This module implements a physics-informed learnable TDA filtration that:
1. Decomposes the input field into wavelet modulus maps U₁(j,l) = |κ * ψ_{j,l}|
   using the Morlet wavelet filter bank (non-learnable, deterministic)
2. Learns M linear combinations (heads) of the U₁ maps via attention weights α
3. Downsamples each combined map to target_size × target_size
4. Computes cubical persistence on each downsampled map
5. Returns persistence diagrams for downstream vectorization (PI) + compression (MOPED)

Physics motivation:
    The wavelet modulus |κ * ψ_{j,l}| isolates structures at scale 2^j and
    orientation l. For weak lensing convergence maps (lognormal fields):
    - Fine scales (small j) capture halo peaks — highly non-Gaussian
    - Coarse scales (large j) capture large-scale structure — contains
      cosmological parameter information
    - The modulus is a nonlinear operation that breaks Gaussianity, enabling
      persistence to capture non-Gaussian spatial topology
    - Wavelet modulus maps have MUCH higher persistence signal than raw κ:
      at j=5, H₁ persistence is ~250× larger than raw κ (diagnostics result)

    Running persistence on U₁ maps captures the spatial topology of scale-
    specific non-Gaussian structures — complementary to the scattering
    transform's S₂ coefficients which capture the same information via
    second-order convolution + spatial averaging (discarding spatial layout).

Learnability:
    The learnable attention weights α define WHICH wavelet filtration functions
    to perform persistence on. Each head m learns a different linear combination
    of U₁ maps → a different topological "view" of the field. Fisher loss
    drives the learning: the attention weights converge to the combination
    of scales/orientations that maximally constrains cosmological parameters.

    Parameter count: M × n_wavelets (e.g., 4 × 7 = 28 parameters for M=4 heads,
    J=7 scales × 1 orientation/scale). Extremely lightweight.

Computational design:
    - Wavelet convolutions in Fourier domain (FFT) — fast on GPU
    - Isotropy: lensing fields are statistically isotropic, so we use L_probe=1
      orientation per scale (diagnostics confirm l=0 ≈ l=4 to <1%)
    - Scale-adapted downsampling: U₁ maps at scale 2^j have effective resolution
      ~512/2^j, so downsampling to 64×64 is safe for j≥2
    - V-construction cubical persistence via GUDHI GPU for maximum Fisher info
    - skip_k topology caching for training speedup

Example:
    >>> filt = WaveletPersistenceFiltration(M=512, N=512, J=7, n_heads=4)
    >>> x = torch.randn(50, 512, 512)
    >>> diagrams = filt(x)  # List[List[Tensor]] for each hom dim
"""
import math
from typing import List, Optional
import torch
import torch.nn as nn

from topofisher.filtrations.scattering import morlet_2d, build_filter_bank
from topofisher.filtrations.differentiable_cubical import DifferentiableCubicalLayer


class WaveletPersistenceFiltration(nn.Module):
    """
    Learnable wavelet persistence filtration for 2D fields.

    Computes Morlet wavelet modulus maps → learned attention mixing →
    downsample → cubical persistence.

    Args:
        M: Spatial height of input (default 512)
        N: Spatial width of input (default 512)
        J: Number of wavelet scales/octaves (default 7)
        L_probe: Number of orientations to probe per scale (default 1).
            For isotropic fields, 1 is sufficient (diagnostics confirm this).
        j_min: Minimum scale index to include (default 2).
            j=0,1 have noise-dominated persistence. j≥2 has meaningful signal.
        j_max: Maximum scale index to include (default 6, exclusive).
            j=6 at 512×512 has effective resolution ~8px — too coarse.
        n_heads: Number of attention heads M (default 4).
            Each head learns a different linear combination of U₁ maps.
        target_size: Resolution to downsample combined maps before persistence
            (default 64). Persistence at 64×64 takes ~0.2s/sample (CPU).
        homology_dimensions: Which homology dimensions to compute (default [0,1])
        persistence_backend: Backend for cubical persistence
            ('gudhi_gpu' recommended for lensing on DGX, 'gudhi' for CPU)
        persistence_construction: 'V' (recommended) or 'T'
        standardize: Per-sample standardize before persistence (default True)
        log_modulus: Apply log1p to wavelet modulus maps (default True).
            Reduces dynamic range (U₁ values span orders of magnitude across
            scales). Stabilizes persistence diagram distributions.
        skip_k: Topology caching frequency for training (default 5)
        gpu_sub_batch_size: Sub-batch size for GUDHI GPU (default 5000)
        min_persistence: Minimum persistence threshold per hom dim (default None)
        superlevel: If True, compute superlevel filtration (default False)
        separate_heads: If True, output separate diagram groups per head × hom
            (n_heads * n_hom groups). If False, concatenate diagrams across heads
            for each hom dim (n_hom groups, denser diagrams). Default False.
    """

    def __init__(
        self,
        M: int = 512,
        N: int = 512,
        J: int = 7,
        L_probe: int = 1,
        j_min: int = 2,
        j_max: int = 6,
        n_heads: int = 4,
        target_size: int = 64,
        homology_dimensions: List[int] = [0, 1],
        persistence_backend: str = 'gudhi_gpu',
        persistence_construction: str = 'V',
        standardize: bool = True,
        log_modulus: bool = True,
        skip_k: int = 5,
        gpu_sub_batch_size: int = 5000,
        min_persistence: Optional[List[float]] = None,
        superlevel: bool = False,
        separate_heads: bool = False,
    ):
        super().__init__()

        self.M = M
        self.N = N
        self.J = J
        self.L_probe = L_probe
        self.j_min = j_min
        self.j_max = j_max
        self.n_heads = n_heads
        self.target_size = target_size
        self.standardize = standardize
        self.log_modulus = log_modulus
        self.homology_dimensions = homology_dimensions
        self.separate_heads = separate_heads
        # Output dimensions: n_heads * n_hom if separate, else n_hom
        if separate_heads:
            self.dimensions = list(range(n_heads * len(homology_dimensions)))
        else:
            self.dimensions = homology_dimensions

        # ── Build wavelet filter bank ──────────────────────────────────
        # Use L=8 for filter bank construction (standard), but only keep
        # L_probe orientations per scale (default 1 for isotropic fields)
        L_full = 8
        fb = build_filter_bank(M, N, J, L_full)

        # Select probe filters: j_min ≤ j < j_max, L_probe orientations
        # Orientations are evenly spaced: l = 0, L_full//L_probe, 2*L_full//L_probe, ...
        probe_indices = []
        probe_jl = []
        l_step = L_full // L_probe
        for j, l, psi_fft in fb['psi']:
            if j_min <= j < j_max and l % l_step == 0 and l // l_step < L_probe:
                probe_indices.append(j * L_full + l)
                probe_jl.append((j, l))

        self.probe_jl = probe_jl  # list of (j, l) tuples
        self.n_wavelets = len(probe_jl)

        # Store only probe wavelet FFTs as registered buffers
        psi_ffts_all = []
        for j, l, psi_fft in fb['psi']:
            psi_ffts_all.append(psi_fft)
        psi_ffts_tensor = torch.stack(psi_ffts_all, dim=0)  # (J*L_full, M, N)

        # Extract probe subset
        probe_idx_in_bank = [j * L_full + l for j, l in probe_jl]
        probe_ffts = psi_ffts_tensor[probe_idx_in_bank]  # (n_wavelets, M, N)
        self.register_buffer('psi_ffts', probe_ffts)

        # ── Learnable attention weights ────────────────────────────────
        # α: (n_heads, n_wavelets) — each head produces a weighted sum of U₁ maps
        # Initialize: each head attends primarily to one wavelet scale
        # (spread across the available scales)
        alpha_init = torch.zeros(n_heads, self.n_wavelets)
        for m in range(n_heads):
            # Distribute heads across wavelets, roughly evenly
            primary_idx = (m * self.n_wavelets) // n_heads
            alpha_init[m, primary_idx] = 2.0  # strong initial attention
            # Small attention on neighbors for smooth initialization
            for offset in [-1, 1]:
                neighbor = primary_idx + offset
                if 0 <= neighbor < self.n_wavelets:
                    alpha_init[m, neighbor] = 0.5

        self.alpha = nn.Parameter(alpha_init)

        # ── Downsampling (non-learnable) ───────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(target_size)

        # ── Cubical persistence ────────────────────────────────────────
        # One shared persistence layer for all heads (same config)
        n_jobs = 8 if persistence_backend == 'gudhi' else 1
        self.cubical = DifferentiableCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=min_persistence,
            superlevel=superlevel,
            n_jobs=n_jobs,
            skip_k=skip_k,
            backend=persistence_backend,
            construction=persistence_construction,
            gpu_sub_batch_size=gpu_sub_batch_size,
        )

        # For sub-batching CNN eval to avoid OOM
        self.cnn_eval_batch_size = 200

        print(f"WaveletPersistenceFiltration initialized:")
        print(f"  Wavelets: {self.n_wavelets} probes at (j,l) = {probe_jl}")
        print(f"  Heads: {n_heads}, learnable params: {n_heads * self.n_wavelets}")
        print(f"  Target size: {target_size}×{target_size}")
        print(f"  Backend: {persistence_backend} ({persistence_construction}-construction)")
        print(f"  log_modulus={log_modulus}, standardize={standardize}")

    def _compute_u1_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute wavelet modulus maps U₁(j,l) = |κ * ψ_{j,l}| for probe filters.

        Args:
            x: (B, H, W) input field

        Returns:
            (B, n_wavelets, H, W) tensor of wavelet modulus maps
        """
        B = x.shape[0]
        x_fft = torch.fft.fft2(x)  # (B, M, N) complex

        u1_maps = []
        for idx in range(self.n_wavelets):
            # Convolve in Fourier domain and take modulus
            psi_fft = self.psi_ffts[idx]  # (M, N) complex
            out_fft = x_fft * psi_fft.unsqueeze(0)  # (B, M, N)
            out = torch.fft.ifft2(out_fft).real  # (B, M, N)
            u1 = torch.abs(out)  # (B, M, N)

            if self.log_modulus:
                u1 = torch.log1p(u1)

            u1_maps.append(u1)

        return torch.stack(u1_maps, dim=1)  # (B, n_wavelets, H, W)

    def _apply_attention(self, u1_maps: torch.Tensor) -> torch.Tensor:
        """
        Apply learned attention weights to produce M combined maps.

        Args:
            u1_maps: (B, n_wavelets, H, W) wavelet modulus maps

        Returns:
            (B, n_heads, H, W) combined maps
        """
        # Softmax attention over wavelets for each head
        attn = torch.softmax(self.alpha, dim=-1)  # (n_heads, n_wavelets)

        # Weighted combination: (n_heads, n_wavelets) × (B, n_wavelets, H, W)
        # → (B, n_heads, H, W)
        # k indexes the wavelet probe dimension
        combined = torch.einsum('mk,bkhw->bmhw', attn, u1_maps)

        return combined

    def _transform_chunk(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """
        Wavelet → attention → pool → standardize for a chunk of samples.

        Args:
            x_chunk: (B_chunk, H, W) input field chunk

        Returns:
            (B_chunk * n_heads, target_size, target_size) downsampled maps
        """
        B_c = x_chunk.shape[0]

        # Wavelet modulus maps
        u1_maps = self._compute_u1_maps(x_chunk)  # (B_c, n_wavelets, H, W)

        # Learned attention mixing
        combined = self._apply_attention(u1_maps)  # (B_c, n_heads, H, W)
        del u1_maps

        # Downsample
        BM_c = B_c * self.n_heads
        maps_flat = combined.reshape(BM_c, 1, self.M, self.N)
        del combined
        maps_down = self.pool(maps_flat).squeeze(1)  # (BM_c, target_size, target_size)
        del maps_flat

        # Per-sample standardize (preserves pixel ordering = topology)
        if self.standardize:
            mean = maps_down.mean(dim=(-2, -1), keepdim=True)
            std = maps_down.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
            maps_down = (maps_down - mean) / std

        return maps_down

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        Full forward pass: wavelets → attention → downsample → persistence.

        Args:
            x: (B, H, W) or (H, W) input field

        Returns:
            List[List[Tensor]]: diagrams[group][sample] = (n_pairs, 2)
            Groups are organized as head0_H0, head0_H1, head1_H0, head1_H1, ...
            giving n_heads * n_hom total groups. Each head's diagrams are kept
            separate for independent PI vectorization downstream.

        Memory handling:
            The wavelet + attention + pool stages produce intermediate tensors
            of size (B, n_wavelets, H, W). For eval with B=20000, H=W=512,
            this is ~80 GB and would OOM. So we chunk the transform stages
            (wavelets → attention → pool → standardize) to keep peak memory
            manageable (~2-3 GB per chunk), then concatenate the downsampled
            maps (B*n_heads, target_size, target_size) which is ~1.3 GB and
            pass them to persistence, which handles its own sub-batching.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        B = x.shape[0]
        chunk_size = self.cnn_eval_batch_size  # 200 samples per chunk

        # Steps 1-4: wavelet → attention → pool → standardize, chunked
        # Each chunk produces (chunk*n_heads, target_size, target_size)
        if B > chunk_size:
            maps_down_chunks = []
            for i in range(0, B, chunk_size):
                x_chunk = x[i:i + chunk_size]
                md = self._transform_chunk(x_chunk)
                maps_down_chunks.append(md.cpu())  # move to CPU to save GPU RAM
                del md
            maps_down = torch.cat(maps_down_chunks, dim=0)  # (B*n_heads, ts, ts)
            maps_down = maps_down.to(x.device)
            del maps_down_chunks
        else:
            maps_down = self._transform_chunk(x)

        # Step 5: Cubical persistence on all BM maps
        # Persistence layer handles its own sub-batching via gpu_sub_batch_size
        diagrams_all = self.cubical(maps_down)  # List[List[Tensor]]
        # diagrams_all[hom_dim] has BM entries
        # BM = B * n_heads, ordered as [sample0_head0, sample0_head1, ..., sample1_head0, ...]

        n_hom = len(self.homology_dimensions)

        if self.separate_heads:
            # Step 6a: Keep heads SEPARATE
            # Output: result[head_idx * n_hom + hom_idx][sample] = (n_pairs, 2)
            result = []
            for m in range(self.n_heads):
                for d_idx in range(n_hom):
                    dim_diagrams = diagrams_all[d_idx]
                    per_sample = []
                    for b in range(B):
                        diag_idx = b * self.n_heads + m
                        diag = dim_diagrams[diag_idx]
                        per_sample.append(diag)
                    result.append(per_sample)
        else:
            # Step 6b: Concatenate diagrams across heads for each hom dim
            # Output: result[hom_idx][sample] = (n_pairs_all_heads, 2)
            # Denser diagrams → fewer dead PI pixels
            result = []
            for d_idx in range(n_hom):
                dim_diagrams = diagrams_all[d_idx]
                per_sample = []
                for b in range(B):
                    head_diagrams = []
                    for m in range(self.n_heads):
                        diag = dim_diagrams[b * self.n_heads + m]
                        if diag.shape[0] > 0:
                            head_diagrams.append(diag)
                    if head_diagrams:
                        per_sample.append(torch.cat(head_diagrams, dim=0))
                    else:
                        per_sample.append(torch.zeros(0, 2, device=x.device))
                result.append(per_sample)

        return result

    def get_attention_summary(self) -> str:
        """Return human-readable attention weight summary for logging."""
        attn = torch.softmax(self.alpha, dim=-1).detach().cpu()
        lines = []
        for m in range(self.n_heads):
            weights = attn[m]
            top_idx = weights.argsort(descending=True)[:3]
            top_str = ", ".join(
                f"(j={self.probe_jl[i][0]},l={self.probe_jl[i][1]}):{weights[i]:.3f}"
                for i in top_idx
            )
            lines.append(f"  Head {m}: {top_str}")
        return "Attention weights:\n" + "\n".join(lines)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"WaveletPersistenceFiltration("
            f"J={self.J}, probes={self.n_wavelets}, "
            f"heads={self.n_heads}, "
            f"params={self.get_num_parameters()}, "
            f"target={self.target_size}×{self.target_size}, "
            f"log_mod={self.log_modulus})"
        )
