"""
Component factory with registry pattern for creating pipeline components.

This module provides a clean factory pattern for creating simulators,
filtrations, vectorizations, and compressions from configuration.
"""
from typing import Dict, Any, Callable, Optional
import torch

from .data_types import (
    SimulatorConfig, FiltrationConfig, VectorizationConfig,
    CompressionConfig, AnalysisConfig
)


# Component registries
SIMULATORS: Dict[str, Callable] = {}
FILTRATIONS: Dict[str, Callable] = {}
VECTORIZATIONS: Dict[str, Callable] = {}
COMPRESSIONS: Dict[str, Callable] = {}


def register_simulator(name: str):
    """Decorator to register a simulator factory."""
    def decorator(func: Callable):
        SIMULATORS[name] = func
        return func
    return decorator


def register_filtration(name: str):
    """Decorator to register a filtration factory."""
    def decorator(func: Callable):
        FILTRATIONS[name] = func
        return func
    return decorator


def register_vectorization(name: str):
    """Decorator to register a vectorization factory."""
    def decorator(func: Callable):
        VECTORIZATIONS[name] = func
        return func
    return decorator


def register_compression(name: str):
    """Decorator to register a compression factory."""
    def decorator(func: Callable):
        COMPRESSIONS[name] = func
        return func
    return decorator


# ============================================================================
# Simulator Factories
# ============================================================================

@register_simulator('grf')
def create_grf_simulator(params: Dict[str, Any]):
    """Create Gaussian Random Field simulator."""
    from ..simulators import GRFSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return GRFSimulator(**params)


@register_simulator('gaussian_vector')
def create_gaussian_vector_simulator(params: Dict[str, Any]):
    """Create Gaussian Vector simulator."""
    from ..simulators import GaussianVectorSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return GaussianVectorSimulator(**params)


@register_simulator('noisy_ring')
def create_noisy_ring_simulator(params: Dict[str, Any]):
    """Create Noisy Ring (Circle) simulator for point clouds."""
    from ..simulators import NoisyRingSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return NoisyRingSimulator(**params)


@register_simulator('grf_fourier')
def create_grf_fourier_simulator(params: Dict[str, Any]):
    """Create Fourier-transformed Gaussian Random Field simulator."""
    from ..simulators import GRFFourierSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return GRFFourierSimulator(**params)


@register_simulator('lensing_lognormal')
def create_lensing_lognormal_simulator(params: Dict[str, Any]):
    """Create Lensing Log-Normal weak lensing simulator.
    
    Requires JAX dependencies: jax, jax_cosmo, and sbi_lens.
    """
    from ..simulators.lensing_lognormal import LensingLogNormalSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return LensingLogNormalSimulator(**params)

@register_simulator('swiss_roll')
def create_swiss_roll_simulator(params: Dict[str, Any]):
    """Create Swiss Roll simulator for point clouds."""
    from ..simulators import SwissRollSimulator

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return SwissRollSimulator(**params)

@register_simulator('swiss_roll_3d')
def create_swiss_roll_3d_simulator(params: Dict[str, Any]):
    """Create Swiss Roll 3D simulator for point clouds."""
    from ..simulators import SwissRollSimulator_3d

    # Auto-detect device if not specified
    if 'device' not in params:
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto-detected device: {params['device']}")

    return SwissRollSimulator_3d(**params)


# ============================================================================
# Filtration Factories
# ============================================================================

@register_filtration('cubical')
def create_cubical_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Cubical filtration.
    
    Supports backends: 'gudhi' (CPU), 'cripser' (CPU), 'gudhi_gpu' (GPU).
    For gudhi_gpu, also accepts 'construction' param ('T' or 'V').
    """
    from ..filtrations import CubicalLayer
    if trainable:
        raise ValueError("Cubical filtration does not support training. Set trainable=false.")
    return CubicalLayer(**params)


@register_filtration('alpha')
def create_alpha_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Alpha filtration."""
    from ..filtrations import AlphaComplexLayer
    if trainable:
        raise ValueError("Alpha filtration does not support training. Set trainable=false.")
    return AlphaComplexLayer(**params)


@register_filtration('identity')
def create_identity_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Identity filtration (pass-through)."""
    from ..filtrations import IdentityFiltration
    if trainable:
        raise ValueError("Identity filtration does not support training. Set trainable=false.")
    return IdentityFiltration(**params)


@register_filtration('learnable')
def create_learnable_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable filtration."""
    from ..filtrations import LearnableFiltration
    if not trainable:
        print("Warning: 'learnable' filtration type but trainable=false. Setting trainable=true.")
    return LearnableFiltration(**params)


@register_filtration('learnable_point')
def create_learnable_point_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Point (Flag Complex) filtration for point clouds."""
    from ..filtrations import LearnablePointFiltration
    if not trainable:
        print("Warning: 'learnable_point' filtration type but trainable=false. Setting trainable=true.")
    return LearnablePointFiltration(**params)

@register_filtration('learnable_edge')
def create_learnable_edge_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Edge (learned metric) filtration for point clouds."""
    from ..filtrations import LearnableEdgeFiltration
    if not trainable:
        print("Warning: 'learnable_edge' filtration type but trainable=false. Setting trainable=true.")
    return LearnableEdgeFiltration(**params)

@register_filtration('learnable_fast_edge')
def create_learnable_fast_edge_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create fast Learnable Edge filtration using Ripser backend."""
    from ..filtrations import LearnableFastEdgeFiltration
    if not trainable:
        print("Warning: 'learnable_fast_edge' filtration type but trainable=false. Setting trainable=true.")
    return LearnableFastEdgeFiltration(**params)

@register_filtration('learnable_dense_point')
def create_learnable_dense_point_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create dense-input learnable filtration (full dist matrix → alpha on embeddings)."""
    from ..filtrations import LearnableDensePointFiltration
    if not trainable:
        print("Warning: 'learnable_dense_point' filtration type but trainable=false. Setting trainable=true.")
    return LearnableDensePointFiltration(**params)

@register_filtration('gnn_point')
def create_gnn_point_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create GNN-based filtration for point clouds."""
    from ..filtrations.gnn_point import LearnableGNNPointFiltration
    if not trainable:
        print("Warning: 'gnn_point' filtration type but trainable=false. Setting trainable=true.")
    return LearnableGNNPointFiltration(**params)

@register_filtration('gnn_readout')
def create_gnn_readout_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create GNN with readout for point clouds (returns vectors, not diagrams)."""
    from ..filtrations.gnn_readout import GNNReadout
    if not trainable:
        print("Warning: 'gnn_readout' filtration type but trainable=false. Setting trainable=true.")
    return GNNReadout(**params)


@register_filtration('alpha_dtm')
def create_alpha_dtm_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Alpha DTM (Distance to Measure) filtration for point clouds.

    This is a non-learnable baseline using the fixed DTM formula:
    vfilt = sqrt(mean(knn_distances^2))
    """
    from ..filtrations import AlphaDTMFiltration
    if trainable:
        raise ValueError("Alpha DTM filtration does not support training. "
                        "Use 'learnable_point' for trainable filtration.")
    return AlphaDTMFiltration(**params)


@register_filtration('learnable_cnn')
def create_learnable_cnn_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable CNN filtration that bypasses TDA.
    
    This filtration applies a CNN to the input field and flattens the output
    to a vector, without computing any topological features (persistence diagrams).
    
    Pipeline: Input (N×N) → CNN → Transformed (N×N) → Flatten → N^2 vector
    """
    from ..filtrations import LearnableCNNFiltration
    if not trainable:
        print("Warning: 'learnable_cnn' filtration type but trainable=false. Setting trainable=true.")
    return LearnableCNNFiltration(**params)


@register_filtration('learnable_mlp')
def create_learnable_mlp_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable MLP filtration that bypasses TDA.
    
    This filtration flattens the input field and passes it through an MLP
    to produce a fixed-size summary vector, without any topological features.
    
    Pipeline: Input (N×N) → Flatten → MLP → output_dim vector
    """
    from ..filtrations import LearnableMLPFiltration
    if not trainable:
        print("Warning: 'learnable_mlp' filtration type but trainable=false. Setting trainable=true.")
    return LearnableMLPFiltration(**params)


@register_filtration('learnable_cnn_mlp')
def create_learnable_cnn_mlp_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable CNN encoder + MLP head filtration that bypasses TDA.
    
    This filtration uses a CNN to encode the input field to a smaller spatial size,
    then an MLP head produces the output vector. More parameter-efficient than
    pure MLP for large inputs (N > 32).
    
    Pipeline: Input (N×N) → CNN Encoder → Flatten → MLP → output_dim vector
    """
    from ..filtrations import LearnableCNNMLPFiltration
    if not trainable:
        print("Warning: 'learnable_cnn_mlp' filtration type but trainable=false. Setting trainable=true.")
    return LearnableCNNMLPFiltration(**params)


@register_filtration('learnable_downsample')
def create_learnable_downsample_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Downsampling filtration: CNN downsample + cubical persistence.

    This filtration uses stride-2 convolutions to downsample high-resolution input
    fields to a smaller target resolution before computing cubical persistence.
    Dramatically speeds up the CPU-bound persistence step (~15× for N=64→16).

    Pipeline: Input (N×N) → Stride-2 CNN → (K×K) → Cubical persistence → Diagrams
    """
    from ..filtrations import LearnableDownsampleFiltration
    if not trainable:
        print("Warning: 'learnable_downsample' filtration type but trainable=false. Setting trainable=true.")
    return LearnableDownsampleFiltration(**params)


@register_filtration('learnable_pointwise')
def create_learnable_pointwise_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Pointwise filtration: per-pixel affine + cubical persistence.

    Each pixel has its own learnable scale and bias (2N² params).
    No CNN, no spatial coupling — persistence gradients flow directly to pixels.
    """
    from ..filtrations.learnable_pointwise import LearnablePointwiseFiltration
    if not trainable:
        print("Warning: 'learnable_pointwise' filtration type but trainable=false. Setting trainable=true.")
    return LearnablePointwiseFiltration(**params)


@register_filtration('learnable_downsample_cnn')
def create_learnable_downsample_cnn_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Downsampling CNN filtration that bypasses TDA.

    Uses the same DownsamplingCNN architecture as learnable_downsample,
    but flattens the output instead of computing persistence diagrams.
    Serves as an ablation baseline to isolate the TDA contribution.

    Pipeline: Input (N×N) → Stride-2 CNN → (K×K) → Flatten → K² vector
    """
    from ..filtrations import LearnableDownsampleCNNFiltration
    if not trainable:
        print("Warning: 'learnable_downsample_cnn' filtration type but trainable=false. Setting trainable=true.")
    return LearnableDownsampleCNNFiltration(**params)


@register_filtration('learnable_dictionary')
def create_learnable_dictionary_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Learnable Dictionary filtration: M learned views of κ → M × persistence.

    Uses M linear combinations of K physics-motivated feature maps (identity,
    gradient, Laplacian, etc.) as independent filtration functions. Each produces
    a different topological "view" of the input field. Total learnable params: M×K.

    Pipeline: Input → AvgPool → K features → M linear heads → M × cubical → diagrams
    """
    from ..filtrations.learnable_dictionary import LearnableDictionaryFiltration
    if not trainable:
        print("Warning: 'learnable_dictionary' filtration type but trainable=false. Setting trainable=true.")
    return LearnableDictionaryFiltration(**params)


@register_filtration('wavelet_persistence')
def create_wavelet_persistence_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Wavelet Persistence filtration: wavelets → learned attention → cubical persistence.

    Physics-informed learnable TDA: Morlet wavelet modulus maps isolate scale-
    specific non-Gaussian structures, learned attention weights select which
    wavelet scales to run persistence on. Extremely lightweight (~28 params).

    Pipeline: Input (H×W) → FFT wavelets → U₁ modulus → attention mix → downsample → cubical
    """
    from ..filtrations.wavelet_persistence import WaveletPersistenceFiltration
    if not trainable:
        print("Warning: 'wavelet_persistence' filtration type but trainable=false. Setting trainable=true.")
    return WaveletPersistenceFiltration(**params)


@register_filtration('cnn_gap')
def create_cnn_gap_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create CNN + Global Average Pooling filtration for 2D fields.

    Deep strided CNN encoder with GAP output. Reduces large 2D inputs
    (e.g., 512×512 lensing maps) to a fixed-size feature vector.
    No BatchNorm — incompatible with Fisher information estimation.

    Pipeline: Input (H×W) → 6 strided conv layers → GAP → 64-dim vector
    """
    from ..filtrations.cnn_gap import CNNGAPFiltration
    return CNNGAPFiltration(**params)


@register_filtration('imnn')
def create_imnn_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create IMNN filtration (Charnock, Lavaux & Wandelt 2018, arXiv:1802.03537).

    A CNN encoder with a fully-connected head that emits n_summaries outputs
    per sample. Trained end-to-end to maximize Fisher information; pair with
    IdentityVectorization and IdentityCompression — the dense head is the
    compression.

    Pipeline: Input (H×W) → strided convs → Flatten → Dense → n_summaries
    """
    from ..filtrations.imnn import IMNNFiltration
    return IMNNFiltration(**params)


@register_filtration('gap_rawtopk')
def create_gap_rawtopk_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create CNN+GAP with precomputed TopK persistence concatenation.

    Combines the slow_stride CNN+GAP encoder (trainable) with precomputed
    TopK cubical persistence features (non-learnable, loaded from sidecar
    files). Requires GAPTopKFiltrationPipeline for data loading.

    Pipeline: Input → [CNN+GAP ∥ precomputed TopK] → concat → vector
    """
    from ..filtrations.gap_rawtopk import GAPRawTopKFiltration
    return GAPRawTopKFiltration(**params)


@register_filtration('cnn_gap_tda')
def create_cnn_gap_tda_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create hybrid CNN+GAP + TDA filtration for 2D fields.

    Two parallel branches: (1) strided CNN → GAP → vector, (2) size-preserving
    CNN → periodic cubical persistence → diagrams. Combined at vectorization
    stage via HybridVectorization.

    Pipeline: Input (H×W) → [CNN+GAP ∥ CNN+Persistence] → [vectors, diagrams]
    """
    from ..filtrations.cnn_gap_tda import CNNGAPTDAFiltration
    if not trainable:
        print("Warning: 'cnn_gap_tda' filtration type but trainable=false. Setting trainable=true.")
    return CNNGAPTDAFiltration(**params)


@register_filtration('frozen_gap_plus_tda')
def create_frozen_gap_plus_tda_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create frozen CNN+GAP + nonlearnable TDA diagnostic filtration.

    Loads a pre-trained CNN+GAP checkpoint (frozen) and runs nonlearnable
    cubical persistence on the raw input. No training required — inference only.

    Pipeline: Input (H×W) → [Frozen CNN+GAP ∥ Cubical persistence] → [vectors, diagrams]
    """
    from ..filtrations.frozen_gap_plus_tda import FrozenGAPPlusTDAFiltration
    if trainable:
        print("Warning: 'frozen_gap_plus_tda' is inference-only. Ignoring trainable=true.")
    return FrozenGAPPlusTDAFiltration(**params)


@register_filtration('cnn_persistence')
def create_cnn_persistence_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create trainable CNN + differentiable cubical persistence filtration.

    Uses a trainable CNN encoder (warm-started from checkpoint) and replaces
    GAP with differentiable cubical persistence on the channel-averaged
    feature map. Gradients flow through persistence back to CNN weights.

    Pipeline: Input → Trainable CNN → channel mean → Diff. persistence → diagrams
    """
    from ..filtrations.cnn_persistence import CNNPersistenceFiltration
    layer = CNNPersistenceFiltration(**params)
    if not trainable:
        print("Note: 'cnn_persistence' loaded with trainable=false; freezing parameters.")
        for p in layer.parameters():
            p.requires_grad = False
    return layer


@register_filtration('cnn_fullres_persistence')
def create_cnn_fullres_persistence_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create full-resolution CNN + 2D periodic cubical persistence filtration.

    A tiny stride-1 CNN transforms the raw input into a single-channel scalar
    field at full resolution.  2D periodic cubical persistence captures the
    topology of this learned filtration function.
    """
    from ..filtrations.cnn_fullres_persistence import CNNFullResPersistenceFiltration
    layer = CNNFullResPersistenceFiltration(**params)
    if not trainable:
        print("Note: 'cnn_fullres_persistence' loaded with trainable=false; freezing parameters.")
        for p in layer.parameters():
            p.requires_grad = False
    return layer


@register_filtration('cnn_fullres_persistence_v2')
def create_cnn_fullres_persistence_v2_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create full-res CNN (skip + identity init) + GPU non-periodic persistence.

    Skip connection ensures output ≈ input at epoch 1, so persistence sees
    real field topology from the start. GPU non-periodic persistence gives
    204× speedup at 512² vs CPU periodic.
    """
    from ..filtrations.cnn_fullres_persistence_v2 import CNNFullResPersistenceV2Filtration
    layer = CNNFullResPersistenceV2Filtration(**params)
    if not trainable:
        print("Note: 'cnn_fullres_persistence_v2' loaded with trainable=false; freezing parameters.")
        for p in layer.parameters():
            p.requires_grad = False
    return layer


@register_filtration('cnn_strided_persistence')
def create_cnn_strided_persistence_filtration(params: Dict[str, Any], trainable: bool = False):
    """Full-res feature extraction + stride-2 cascade to output_size + periodic persistence.

    CNN sees 512×512 at full resolution (fine topology visible to gradient).
    Stride-2 conv layers bring output down to output_size×output_size for
    persistence, making persistence and its backward cheap. Residual skip
    avg_pool(input) gives identity-like init — at epoch 0 output ≈ avg_pool(input).
    output_size=32 → GPU periodic (~3 ms/batch=32);
    output_size=64 → CPU periodic (~160 ms/batch=32).
    """
    from ..filtrations.cnn_strided_persistence import CNNStridedPersistenceFiltration
    layer = CNNStridedPersistenceFiltration(**params)
    if not trainable:
        for p in layer.parameters():
            p.requires_grad = False
    return layer


@register_filtration('cnn_fullres_flat')
def create_cnn_fullres_flat_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create full-res CNN + pool + flatten (NN baseline).

    Same CNN architecture as cnn_fullres_persistence_v2 but pools and flattens
    the output instead of computing persistence. Fair comparison for Phase 6.
    """
    from ..filtrations.cnn_fullres_flat import CNNFullResFlatFiltration
    layer = CNNFullResFlatFiltration(**params)
    if not trainable:
        print("Note: 'cnn_fullres_flat' loaded with trainable=false; freezing parameters.")
        for p in layer.parameters():
            p.requires_grad = False
    return layer


@register_filtration('cnn_fullres_histogram')
def create_cnn_fullres_histogram_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create full-res CNN (skip + identity init) + soft pixel histogram.

    Same skip-connection CNN as cnn_fullres_persistence_v2 but computes a
    differentiable histogram of pixel values instead of persistence.
    Captures the 1-point PDF but not spatial/topological information.
    """
    from ..filtrations.cnn_fullres_histogram import CNNFullResHistogramFiltration
    if not trainable:
        print("Warning: 'cnn_fullres_histogram' is designed for training. Setting trainable=true.")
    return CNNFullResHistogramFiltration(**params)


@register_filtration('shared_backbone_hybrid')
def create_shared_backbone_hybrid_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create shared-backbone hybrid filtration: single CNN with TDA tap.

    A single CNN backbone with an intermediate TDA tap point. Forces the CNN
    to learn features useful for both global statistics (GAP) and topology.

    Pipeline: Input → Early CNN → [tap→Persistence ∥ Late CNN→GAP] → [vectors, diagrams]
    """
    from ..filtrations.shared_backbone_hybrid import SharedBackboneHybridFiltration
    if not trainable:
        print("Warning: 'shared_backbone_hybrid' filtration type but trainable=false. Setting trainable=true.")
    return SharedBackboneHybridFiltration(**params)


@register_filtration('cnn_flat')
def create_cnn_flat_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create single-bin CNN + flatten filtration (NN baseline).

    Same encoder as cnn_persistence but flattens spatial features instead
    of computing persistence.  Fair comparison: same encoder, different
    feature extraction.

    Pipeline: Input (H×W) → CNN encoder → Flatten → N-dim vector
    """
    from ..filtrations.cnn_flat import CNNFlatFiltration
    if not trainable:
        print("Warning: 'cnn_flat' is designed for training. Setting trainable=true.")
    return CNNFlatFiltration(**params)


@register_filtration('cnn_multibin_flat')
def create_cnn_multibin_flat_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create multi-bin CNN + flatten filtration (NN baseline).

    Takes all tomographic bins as input channels, processes through a shared
    CNN backbone with 1×1 projection, and flattens to a feature vector.
    Matched output dimension with CNN + 3D persistence for fair comparison.

    Pipeline: Input (n_bins×H×W) → CNN → 1×1 proj → Flatten → N-dim vector
    """
    from ..filtrations.cnn_multibin_flat import CNNMultiBinFlatFiltration
    if not trainable:
        print("Warning: 'cnn_multibin_flat' is designed for training. Setting trainable=true.")
    return CNNMultiBinFlatFiltration(**params)


@register_filtration('cnn_3d_persistence')
def create_cnn_3d_persistence_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create multi-bin CNN + 3D cubical persistence filtration.

    Same CNN backbone as cnn_multibin_flat, but replaces flatten with 3D
    cubical persistence where the third axis is the bin/redshift dimension.
    Captures cross-bin topological structure.

    Pipeline: Input (n_bins×H×W) → CNN → 1×1 proj → 3D cubical persistence → diagrams
    """
    from ..filtrations.cnn_3d_persistence import CNN3DPersistenceFiltration
    if not trainable:
        print("Warning: 'cnn_3d_persistence' is designed for training. Setting trainable=true.")
    return CNN3DPersistenceFiltration(**params)


@register_filtration('scattering')
def create_scattering_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Wavelet Scattering Transform filtration for 2D fields.

    Non-learnable feature extractor based on cascaded wavelet convolutions
    + modulus. Captures non-Gaussian information at multiple scales.
    Deterministic — no training required.

    Pipeline: Input (H×W) → Morlet wavelets → S0 + S1 + S2 coefficients
    """
    from ..filtrations.scattering import ScatteringFiltration
    if trainable:
        raise ValueError("Scattering filtration is non-learnable. Set trainable=false.")
    return ScatteringFiltration(**params)


@register_filtration('power_spectrum')
def create_power_spectrum_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Power Spectrum filtration for 2D fields.

    Non-learnable: computes radially-averaged 2D power spectrum.
    For convergence maps, captures all Gaussian 2-point information.

    Pipeline: Input (H×W) → FFT → |FFT|² → radial binning → n_ell_bins vector
    """
    from ..filtrations.power_spectrum import PowerSpectrumFiltration
    if trainable:
        raise ValueError("Power spectrum filtration is non-learnable. Set trainable=false.")
    return PowerSpectrumFiltration(**params)


@register_filtration('peak_counts')
def create_peak_counts_filtration(params: Dict[str, Any], trainable: bool = False):
    """Create Peak Counts filtration for 2D convergence maps.

    Non-learnable: counts local maxima (pixels strictly greater than all
    8 neighbours) and bins peak heights into a fixed histogram.
    Captures non-Gaussian information from the halo mass function.

    Pipeline: Input (H×W) → local-max detection → histogram → n_bins vector
    """
    from ..filtrations.peak_counts import PeakCountsFiltration
    if trainable:
        raise ValueError("Peak counts filtration is non-learnable. Set trainable=false.")
    return PeakCountsFiltration(**params)


# ============================================================================
# Vectorization Factories
# ============================================================================

@register_vectorization('topk')
def create_topk_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create TopK vectorization."""
    from ..vectorizations import TopKLayer
    if trainable:
        raise ValueError("TopK vectorization does not support training. Set trainable=false.")
    return TopKLayer(**params)


@register_vectorization('persistence_image')
def create_persistence_image_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Persistence Image vectorization."""
    from ..vectorizations import PersistenceImageLayer
    if trainable:
        # Use differentiable version for learnable pipelines (pure PyTorch, preserves gradients)
        from ..vectorizations import DifferentiablePersistenceImageLayer
        return DifferentiablePersistenceImageLayer(**params)
    return PersistenceImageLayer(**params)


@register_vectorization('combined')
def create_combined_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Combined vectorization."""
    from ..vectorizations import CombinedVectorization

    if trainable:
        raise ValueError("Combined vectorization itself doesn't support training. "
                       "Set trainable=true on individual sub-vectorizations.")

    # Extract sub-vectorization configs (support both 'layers' and 'configs')
    if 'layers' in params:
        sub_configs = params['layers']
    elif 'configs' in params:
        sub_configs = params['configs']
    else:
        raise ValueError("Combined vectorization requires 'layers' (or 'configs') parameter")

    if not isinstance(sub_configs, list) or len(sub_configs) == 0:
        raise ValueError("Combined vectorization 'layers' must be a non-empty list")

    # Create sub-vectorizations
    vectorizations = []
    for config in sub_configs:
        vec_type = config['type']
        vec_params = config.get('params', {})
        vec_trainable = config.get('trainable', False)

        if vec_type not in VECTORIZATIONS:
            raise ValueError(f"Unknown vectorization type: {vec_type}")

        vec = VECTORIZATIONS[vec_type](vec_params, vec_trainable)
        vectorizations.append(vec)

    return CombinedVectorization(vectorizations)


@register_vectorization('persistence_histogram')
def create_persistence_histogram_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Persistence Histogram vectorization (Yip et al. 2024)."""
    from ..vectorizations import PersistenceHistogramLayer
    if trainable:
        raise ValueError("Persistence histogram vectorization does not support training. "
                        "Set trainable=false (the MLP compression handles learning).")
    return PersistenceHistogramLayer(**params)


@register_vectorization('persistence_curves')
def create_persistence_curves_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Persistence Curves (EDF) vectorization (Biagetti, Cole & Shiu 2021)."""
    from ..vectorizations import PersistenceCurveLayer
    if trainable:
        raise ValueError("Persistence curves vectorization does not support training. "
                        "Set trainable=false (the MLP compression handles learning).")
    return PersistenceCurveLayer(**params)


@register_vectorization('differentiable_persistence_curves')
def create_diff_persistence_curves_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Differentiable Persistence Curves (soft-sigmoid EDF) for learnable pipelines."""
    from ..vectorizations.differentiable_persistence_curves import DifferentiablePersistenceCurveLayer
    return DifferentiablePersistenceCurveLayer(**params)


@register_vectorization('atol')
def create_atol_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create ATOL (Royer et al. 2021) — set-based vectorization with K Gaussian centres.

    When trainable=True, both the centres and the bandwidth become learnable
    (mirrors the trainable-DiffPI / DiffCurves convention).
    """
    from ..vectorizations import ATOLLayer
    if trainable:
        params = {**params, 'learn_centers': True, 'learn_bandwidth': True}
    return ATOLLayer(**params)


@register_vectorization('persistence_landscape')
def create_landscape_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Differentiable Persistence Landscape (Bubenik 2015).

    When trainable=True, the triangle slope is multiplied by a learnable
    scalar (`learn_slope=True`).
    """
    from ..vectorizations import DifferentiablePersistenceLandscapeLayer
    if trainable:
        params = {**params, 'learn_slope': True}
    return DifferentiablePersistenceLandscapeLayer(**params)


@register_vectorization('persistence_silhouette')
def create_silhouette_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Persistence Silhouette (Chazal et al. 2014).

    Mean-pools tent functions over all diagram points (uniform weighting by
    default).  When trainable=True, the tent slope is multiplied by a learnable
    scalar (`learn_slope=True`).
    """
    from ..vectorizations import PersistenceSilhouetteLayer
    if trainable:
        params = {**params, 'learn_slope': True}
    return PersistenceSilhouetteLayer(**params)


@register_vectorization('perslay')
def create_perslay_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create PersLay (Carrière et al., AISTATS 2020) — fully learnable PD vectorizer.

    Permutation-invariant neural network over diagram points: each point is
    embedded by a small MLP φ, weighted by a diagonal-vanishing scalar w(p), and
    sum-pooled into a fixed-size vector. Every existing vectorization (PI,
    DiffCurves, Silhouette, Landscape, ATOL) is a special case.

    When trainable=False, all parameters (φ, w, optional ρ) are frozen at
    initialisation — a smoke-test baseline using random nonlinear features.
    When trainable=True (the scientifically meaningful setting), all parameters
    are learnable end-to-end via backprop from MOPED / MLP / Fisher loss.
    """
    from ..vectorizations import PersLayLayer
    if not trainable:
        params = {**params, 'learn_features': False}
    return PersLayLayer(**params)


@register_vectorization('identity')
def create_identity_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Identity vectorization (pass-through)."""
    from ..vectorizations import IdentityVectorization
    if trainable:
        raise ValueError("Identity vectorization does not support training. Set trainable=false.")
    return IdentityVectorization(**params)


@register_vectorization('hybrid')
def create_hybrid_vectorization(params: Dict[str, Any], trainable: bool = False):
    """Create Hybrid vectorization for mixed vector + diagram outputs.

    Handles output from cnn_gap_tda filtration: slot 0 = GAP vectors (stacked
    directly), slots 1+ = persistence diagrams (vectorized by diagram_layers).
    All features concatenated into a single vector per sample.
    """
    from ..vectorizations.hybrid import HybridVectorization

    sub_configs = params.get('diagram_layers', [])
    if not isinstance(sub_configs, list) or len(sub_configs) == 0:
        raise ValueError("Hybrid vectorization requires 'diagram_layers' parameter (non-empty list)")

    diagram_layers = []
    for config in sub_configs:
        vec_type = config['type']
        vec_params = config.get('params', {})
        vec_trainable = config.get('trainable', False)

        if vec_type not in VECTORIZATIONS:
            raise ValueError(f"Unknown vectorization type: {vec_type}")

        vec = VECTORIZATIONS[vec_type](vec_params, vec_trainable)
        diagram_layers.append(vec)

    return HybridVectorization(diagram_layers)


# ============================================================================
# Compression Factories
# ============================================================================

@register_compression('identity')
def create_identity_compression(params: Dict[str, Any], trainable: bool = False):
    """Create Identity compression (pass-through)."""
    from ..compressions import IdentityCompression
    if trainable:
        raise ValueError("Identity compression does not support training. Set trainable=false.")
    return IdentityCompression(**params)


@register_compression('moped')
def create_moped_compression(params: Dict[str, Any], trainable: bool = False):
    """Create MOPED compression."""
    from ..compressions import MOPEDCompression
    if trainable:
        raise ValueError("MOPED compression does not require training. Set trainable=false.")

    # Ensure reg is a float if provided
    if 'reg' in params:
        params['reg'] = float(params['reg'])

    return MOPEDCompression(**params)


@register_compression('mlp')
def create_mlp_compression(params: Dict[str, Any], trainable: bool = False):
    """Create MLP compression."""
    from ..compressions import MLPCompression
    if not trainable:
        print("Warning: MLP compression typically requires training. Consider setting trainable=true.")
    return MLPCompression(**params)


@register_compression('cnn')
def create_cnn_compression(params: Dict[str, Any], trainable: bool = False):
    """Create CNN compression."""
    from ..compressions import CNNCompression
    if not trainable:
        print("Warning: CNN compression typically requires training. Consider setting trainable=true.")
    return CNNCompression(**params)


@register_compression('inception')
def create_inception_compression(params: Dict[str, Any], trainable: bool = False):
    """Create Inception block compression."""
    from ..compressions import InceptionCompression
    if not trainable:
        print("Warning: Inception compression typically requires training. Consider setting trainable=true.")
    return InceptionCompression(**params)


# ============================================================================
# Main Factory Functions
# ============================================================================

def create_simulator(config: SimulatorConfig):
    """
    Create a simulator from configuration.

    Args:
        config: Simulator configuration

    Returns:
        Simulator instance

    Raises:
        ValueError: If simulator type is unknown
    """
    if config.type not in SIMULATORS:
        available = ', '.join(SIMULATORS.keys())
        raise ValueError(f"Unknown simulator type: {config.type}. Available: {available}")

    return SIMULATORS[config.type](config.params)


def create_filtration(config: FiltrationConfig):
    """
    Create a filtration from configuration.

    Args:
        config: Filtration configuration

    Returns:
        Filtration instance

    Raises:
        ValueError: If filtration type is unknown
    """
    if config.type not in FILTRATIONS:
        available = ', '.join(FILTRATIONS.keys())
        raise ValueError(f"Unknown filtration type: {config.type}. Available: {available}")

    return FILTRATIONS[config.type](config.params, config.trainable)


def create_vectorization(config: VectorizationConfig):
    """
    Create a vectorization from configuration.

    Args:
        config: Vectorization configuration

    Returns:
        Vectorization instance

    Raises:
        ValueError: If vectorization type is unknown
    """
    if config.type not in VECTORIZATIONS:
        available = ', '.join(VECTORIZATIONS.keys())
        raise ValueError(f"Unknown vectorization type: {config.type}. Available: {available}")

    return VECTORIZATIONS[config.type](config.params, config.trainable)


def create_compression(config: CompressionConfig):
    """
    Create a compression from configuration.

    Args:
        config: Compression configuration

    Returns:
        Compression instance

    Raises:
        ValueError: If compression type is unknown
    """
    if config.type not in COMPRESSIONS:
        available = ', '.join(COMPRESSIONS.keys())
        raise ValueError(f"Unknown compression type: {config.type}. Available: {available}")

    return COMPRESSIONS[config.type](config.params, config.trainable)


def create_fisher_analyzer(clean_data: bool = True):
    """
    Create a Fisher analyzer.

    Args:
        clean_data: Whether to clean data before analysis

    Returns:
        FisherAnalyzer instance
    """
    from ..fisher import FisherAnalyzer
    return FisherAnalyzer(clean_data=clean_data)

