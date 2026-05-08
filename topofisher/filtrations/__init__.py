"""Filtration methods for TopoFisher pipeline."""

from .cubical import CubicalLayer
from .alpha import AlphaComplexLayer
from .identity import IdentityFiltration
from .learnable import LearnableFiltration
from .learnable_cnn import LearnableCNNFiltration
from .learnable_mlp import LearnableMLPFiltration
from .learnable_cnn_mlp import LearnableCNNMLPFiltration
from .learnable_point import LearnablePointFiltration, VertexFiltrationMLP
from .learnable_edge import LearnableEdgeFiltration, EdgeEmbeddingMLP
from .learnable_fast_edge import LearnableFastEdgeFiltration, FastEdgeEmbeddingMLP
from .learnable_dense_point import LearnableDensePointFiltration, DenseDistanceMLP
from .learnable_downsample import LearnableDownsampleFiltration
from .learnable_downsample_cnn import LearnableDownsampleCNNFiltration
from .learnable_dictionary import LearnableDictionaryFiltration
from .wavelet_persistence import WaveletPersistenceFiltration
from .alpha_dtm import AlphaDTMFiltration
from .cnn_gap import CNNGAPFiltration
from .imnn import IMNNFiltration
from .cnn_gap_tda import CNNGAPTDAFiltration
from .frozen_gap_plus_tda import FrozenGAPPlusTDAFiltration
from .gap_rawtopk import GAPRawTopKFiltration
from .cnn_persistence import CNNPersistenceFiltration
from .cnn_fullres_persistence import CNNFullResPersistenceFiltration
from .cnn_fullres_persistence_v2 import CNNFullResPersistenceV2Filtration
from .cnn_strided_persistence import CNNStridedPersistenceFiltration
from .cnn_fullres_flat import CNNFullResFlatFiltration
from .cnn_fullres_histogram import CNNFullResHistogramFiltration
from .cnn_flat import CNNFlatFiltration
from .cnn_multibin_flat import CNNMultiBinFlatFiltration
from .cnn_3d_persistence import CNN3DPersistenceFiltration
from .shared_backbone_hybrid import SharedBackboneHybridFiltration
from .scattering import ScatteringFiltration
from .power_spectrum import PowerSpectrumFiltration
from .peak_counts import PeakCountsFiltration
from .differentiable_cubical import DifferentiableCubicalLayer
from .gpu_cubical import GPUCubicalLayer, is_cmp_available

__all__ = [
    'CubicalLayer',
    'AlphaComplexLayer',
    'IdentityFiltration',
    'LearnableFiltration',
    'LearnableCNNFiltration',
    'LearnableMLPFiltration',
    'LearnableCNNMLPFiltration',
    'LearnablePointFiltration',
    'VertexFiltrationMLP',
    'LearnableEdgeFiltration',
    'EdgeEmbeddingMLP',
    'LearnableFastEdgeFiltration',
    'FastEdgeEmbeddingMLP',
    'LearnableDensePointFiltration',
    'DenseDistanceMLP',
    'LearnableDownsampleFiltration',
    'LearnableDownsampleCNNFiltration',
    'LearnableDictionaryFiltration',
    'WaveletPersistenceFiltration',
    'AlphaDTMFiltration',
    'CNNGAPFiltration',
    'IMNNFiltration',
    'CNNGAPTDAFiltration',
    'FrozenGAPPlusTDAFiltration',
    'GAPRawTopKFiltration',
    'SharedBackboneHybridFiltration',
    'ScatteringFiltration',
    'PowerSpectrumFiltration',
    'PeakCountsFiltration',
    'DifferentiableCubicalLayer',
    'GPUCubicalLayer',
    'is_cmp_available',
    'CNNFullResPersistenceFiltration',
    'CNNFullResPersistenceV2Filtration',
    'CNNFullResFlatFiltration',
    'CNNFullResHistogramFiltration',
    'CNNFlatFiltration',
    'CNNMultiBinFlatFiltration',
    'CNN3DPersistenceFiltration',
]