"""
Configuration module for YAML pipeline loading.

This module provides a clean way to load and create pipelines from YAML configurations.
"""

# Data types
from .data_types import (
    CacheConfig,
    ExperimentConfig,
    AnalysisConfig,
    SimulatorConfig,
    FiltrationConfig,
    VectorizationConfig,
    CompressionConfig,
    TrainingConfig,
    PipelineYAMLConfig,
    FisherResult,
)

# Component factory
from .component_factory import (
    create_simulator,
    create_filtration,
    create_vectorization,
    create_compression,
    create_fisher_analyzer,
    # Registries for extending
    register_simulator,
    register_filtration,
    register_vectorization,
    register_compression
)

# Loader utilities
from .loader import (
    load_pipeline_config,
    create_pipeline_from_config,
    load_and_create_pipeline,
    load_pipeline_checkpoint
)

__all__ = [
    # Data types
    'CacheConfig',
    'ExperimentConfig',
    'AnalysisConfig',
    'SimulatorConfig',
    'FiltrationConfig',
    'VectorizationConfig',
    'CompressionConfig',
    'TrainingConfig',
    'PipelineYAMLConfig',
    'FisherResult',
    # Factory functions
    'create_simulator',
    'create_filtration',
    'create_vectorization',
    'create_compression',
    'create_fisher_analyzer',
    # Registry decorators
    'register_simulator',
    'register_filtration',
    'register_vectorization',
    'register_compression',
    # Loader functions
    'load_pipeline_config',
    'create_pipeline_from_config',
    'load_and_create_pipeline',
    'load_pipeline_checkpoint',
]