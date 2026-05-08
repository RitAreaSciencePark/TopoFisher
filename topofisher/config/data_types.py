"""
Configuration data types for YAML pipeline loading.

This module defines dataclasses for all configuration sections
used in YAML pipeline definitions.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import torch


@dataclass
class CacheConfig:
    """Configuration for caching diagrams and summaries."""
    mode: str                                # "generate" or "load"
    data_type: str                           # "diagrams" or "summaries"
    save_path: Optional[str] = None          # Path to save (for generate mode)
    load_path: Optional[str] = None          # Path to load from (for load mode)


@dataclass
class ExperimentConfig:
    """Experiment metadata configuration."""
    name: str
    output_dir: str = "experiments"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AnalysisConfig:
    """
    Analysis parameters configuration.

    Accepts both lists (from YAML) and tensors (programmatic use).
    Lists are auto-converted to tensors in __post_init__.
    """
    theta_fid: torch.Tensor      # Fiducial parameter values
    delta_theta: torch.Tensor    # Step sizes for finite differences (±Δθ/2)
    n_s: int                     # Number of samples for covariance estimation
    n_d: int                     # Number of samples for derivative estimation
    seed_cov: int = 42           # Seed for fiducial samples
    seed_ders: List[int] = field(default_factory=list)  # Seeds for derivative samples
    cache: Optional[CacheConfig] = None  # Cache configuration for cached pipelines
    data_dir: Optional[str] = None  # Directory for pre-generated simulation data (scratch)
    bin_idx: Optional[int] = None  # If set, load only this bin from per-bin files
    summaries_dir: Optional[str] = None  # If set, cache/load vectorized summaries here

    def __post_init__(self):
        """Convert lists to tensors and validate."""
        # Auto-convert lists to tensors (for YAML compatibility)
        if not isinstance(self.theta_fid, torch.Tensor):
            self.theta_fid = torch.tensor(self.theta_fid)
        if not isinstance(self.delta_theta, torch.Tensor):
            self.delta_theta = torch.tensor(self.delta_theta)

        # Auto-generate derivative seeds if not provided
        if not self.seed_ders:
            n_params = len(self.theta_fid)
            self.seed_ders = [self.seed_cov + i + 1 for i in range(n_params)]

        # Validate dimensions match
        if len(self.delta_theta) != len(self.theta_fid):
            raise ValueError(
                f"delta_theta length ({len(self.delta_theta)}) must match "
                f"theta_fid length ({len(self.theta_fid)})"
            )
        if len(self.seed_ders) != len(self.theta_fid):
            raise ValueError(
                f"seed_ders length ({len(self.seed_ders)}) must match "
                f"theta_fid length ({len(self.theta_fid)})"
            )


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    type: str  # e.g., 'grf', 'gaussian_vector'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FiltrationConfig:
    """Filtration configuration."""
    type: str  # e.g., 'cubical', 'alpha', 'learnable', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizationConfig:
    """Vectorization configuration."""
    type: str  # e.g., 'topk', 'persistence_image', 'combined', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionConfig:
    """Compression configuration."""
    type: str  # e.g., 'moped', 'mlp', 'cnn', 'identity'
    trainable: bool = False  # Whether this component requires training
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Training configuration for learnable components."""
    n_epochs: int = 1000
    lr: float = 1e-3
    batch_size: int = 500
    weight_decay: float = 1e-4
    train_frac: float = 0.5
    val_frac: float = 0.25
    validate_every: int = 10
    verbose: bool = True
    patience: Optional[int] = None  # Early stopping patience
    min_delta: float = 1e-6  # Minimum improvement for early stopping
    lambda_k: float = 0.0  # Kurtosis regularization strength (0 = disabled)
    lambda_s: float = 0.0  # Skewness regularization strength (0 = disabled)
    # Stability improvements
    grad_clip: Optional[float] = 1.0  # Gradient clipping max norm (None = disabled)
    lr_scheduler: str = "plateau"  # "plateau", "cosine", "cosine_restarts", or "none"
    lr_patience: int = 20  # Patience for ReduceLROnPlateau
    lr_factor: float = 0.5  # Factor for ReduceLROnPlateau
    lr_min: float = 1e-6  # Minimum learning rate
    lr_T0: int = 500  # Period of first restart for cosine_restarts (in epochs)
    lr_T_mult: int = 1  # Period multiplier after each restart (1 = fixed period)
    lr_warmup: int = 0  # Linear warmup epochs (0 = disabled)
    seed: int = 42  # Random seed for model initialization and training
    # NaN recovery settings (for topology-heavy pipelines like DiffCurves)
    nan_recovery_patience: int = 3  # Consecutive NaN epochs before reloading best model (was 10)
    nan_recovery_noise: float = 1e-3  # Std of Gaussian noise added to parameters after NaN recovery
    nan_abort_threshold: int = 100  # Abort training after this many consecutive NaN epochs
    loss_clamp: Optional[float] = 50.0  # Clamp loss magnitude to prevent slogdet spikes (None = disabled)
    # MOPED periodic refit: recompute compression matrix on training data every N epochs.
    # (0 = disabled). moped_refit_n_samples > 0 limits the subset used (cheaper for raw maps).
    moped_refit_every: int = 0
    moped_refit_n_samples: int = 0
    # PersLay normalization refit: update input_mean/input_std from fiducial training data
    # every N epochs (0 = disabled). renormalize_n_samples > 0 limits the subset used.
    renormalize_every: int = 0
    renormalize_n_samples: int = 0
    # Two-phase curriculum: freeze filtration (CNN) for the first N epochs so
    # PersLay + MOPED can converge on the near-identity filtration, then unfreeze
    # for joint optimization. (0 = disabled, train everything from epoch 0.)
    freeze_cnn_epochs: int = 0
    # Lambda annealing: linearly decay lambda_s and lambda_k from their initial
    # values to 0 over the first lambda_anneal_epochs epochs.  Useful for
    # noreg variants where lambda=0 is the target but bins with weak Fisher
    # signal (e.g. bin0) cannot escape the singular-Fisher basin at random
    # PersLay initialization without a short stabilization period.
    # (0 = disabled, lambdas stay constant at their configured values.)
    lambda_anneal_epochs: int = 0


@dataclass
class PipelineYAMLConfig:
    """
    Complete pipeline configuration from YAML.

    Simple data container that combines all configuration sections.
    Simulator and filtration are optional for load mode (diagrams already cached).
    """
    experiment: ExperimentConfig
    analysis: AnalysisConfig
    vectorization: VectorizationConfig
    compression: CompressionConfig
    simulator: Optional[SimulatorConfig] = None      # Optional for load mode
    filtration: Optional[FiltrationConfig] = None    # Optional for load mode
    training: Optional[TrainingConfig] = None

    def is_trainable(self) -> bool:
        """
        Check if any component requires training.

        Returns:
            True if any component has trainable=True
        """
        filtration_trainable = self.filtration.trainable if self.filtration else False
        return (filtration_trainable or
                self.vectorization.trainable or
                self.compression.trainable)

    def get_trainable_component(self) -> Optional[str]:
        """
        Get which component is trainable.

        Returns:
            'filtration', 'vectorization', 'compression', or None
        """
        if self.filtration and self.filtration.trainable:
            return 'filtration'
        elif self.vectorization.trainable:
            return 'vectorization'
        elif self.compression.trainable:
            return 'compression'
        return None

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If configuration is invalid
        """
        # Require training config for trainable pipelines
        if self.is_trainable() and self.training is None:
            component = self.get_trainable_component()
            raise ValueError(
                f"Training configuration required for trainable {component} component. "
                "Please add a 'training:' section to your YAML config."
            )


# =============================================================================
# Pipeline Output
# =============================================================================

@dataclass
class FisherResult:
    """Results from Fisher information analysis."""
    fisher_matrix: torch.Tensor      # Fisher information matrix (n_params, n_params)
    inverse_fisher: torch.Tensor     # Covariance matrix = F^-1
    derivatives: torch.Tensor        # Parameter derivatives (n_params, n_d, n_features)
    covariance: torch.Tensor         # Summary covariance matrix (n_features, n_features)
    log_det_fisher: torch.Tensor     # log|F| (scalar)
    constraints: torch.Tensor        # 1-sigma constraints = sqrt(diag(F^-1))

    # Optional diagnostic information
    bias_error: Optional[torch.Tensor] = None
    fractional_bias: Optional[torch.Tensor] = None
    is_gaussian: Optional[bool] = None
    gaussianity_details: Optional[Dict[str, Any]] = None

    # Gaussianity regularization penalties (computed when compute_moments=True)
    skewness_penalty: Optional[torch.Tensor] = None   # Mean squared skewness
    kurtosis_penalty: Optional[torch.Tensor] = None   # Mean squared excess kurtosis

    def print_gaussianity(self, print_details: bool = False):
        """Print Gaussianity check result.

        Args:
            print_details: If True, print detailed test statistics.
        """
        if self.is_gaussian is None:
            print("\nGaussianity Check: Not performed")
        else:
            gauss_mark = "✓ PASS" if self.is_gaussian else "✗ FAIL"
            print(f"\nGaussianity Check: {gauss_mark}")

            if print_details:
                d = self.gaussianity_details
                print(f"  Test: {d.get('test', 'unknown')} (α={d.get('alpha', 0.05)})")
                print(f"  Datasets: {d.get('n_datasets_all_gaussian', '?')}/{d.get('n_datasets', '?')} all Gaussian")
                print(f"  Features: {d.get('total_passed', '?')}/{d.get('total_tests', '?')} tests passed")
                # Print per-feature max KS statistic across datasets
                if 'per_dataset' in d:
                    print("  Per-feature max KS statistic (D):")
                    # Collect all feature keys from first dataset
                    first_ds = next(iter(d['per_dataset'].values()))
                    for fkey in sorted(first_ds['features'].keys(), key=lambda k: int(k.split('_')[1])):
                        ds_stats = []
                        for ds_name, ds_data in d['per_dataset'].items():
                            feat = ds_data['features'].get(fkey, {})
                            ds_stats.append(feat.get('ks_statistic', float('nan')))
                        max_d = max(ds_stats)
                        min_p = min(feat.get('ks_pvalue', 1.0) for ds_data in d['per_dataset'].values()
                                    for feat in [ds_data['features'].get(fkey, {})])
                        print(f"    {fkey}: max D={max_d:.4f}, min p={min_p:.4f}")