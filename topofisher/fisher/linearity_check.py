"""
Linearity check for Fisher information numerical derivatives.

This module provides tools to verify that the summary statistic response
to parameter changes is linear, which is essential for accurate Fisher
information estimation via numerical derivatives.

The check verifies:
- D(θ+dθ) + D(θ-dθ) - 2*D(θ) ≈ 0 (second derivative test)
- The ratio of responses for different step sizes is consistent
- Fisher convergence: F(δ) ≈ F(δ/2) (the Fisher matrix is stable under step size changes)
"""
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field


@dataclass
class LinearityResult:
    """Results from linearity check."""
    # Parameter step sizes used
    delta_theta1: torch.Tensor
    delta_theta2: torch.Tensor
    
    # Mean summary vectors at each parameter value
    mean_fiducial: torch.Tensor
    mean_plus1: torch.Tensor
    mean_minus1: torch.Tensor
    mean_plus2: torch.Tensor
    mean_minus2: torch.Tensor
    
    # Symmetric deviation: D(θ+dθ) + D(θ-dθ) - 2*D(θ)
    symmetric_deviation1: torch.Tensor  # For dθ1
    symmetric_deviation2: torch.Tensor  # For dθ2
    
    # Relative symmetric deviation (normalized by std of fiducial)
    relative_deviation1: torch.Tensor
    relative_deviation2: torch.Tensor
    
    # Numerical derivatives using centered differences
    derivative1: torch.Tensor  # [D(θ+dθ1) - D(θ-dθ1)] / dθ1
    derivative2: torch.Tensor  # [D(θ+dθ2) - D(θ-dθ2)] / dθ2
    
    # Derivative consistency: ratio should be ~1 for linear response
    derivative_ratio: torch.Tensor
    
    # Summary statistics
    linearity_score1: float  # Mean absolute relative deviation for dθ1
    linearity_score2: float  # Mean absolute relative deviation for dθ2
    derivative_consistency: float  # How close derivative_ratio is to 1
    
    # Pass/fail assessment
    is_linear: bool
    tolerance_used: float


@dataclass
class FisherConvergenceResult:
    """Results from Fisher matrix convergence test.
    
    Compares F(δ₁) vs F(δ₂) where δ₂ = δ₁/2 to verify that the
    Fisher information estimate is stable under step size changes.
    This is the gold-standard diagnostic for numerical derivative validity.
    """
    # Fisher matrices at two step sizes (may be None if covariance is singular)
    fisher_matrix1: Optional[torch.Tensor]
    fisher_matrix2: Optional[torch.Tensor]
    
    # Log determinants
    log_det_fisher1: float
    log_det_fisher2: float
    
    # Fisher-level convergence metrics
    log_det_relative_change: float  # |logdet F1 - logdet F2| / |logdet F1|
    frobenius_relative_change: float  # ||F1 - F2||_F / ||F1||_F
    element_wise_max_change: float  # max |F1_ij - F2_ij| / max |F1_ij|
    
    # Derivative-level convergence (robust, doesn't need covariance)
    # ||μ'(δ₁) - μ'(δ₂)|| / ||μ'(δ₁)|| averaged over parameters
    derivative_vector_convergence: float
    per_param_derivative_convergence: Dict[int, float]
    
    # Step sizes used
    delta_theta1: torch.Tensor
    delta_theta2: torch.Tensor
    
    # Pass/fail
    is_converged: bool
    tolerance_used: float
    n_samples: int  # Number of samples used for covariance
    n_features: int
    covariance_well_conditioned: bool  # Whether n_samples >> n_features


def compute_fisher_convergence(
    summaries_fid: torch.Tensor,
    per_param_derivatives1: Dict[int, torch.Tensor],
    per_param_derivatives2: Dict[int, torch.Tensor],
    delta_theta1: torch.Tensor,
    delta_theta2: torch.Tensor,
    n_samples: int,
    tolerance: float = 0.1
) -> FisherConvergenceResult:
    """Compute Fisher matrices at two step sizes and compare.
    
    Args:
        summaries_fid: Fiducial summary samples (n_samples, n_features) for covariance
        per_param_derivatives1: {param_idx: mean_derivative1} at step δ₁
        per_param_derivatives2: {param_idx: mean_derivative2} at step δ₂
        delta_theta1: First step sizes
        delta_theta2: Second step sizes  
        n_samples: Number of samples used (for Hartlap correction)
        tolerance: Tolerance for convergence assessment
    
    Returns:
        FisherConvergenceResult
    """
    n_params = len(per_param_derivatives1)
    param_indices = sorted(per_param_derivatives1.keys())
    n_features = summaries_fid.shape[1]
    n_s = summaries_fid.shape[0]
    
    # Stack mean derivatives into matrices: (n_params, n_features)
    mu_prime1 = torch.stack([per_param_derivatives1[i] for i in param_indices])
    mu_prime2 = torch.stack([per_param_derivatives2[i] for i in param_indices])
    
    # === DERIVATIVE VECTOR CONVERGENCE (always robust) ===
    # ||μ'(δ₁) - μ'(δ₂)|| / ||μ'(δ₁)|| per parameter
    per_param_dc = {}
    for i, pidx in enumerate(param_indices):
        norm1 = torch.norm(mu_prime1[i]).item()
        if norm1 > 1e-10:
            per_param_dc[pidx] = torch.norm(mu_prime1[i] - mu_prime2[i]).item() / norm1
        else:
            per_param_dc[pidx] = torch.norm(mu_prime1[i] - mu_prime2[i]).item()
    deriv_vec_conv = np.mean(list(per_param_dc.values()))
    
    # === FISHER MATRIX CONVERGENCE (only if covariance is well-conditioned) ===
    covariance_ok = n_s > n_features + 10  # Need n_samples >> n_features
    
    F1 = None
    F2 = None
    ldf1 = float('nan')
    ldf2 = float('nan')
    log_det_rel = float('nan')
    frob_rel = float('nan')
    elem_max_rel = float('nan')
    
    if covariance_ok:
        # Compute covariance with Hartlap correction
        C = torch.cov(summaries_fid.T)  # (n_features, n_features)
        hartlap = (n_s - 1.0) / (n_s - n_features - 2.0)
        if hartlap > 0:
            C = C / hartlap
        
        # Regularize if needed
        try:
            inv_C = torch.linalg.inv(C)
        except:
            eps = 1e-6 * torch.trace(C).abs() / n_features
            C_reg = C + max(eps.item(), 1e-10) * torch.eye(n_features, dtype=C.dtype)
            inv_C = torch.linalg.inv(C_reg)
        
        # Fisher matrices: F_ij = dμ_i^T C^{-1} dμ_j
        F1 = mu_prime1 @ inv_C @ mu_prime1.T
        F2 = mu_prime2 @ inv_C @ mu_prime2.T
        
        # Log determinants
        sign1, logdet1 = torch.linalg.slogdet(F1)
        sign2, logdet2 = torch.linalg.slogdet(F2)
        ldf1 = (sign1 * logdet1).item() if sign1.item() > 0 else float('-inf')
        ldf2 = (sign2 * logdet2).item() if sign2.item() > 0 else float('-inf')
        
        # Convergence metrics
        if np.isfinite(ldf1) and np.isfinite(ldf2) and abs(ldf1) > 1e-10:
            log_det_rel = abs(ldf1 - ldf2) / abs(ldf1)
        elif np.isfinite(ldf1) and np.isfinite(ldf2):
            log_det_rel = abs(ldf1 - ldf2)
        else:
            log_det_rel = float('inf')
        
        F_diff = F1 - F2
        frob_F1 = torch.norm(F1, p='fro').item()
        if frob_F1 > 1e-10:
            frob_rel = torch.norm(F_diff, p='fro').item() / frob_F1
        else:
            frob_rel = torch.norm(F_diff, p='fro').item()
        
        max_F1 = F1.abs().max().item()
        if max_F1 > 1e-10:
            elem_max_rel = F_diff.abs().max().item() / max_F1
        else:
            elem_max_rel = F_diff.abs().max().item()
    
    # Convergence: derivative vector convergence must pass;
    # Fisher convergence must also pass if covariance is well-conditioned
    is_converged = deriv_vec_conv < tolerance
    if covariance_ok and np.isfinite(frob_rel):
        is_converged = is_converged and frob_rel < tolerance
    
    return FisherConvergenceResult(
        fisher_matrix1=F1,
        fisher_matrix2=F2,
        log_det_fisher1=ldf1,
        log_det_fisher2=ldf2,
        log_det_relative_change=log_det_rel,
        frobenius_relative_change=frob_rel,
        element_wise_max_change=elem_max_rel,
        derivative_vector_convergence=deriv_vec_conv,
        per_param_derivative_convergence=per_param_dc,
        delta_theta1=delta_theta1,
        delta_theta2=delta_theta2,
        is_converged=is_converged,
        tolerance_used=tolerance,
        n_samples=n_s,
        n_features=n_features,
        covariance_well_conditioned=covariance_ok
    )


def print_fisher_convergence_report(result: FisherConvergenceResult):
    """Print a formatted report of Fisher convergence results."""
    print("\n" + "=" * 60)
    print("FISHER CONVERGENCE TEST")
    print("  F(δ₁) vs F(δ₂) where δ₂ = δ₁/2")
    print("=" * 60)
    
    print(f"\nStep sizes:")
    print(f"  δ₁ = {result.delta_theta1.tolist()}")
    print(f"  δ₂ = {result.delta_theta2.tolist()}")
    print(f"  n_samples = {result.n_samples}, n_features = {result.n_features}")
    
    print(f"\nDerivative vector convergence (||Δμ'|| / ||μ'||):")
    print(f"  Mean across params = {result.derivative_vector_convergence:.4f}")
    for pidx, dc in sorted(result.per_param_derivative_convergence.items()):
        print(f"  Parameter {pidx}: {dc:.4f}")
    
    if result.covariance_well_conditioned:
        print(f"\nFisher matrix log-determinants:")
        print(f"  log det F(δ₁) = {result.log_det_fisher1:.4f}")
        print(f"  log det F(δ₂) = {result.log_det_fisher2:.4f}")
        
        print(f"\nFisher convergence metrics:")
        print(f"  |Δ log det F| / |log det F| = {result.log_det_relative_change:.4f}")
        print(f"  ||ΔF||_F / ||F||_F          = {result.frobenius_relative_change:.4f}")
        print(f"  max|ΔF_ij| / max|F_ij|      = {result.element_wise_max_change:.4f}")
        
        n = result.fisher_matrix1.shape[0]
        print(f"\nFisher matrix F(δ₁):")
        for i in range(n):
            row = "  [" + ", ".join(f"{result.fisher_matrix1[i,j].item():.4f}" for j in range(n)) + "]"
            print(row)
        print(f"\nFisher matrix F(δ₂):")
        for i in range(n):
            row = "  [" + ", ".join(f"{result.fisher_matrix2[i,j].item():.4f}" for j in range(n)) + "]"
            print(row)
    else:
        print(f"\n  ⚠ Covariance not well-conditioned (n_samples={result.n_samples} ≤ n_features={result.n_features})")
        print(f"    Fisher matrix convergence skipped; using derivative convergence only.")
    
    print(f"\nAssessment (tolerance = {result.tolerance_used}):")
    if result.is_converged:
        print("  ✓ CONVERGED - Numerical derivatives are stable under step size changes")
    else:
        print("  ✗ NOT CONVERGED - Derivatives change significantly with step size")
        if result.derivative_vector_convergence >= result.tolerance_used:
            print(f"    - Derivative vector convergence too large: {result.derivative_vector_convergence:.4f}")
        if result.covariance_well_conditioned and np.isfinite(result.frobenius_relative_change):
            if result.frobenius_relative_change >= result.tolerance_used:
                print(f"    - Frobenius relative change too large: {result.frobenius_relative_change:.4f}")
    print("=" * 60 + "\n")


class LinearityChecker(nn.Module):
    """
    Check linearity of summary statistic response to parameter changes.
    
    For Fisher information to be accurately computed via numerical derivatives,
    the summary statistic D(θ) must respond linearly to small parameter changes:
    
        D(θ + dθ) ≈ D(θ) + dθ * (dD/dθ)
    
    This implies:
        1. D(θ+dθ) + D(θ-dθ) - 2*D(θ) ≈ 0 (symmetric around fiducial)
        2. [D(θ+dθ1) - D(θ-dθ1)]/dθ1 ≈ [D(θ+dθ2) - D(θ-dθ2)]/dθ2
    """
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize linearity checker.
        
        Args:
            tolerance: Maximum allowed relative deviation for linearity
                      (default: 0.1 = 10% deviation allowed)
        """
        super().__init__()
        self.tolerance = tolerance
    
    def __repr__(self):
        return f"LinearityChecker(tolerance={self.tolerance})"
    
    def forward(
        self,
        summaries_fid: torch.Tensor,
        summaries_plus1: torch.Tensor,
        summaries_minus1: torch.Tensor,
        summaries_plus2: torch.Tensor,
        summaries_minus2: torch.Tensor,
        delta_theta1: torch.Tensor,
        delta_theta2: torch.Tensor,
        param_idx: int = 0
    ) -> LinearityResult:
        """
        Perform linearity check on summary statistics.
        
        Args:
            summaries_fid: Summaries at fiducial θ, shape (n_samples, n_features)
            summaries_plus1: Summaries at θ + dθ1, shape (n_samples, n_features)
            summaries_minus1: Summaries at θ - dθ1, shape (n_samples, n_features)
            summaries_plus2: Summaries at θ + dθ2, shape (n_samples, n_features)
            summaries_minus2: Summaries at θ - dθ2, shape (n_samples, n_features)
            delta_theta1: First step size (scalar or tensor)
            delta_theta2: Second step size (scalar or tensor)
            param_idx: Index of parameter being varied (for multi-parameter cases)
        
        Returns:
            LinearityResult with all computed metrics
        """
        # Compute mean summary vectors
        mean_fid = summaries_fid.mean(dim=0)
        mean_plus1 = summaries_plus1.mean(dim=0)
        mean_minus1 = summaries_minus1.mean(dim=0)
        mean_plus2 = summaries_plus2.mean(dim=0)
        mean_minus2 = summaries_minus2.mean(dim=0)
        
        # Standard deviation of fiducial (for normalization)
        std_fid = summaries_fid.std(dim=0)
        # Avoid division by zero
        std_fid = torch.clamp(std_fid, min=1e-10)
        
        # Symmetric deviation: D(θ+dθ) + D(θ-dθ) - 2*D(θ)
        # For linear response, this should be ~0
        sym_dev1 = mean_plus1 + mean_minus1 - 2 * mean_fid
        sym_dev2 = mean_plus2 + mean_minus2 - 2 * mean_fid
        
        # Relative deviation (normalized by std)
        rel_dev1 = sym_dev1 / std_fid
        rel_dev2 = sym_dev2 / std_fid
        
        # Get scalar step size for this parameter
        dtheta1 = delta_theta1[param_idx] if delta_theta1.numel() > 1 else delta_theta1.item()
        dtheta2 = delta_theta2[param_idx] if delta_theta2.numel() > 1 else delta_theta2.item()
        
        # Numerical derivatives using centered differences
        # d/dθ ≈ [D(θ+dθ) - D(θ-dθ)] / (2*dθ)
        # But since we use θ ± dθ/2, the full step is dθ
        deriv1 = (mean_plus1 - mean_minus1) / dtheta1
        deriv2 = (mean_plus2 - mean_minus2) / dtheta2
        
        # Derivative ratio (should be ~1 for linear response)
        # Avoid division by zero
        deriv2_safe = torch.where(torch.abs(deriv2) > 1e-10, deriv2, torch.ones_like(deriv2) * 1e-10)
        deriv_ratio = deriv1 / deriv2_safe
        
        # Summary statistics
        linearity_score1 = torch.abs(rel_dev1).mean().item()
        linearity_score2 = torch.abs(rel_dev2).mean().item()
        
        # Derivative consistency: how close is the mean ratio to 1?
        # Only consider features with meaningful derivatives
        meaningful_mask = torch.abs(deriv2) > 1e-8
        if meaningful_mask.sum() > 0:
            deriv_consistency = torch.abs(deriv_ratio[meaningful_mask] - 1).mean().item()
        else:
            deriv_consistency = float('inf')
        
        # Pass/fail assessment
        is_linear = (linearity_score1 < self.tolerance and 
                    linearity_score2 < self.tolerance and
                    deriv_consistency < self.tolerance)
        
        return LinearityResult(
            delta_theta1=delta_theta1,
            delta_theta2=delta_theta2,
            mean_fiducial=mean_fid,
            mean_plus1=mean_plus1,
            mean_minus1=mean_minus1,
            mean_plus2=mean_plus2,
            mean_minus2=mean_minus2,
            symmetric_deviation1=sym_dev1,
            symmetric_deviation2=sym_dev2,
            relative_deviation1=rel_dev1,
            relative_deviation2=rel_dev2,
            derivative1=deriv1,
            derivative2=deriv2,
            derivative_ratio=deriv_ratio,
            linearity_score1=linearity_score1,
            linearity_score2=linearity_score2,
            derivative_consistency=deriv_consistency,
            is_linear=is_linear,
            tolerance_used=self.tolerance
        )


def print_linearity_report(result: LinearityResult, param_name: str = "θ"):
    """
    Print a formatted report of linearity check results.
    
    Args:
        result: LinearityResult from LinearityChecker
        param_name: Name of the parameter being varied
    """
    print("\n" + "=" * 60)
    print(f"LINEARITY CHECK REPORT for parameter: {param_name}")
    print("=" * 60)
    
    print(f"\nStep sizes:")
    print(f"  δθ₁ = {result.delta_theta1}")
    print(f"  δθ₂ = {result.delta_theta2}")
    
    print(f"\nSymmetric deviation test: D(θ+δθ) + D(θ-δθ) - 2·D(θ) ≈ 0")
    print(f"  Mean |relative deviation| for δθ₁: {result.linearity_score1:.4f}")
    print(f"  Mean |relative deviation| for δθ₂: {result.linearity_score2:.4f}")
    
    print(f"\nDerivative consistency test: dD/dθ should be step-size independent")
    print(f"  Mean |derivative ratio - 1|: {result.derivative_consistency:.4f}")
    
    print(f"\nAssessment (tolerance = {result.tolerance_used}):")
    if result.is_linear:
        print("  ✓ PASSED - Response is linear within tolerance")
    else:
        print("  ✗ FAILED - Response shows non-linear behavior")
        if result.linearity_score1 >= result.tolerance_used:
            print(f"    - Symmetric deviation too large for δθ₁")
        if result.linearity_score2 >= result.tolerance_used:
            print(f"    - Symmetric deviation too large for δθ₂")
        if result.derivative_consistency >= result.tolerance_used:
            print(f"    - Derivatives inconsistent across step sizes")
    
    print("\n" + "-" * 60)
    print("Per-feature statistics (first 10 features):")
    n_show = min(10, result.mean_fiducial.shape[0])
    print(f"{'Feature':<10} {'Mean(fid)':<12} {'Sym.Dev1':<12} {'Sym.Dev2':<12} {'Deriv.Ratio':<12}")
    for i in range(n_show):
        print(f"{i:<10} {result.mean_fiducial[i].item():<12.4f} "
              f"{result.relative_deviation1[i].item():<12.4f} "
              f"{result.relative_deviation2[i].item():<12.4f} "
              f"{result.derivative_ratio[i].item():<12.4f}")
    
    if result.mean_fiducial.shape[0] > n_show:
        print(f"  ... ({result.mean_fiducial.shape[0] - n_show} more features)")
    
    print("=" * 60 + "\n")


def _vectorization_needs_fit(vectorization) -> bool:
    """Check if a vectorization module still needs .fit() to be called.
    
    TopK layers save k_values in checkpoint → no fit needed.
    PersistenceImage layers need birth/death range fitting → always need fit.
    """
    # Check for CombinedVectorization with sub-layers
    if hasattr(vectorization, 'layers'):
        for layer in vectorization.layers:
            layer_type = type(layer).__name__
            # PersistenceImage always needs fitting (ranges not in checkpoint)
            if 'PersistenceImage' in layer_type:
                if hasattr(layer, 'fitted') and not layer.fitted:
                    return True
                # Also check if the layer has the fitted flag at all
                if not hasattr(layer, 'fitted'):
                    return True
    
    # Single vectorization module
    layer_type = type(vectorization).__name__
    if 'PersistenceImage' in layer_type:
        if hasattr(vectorization, 'fitted') and not vectorization.fitted:
            return True
        if not hasattr(vectorization, 'fitted'):
            return True
    
    return False


def run_linearity_check_from_pipeline(
    pipeline,
    theta_fid: torch.Tensor,
    delta_theta1: torch.Tensor,
    delta_theta2: torch.Tensor,
    n_samples: int = 100,
    seed: int = 42,
    param_indices: Optional[List[int]] = None,
    tolerance: float = 0.1,
    verbose: bool = True,
    batch_size: int = 1000,
    skip_vectorization_fit: bool = False,
    skip_compression: bool = False
) -> Dict[int, LinearityResult]:
    """
    Run linearity check using a TopoFisher pipeline.
    
    This function generates data at multiple parameter values and
    computes the linearity metrics for each parameter of interest.
    
    Args:
        pipeline: A BasePipeline or similar with simulator, filtration, vectorization
        theta_fid: Fiducial parameter values
        delta_theta1: First set of step sizes (one per parameter)
        delta_theta2: Second set of step sizes (typically smaller)
        n_samples: Number of samples to generate at each parameter value
        seed: Random seed for reproducibility
        param_indices: Which parameters to check (default: all)
        tolerance: Tolerance for linearity assessment
        verbose: Whether to print progress and reports
        batch_size: Batch size for processing filtration/vectorization (memory management)
        skip_vectorization_fit: If True, skip fitting vectorization (use when loaded from checkpoint)
        skip_compression: If True, check linearity at the vectorization stage (before compression).
                         This is appropriate for MOPED compression, which is a linear operation
                         and cannot be applied point-by-point (it needs the full data layout).
    
    Returns:
        Dictionary mapping param_index -> LinearityResult
    """
    import gc
    
    if param_indices is None:
        param_indices = list(range(theta_fid.numel()))
    
    checker = LinearityChecker(tolerance=tolerance)
    results = {}
    
    # Determine device from pipeline — check ALL components (filtration, vectorization, compression)
    # Some components (e.g. cubical filtration) have no parameters, so we must check all.
    device = torch.device('cpu')
    for component in [pipeline.filtration, pipeline.vectorization, pipeline.compression]:
        if hasattr(component, 'parameters'):
            try:
                device = next(component.parameters()).device
                break  # Found a component with parameters on a device
            except StopIteration:
                continue  # No parameters in this component, try next
    
    def generate_and_process(theta: torch.Tensor, seed_val: int, desc: str) -> torch.Tensor:
        """Generate data and process through full pipeline in batches, returning CPU summaries."""
        all_summaries = []
        
        # Generate and process in batches to avoid memory issues
        n_remaining = n_samples
        batch_seed = seed_val
        
        while n_remaining > 0:
            current_batch = min(batch_size, n_remaining)
            
            # Generate batch
            batch_data = pipeline.simulator.generate(
                theta=theta,
                n_samples=current_batch,
                seed=batch_seed,
                desc=None  # Suppress progress bar for batches
            )
            
            # Move data to same device as model for processing
            batch_data = batch_data.to(device)
            
            # Process through filtration (on model's device)
            batch_diagrams = pipeline.filtration(batch_data)
            del batch_data
            
            # Process through vectorization
            batch_vectors = pipeline.vectorization(batch_diagrams)
            del batch_diagrams
            
            # Process through compression (skip for MOPED which needs full data layout)
            if skip_compression:
                batch_summaries = batch_vectors
            else:
                compressed_list = pipeline.compression([batch_vectors])
                batch_summaries = compressed_list[0]
            del batch_vectors
            
            # Move summaries to CPU to free GPU memory
            if hasattr(batch_summaries, 'is_cuda') and batch_summaries.is_cuda:
                batch_summaries = batch_summaries.cpu()
            
            all_summaries.append(batch_summaries)
            
            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            n_remaining -= current_batch
            batch_seed += 1000  # Different seed for each batch
        
        return torch.cat(all_summaries, dim=0)
    
    # Collect per-parameter mean derivatives for Fisher convergence test
    _per_param_deriv1 = {}  # {param_idx: mean_derivative1_tensor}
    _per_param_deriv2 = {}  # {param_idx: mean_derivative2_tensor}
    _fiducial_summaries = None  # Will store fiducial samples for covariance
    
    for param_idx in param_indices:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Checking linearity for parameter {param_idx}")
            print(f"{'='*60}")
        
        # First, fit vectorization on a small sample if needed.
        # For TopK: skip if checkpoint was loaded (k values restored).
        # For PersistenceImage: ALWAYS fit (birth/death ranges not saved in checkpoint).
        needs_fit = False
        if hasattr(pipeline.vectorization, 'fit'):
            if not skip_vectorization_fit:
                needs_fit = True  # No checkpoint, always fit
            else:
                # Checkpoint was loaded — check if vectorization still needs fitting
                # (e.g. PersistenceImage needs birth/death ranges that aren't in checkpoint)
                needs_fit = _vectorization_needs_fit(pipeline.vectorization)
        
        if needs_fit:
            if verbose:
                print("Fitting vectorization on sample data...")
            fit_batch_size = min(100, n_samples)
            fit_diagrams = []
            
            # Generate small samples for fitting
            for i, (theta, s) in enumerate([
                (theta_fid, seed),
                (theta_fid.clone(), seed + 1),  # We'll use fiducial for fitting
            ]):
                fit_data = pipeline.simulator.generate(
                    theta=theta, n_samples=fit_batch_size, seed=s, desc=None
                )
                # Move to model's device for processing
                fit_data = fit_data.to(device)
                fit_diagrams.append(pipeline.filtration(fit_data))
                del fit_data
            
            pipeline.vectorization.fit(fit_diagrams)
            del fit_diagrams
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Process fiducial
        if verbose:
            print(f"Processing fiducial samples at θ = {theta_fid} (n={n_samples}, batch_size={batch_size})")
        summaries_fid = generate_and_process(theta_fid, seed, "Fiducial")
        
        # Save fiducial summaries for Fisher convergence (first param only, same data)
        if _fiducial_summaries is None:
            _fiducial_summaries = summaries_fid.clone()
        
        # Process θ + dθ1
        theta_plus1 = theta_fid.clone()
        theta_plus1[param_idx] += delta_theta1[param_idx]
        if verbose:
            print(f"Processing samples at θ + δθ₁ = {theta_plus1}")
        summaries_plus1 = generate_and_process(theta_plus1, seed + 1, "θ + δθ₁")
        
        # Process θ - dθ1
        theta_minus1 = theta_fid.clone()
        theta_minus1[param_idx] -= delta_theta1[param_idx]
        if verbose:
            print(f"Processing samples at θ - δθ₁ = {theta_minus1}")
        summaries_minus1 = generate_and_process(theta_minus1, seed + 2, "θ - δθ₁")
        
        # Process θ + dθ2
        theta_plus2 = theta_fid.clone()
        theta_plus2[param_idx] += delta_theta2[param_idx]
        if verbose:
            print(f"Processing samples at θ + δθ₂ = {theta_plus2}")
        summaries_plus2 = generate_and_process(theta_plus2, seed + 3, "θ + δθ₂")
        
        # Process θ - dθ2
        theta_minus2 = theta_fid.clone()
        theta_minus2[param_idx] -= delta_theta2[param_idx]
        if verbose:
            print(f"Processing samples at θ - δθ₂ = {theta_minus2}")
        summaries_minus2 = generate_and_process(theta_minus2, seed + 4, "θ - δθ₂")
        
        # Run linearity check (all summaries are on CPU now)
        result = checker(
            summaries_fid=summaries_fid,
            summaries_plus1=summaries_plus1,
            summaries_minus1=summaries_minus1,
            summaries_plus2=summaries_plus2,
            summaries_minus2=summaries_minus2,
            delta_theta1=delta_theta1,
            delta_theta2=delta_theta2,
            param_idx=param_idx
        )
        
        results[param_idx] = result
        
        # Save mean derivatives for Fisher convergence test
        _per_param_deriv1[param_idx] = result.derivative1.clone()
        _per_param_deriv2[param_idx] = result.derivative2.clone()
        
        # Clean up summaries
        del summaries_fid, summaries_plus1, summaries_minus1, summaries_plus2, summaries_minus2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if verbose:
            print_linearity_report(result, param_name=f"θ[{param_idx}]")
    
    # Compute Fisher convergence test
    fisher_convergence = None
    if len(param_indices) >= 1 and _fiducial_summaries is not None:
        if verbose:
            print(f"\nComputing Fisher convergence test...")
        try:
            fisher_convergence = compute_fisher_convergence(
                summaries_fid=_fiducial_summaries,
                per_param_derivatives1=_per_param_deriv1,
                per_param_derivatives2=_per_param_deriv2,
                delta_theta1=delta_theta1,
                delta_theta2=delta_theta2,
                n_samples=n_samples,
                tolerance=tolerance
            )
            if verbose:
                print_fisher_convergence_report(fisher_convergence)
        except Exception as e:
            if verbose:
                print(f"  ⚠ Fisher convergence test failed: {e}")
    
    del _fiducial_summaries, _per_param_deriv1, _per_param_deriv2
    gc.collect()
    
    return results, fisher_convergence
