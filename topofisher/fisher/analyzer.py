"""
Fisher information analysis.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
from ..config import FisherResult
from .gaussianity import test_gaussianity


class FisherAnalyzer(nn.Module):
    """
    Compute Fisher information matrix from summary statistics.

    Uses centered finite differences for derivatives and
    estimates covariance from simulations at fiducial parameters.
    """

    def __init__(self, clean_data: bool = True):
        """
        Initialize Fisher analyzer.

        Args:
            clean_data: If True, remove zero-variance features
        """
        super().__init__()
        self.clean_data = clean_data

    def __repr__(self):
        """String representation showing configuration."""
        return f"FisherAnalyzer(clean_data={self.clean_data})"

    def forward(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor,
        check_gaussianity: bool = False,
        compute_moments: bool = False
    ) -> FisherResult:
        """
        Compute Fisher information from summaries.

        Args:
            summaries: List of summary tensors.
                summaries[0]: shape (n_s, n_features) at theta_fid (for covariance)
                summaries[1:]: shape (n_d, n_features) at perturbed values
                    Ordered as [..., theta_minus_i, theta_plus_i, ...]
            delta_theta: Tensor of shape (n_params,) with step sizes
            check_gaussianity: If True, run Gaussianity test on summaries
            compute_moments: If True, compute skewness/kurtosis penalties for regularization

        Returns:
            FisherResult containing Fisher matrix and related quantities
        """
        # Clean data if requested
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Compute Fisher matrix
        fisher_matrix, inv_fisher, mean_derivatives, C, log_det_fisher, constraints = \
            self._compute_fisher(summaries, delta_theta)

        # Gaussianity check
        is_gaussian = None
        gaussianity_details = None
        if check_gaussianity:
            results, is_gaussian = test_gaussianity(summaries, verbose=False)
            # Compute summary statistics for details
            n_datasets = len(results)
            n_datasets_all_gaussian = sum(1 for r in results.values() if r['all_gaussian'])
            total_tests = sum(r['n_total'] for r in results.values())
            total_passed = sum(r['n_gaussian'] for r in results.values())
            # Collect per-dataset, per-feature KS statistics
            ks_per_dataset = {}
            for dataset_name, dataset_result in results.items():
                ks_per_feature = {}
                for feature_key, feature_data in dataset_result['features'].items():
                    ks_d = feature_data['ks_statistic']
                    ks_p = feature_data['ks_pvalue']
                    ks_per_feature[feature_key] = {
                        'ks_statistic': None if (isinstance(ks_d, float) and ks_d != ks_d) else float(ks_d),
                        'ks_pvalue': float(ks_p),
                        'is_gaussian': bool(feature_data['is_gaussian'])
                    }
                ks_per_dataset[dataset_name] = {
                    'features': ks_per_feature,
                    'all_gaussian': bool(dataset_result['all_gaussian'])
                }

            gaussianity_details = {
                'n_datasets': n_datasets,
                'n_datasets_all_gaussian': n_datasets_all_gaussian,
                'total_tests': total_tests,
                'total_passed': total_passed,
                'alpha': 0.05,
                'test': 'Kolmogorov-Smirnov',
                'per_dataset': ks_per_dataset
            }

        # Compute moment penalties for regularization
        skewness_penalty = None
        kurtosis_penalty = None
        if compute_moments:
            skewness_penalty, kurtosis_penalty = self.compute_gaussianity_penalty(summaries)

        return FisherResult(
            fisher_matrix=fisher_matrix,
            inverse_fisher=inv_fisher,
            derivatives=mean_derivatives,
            covariance=C,
            log_det_fisher=log_det_fisher,
            constraints=constraints,
            is_gaussian=is_gaussian,
            gaussianity_details=gaussianity_details,
            skewness_penalty=skewness_penalty,
            kurtosis_penalty=kurtosis_penalty
        )

    def _compute_fisher(
        self,
        summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ):
        """
        Compute Fisher matrix and related quantities from summaries.

        Args:
            summaries: List of summary tensors
            delta_theta: Step sizes for derivatives

        Returns:
            Tuple of (fisher_matrix, inverse_fisher, mean_derivatives, covariance, log_det_fisher, constraints)
        """
        # Extract covariance samples
        vecs_cov = summaries[0]

        # Compute covariance matrix with Hartlap correction
        C = self._compute_covariance(vecs_cov)
        inv_C = self._robust_matrix_inverse(C, "covariance")

        # Compute derivatives using centered differences
        derivatives = self._compute_derivatives(summaries[1:], delta_theta)

        # Mean derivatives
        mean_derivatives = derivatives.mean(dim=1)

        # Fisher matrix: F_ij = dμ_i^T C^{-1} dμ_j
        fisher_matrix = mean_derivatives @ inv_C @ mean_derivatives.T

        # Fisher forecast (inverse Fisher matrix = covariance of parameters)
        # Use robust inverse for stability (especially during early training)
        inv_fisher = self._robust_matrix_inverse(fisher_matrix, "Fisher")

        # Constraints (1-sigma errors)
        constraints = torch.sqrt(torch.diag(inv_fisher).clamp(min=1e-10))

        # Log determinant - use regularized version if needed
        try:
            sign, logdet = torch.linalg.slogdet(fisher_matrix)
            log_det_fisher = sign * logdet
        except:
            # Fallback for singular matrix
            log_det_fisher = torch.tensor(-float('inf'), device=fisher_matrix.device)

        return fisher_matrix, inv_fisher, mean_derivatives, C, log_det_fisher, constraints
    
    def _robust_matrix_inverse(self, M: torch.Tensor, name: str = "matrix") -> torch.Tensor:
        """
        Robustly invert a matrix using multiple fallback strategies.
        """
        n = M.shape[0]
        
        # Try direct inverse
        try:
            return torch.linalg.inv(M)
        except:
            pass
        
        # Try with regularization
        try:
            eps = 1e-6 * torch.trace(M).abs() / n
            eps = max(eps.item(), 1e-10)
            M_reg = M + eps * torch.eye(n, device=M.device, dtype=M.dtype)
            return torch.linalg.inv(M_reg)
        except:
            pass
        
        # Try pseudo-inverse
        try:
            return torch.linalg.pinv(M, rcond=1e-6)
        except:
            pass
        
        # Eigenvalue decomposition with truncation
        try:
            w, V = torch.linalg.eigh(M)
            max_w = w.abs().max()
            threshold = max(1e-6 * max_w.item(), 1e-10)
            w_safe = torch.where(w > threshold, w, threshold * torch.ones_like(w))
            w_inv = 1.0 / w_safe
            return V @ torch.diag(w_inv) @ V.T
        except:
            pass
        
        # Last resort: diagonal approximation
        print(f"Warning: All {name} inverse methods failed, using diagonal approximation")
        diag = torch.diag(M)
        diag_safe = torch.where(diag.abs() > 1e-10, diag, 1e-10 * torch.ones_like(diag))
        return torch.diag(1.0 / diag_safe)

    def _clean_summaries(self, summaries: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Remove zero-variance features.

        Args:
            summaries: List of summary tensors

        Returns:
            Cleaned summaries
        """
        vecs_cov = summaries[0]

        # Compute variance
        var = vecs_cov.var(dim=0)
        valid_idx = var > 1e-10

        if (~valid_idx).any():
            if not getattr(self, '_warned_zero_var', False):
                print(f"FisherAnalyzer: Removing {(~valid_idx).sum().item()} zero-variance features (suppressing further)")
                self._warned_zero_var = True

        # Filter all summaries
        return [s[:, valid_idx] for s in summaries]

    def _compute_covariance(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix with Hartlap correction when applicable.

        The Hartlap et al. (2007) correction factor (n-d-2)/(n-1) corrects
        the bias in the inverse of the sample covariance. It is only valid
        when n_samples > n_features + 2; otherwise the sample covariance is
        singular anyway and the correction is undefined.

        Args:
            vecs: Tensor of shape (n_samples, n_features)

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        n_samples, n_features = vecs.shape

        # Compute covariance
        vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)
        cov = (vecs_centered.T @ vecs_centered) / (n_samples - 1)

        # Hartlap correction is only valid when n_samples > n_features + 2.
        # For tiny training batches (e.g. after MOPED train/test splitting),
        # the factor can be zero or negative, which would make the covariance
        # unusable. In that regime, skip the correction and add a tiny diagonal
        # regularizer for numerical stability.
        if n_samples > n_features + 2:
            hartlap_factor = (n_samples - n_features - 2.0) / (n_samples - 1.0)
            return cov / hartlap_factor

        if not getattr(self, '_warned_hartlap', False):
            print(
                f"FisherAnalyzer: Skipping Hartlap correction "
                f"(n_samples={n_samples}, n_features={n_features})"
            )
            self._warned_hartlap = True

        eps = 1e-6 * torch.trace(cov).abs() / max(1, n_features)
        eps = max(eps.item(), 1e-12)
        return cov + eps * torch.eye(n_features, device=cov.device, dtype=cov.dtype)

    def _compute_derivatives(
        self,
        perturbed_vecs: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute derivatives using centered finite differences.

        Args:
            perturbed_vecs: List of tensors ordered as [theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]
            delta_theta: Step sizes

        Returns:
            Tensor of shape (n_params, n_d, n_features)
        """
        n_params = len(delta_theta)

        # Validate input structure: should have pairs (minus, plus) for each parameter
        assert len(perturbed_vecs) == 2 * n_params, \
            f"Expected {2 * n_params} perturbed vectors (2 per parameter), got {len(perturbed_vecs)}"

        derivatives = []

        for i in range(n_params):
            vecs_minus = perturbed_vecs[2 * i]      # theta - delta/2
            vecs_plus = perturbed_vecs[2 * i + 1]   # theta + delta/2

            # Centered difference
            deriv = (vecs_plus - vecs_minus) / delta_theta[i]
            derivatives.append(deriv)

        return torch.stack(derivatives)  # (n_params, n_d, n_features)

    def compute_gaussianity_penalty(
        self,
        summaries: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute skewness and kurtosis penalties for Gaussianity regularization.

        - Skewness: E[(X - μ)³] / σ³ (should be 0 for Gaussian)
        - Excess kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (should be 0 for Gaussian)

        Args:
            summaries: List of summary tensors [fid, minus_0, plus_0, ...]
                Each tensor shape: (n_samples, n_features)

        Returns:
            (skewness_penalty, kurtosis_penalty): Mean of squared deviations
        """
        skew_penalty = torch.tensor(0.0, device=summaries[0].device)
        kurt_penalty = torch.tensor(0.0, device=summaries[0].device)
        n_summaries = 0

        for s in summaries:
            # Standardize to mean=0, std=1
            mean = s.mean(dim=0, keepdim=True)
            std = s.std(dim=0, keepdim=True) + 1e-8
            s_norm = (s - mean) / std

            # Skewness: E[(X - μ)^3] / σ^3
            skewness = torch.mean(s_norm ** 3, dim=0)
            skew_penalty = skew_penalty + torch.mean(skewness ** 2)

            # Excess kurtosis: E[(X - μ)^4] / σ^4 - 3
            kurtosis = torch.mean(s_norm ** 4, dim=0) - 3.0
            kurt_penalty = kurt_penalty + torch.mean(kurtosis ** 2)

            n_summaries += 1

        return skew_penalty / n_summaries, kurt_penalty / n_summaries
