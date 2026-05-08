"""
MOPED (Multiple Optimized Parameter Estimation and Data compression) compression.

Reference: Heavens et al. 2000 (https://arxiv.org/abs/astro-ph/9911102)

Note: While the original paper uses actual parameter step sizes (delta_theta) in
the compression matrix computation, we use unit values internally. This is simpler
and produces the same Fisher information matrix due to MOPED's scale-equivariance
property. The FisherAnalyzer will apply the correct finite differences regardless
of what scaling MOPED uses internally.
"""
from typing import List, Optional, Tuple
import torch

from . import Compression


class MOPEDCompression(Compression):
    """
    MOPED compression: B = C^{-1} dμ

    Where:
    - C is the covariance matrix of summaries
    - dμ are the derivatives of mean summaries w.r.t. parameters
    - B is the compression matrix: (n_features, n_params)

    The compressed summaries are: compressed = summaries @ B

    This compression is optimal for Gaussian likelihoods and maximizes
    the Fisher information preserved for a given compression dimension.

    Note: MOPED always splits data internally and returns ONLY the test set.
    This is because it computes the compression matrix on train data and
    evaluates on test data to avoid overfitting.
    """

    def __init__(
        self,
        train_frac: float = 0.5,
        clean_data: bool = True,
        reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize MOPED compression.

        Args:
            train_frac: Fraction of data for training set
                       Compression matrix is learned on train set, applied to test set
            clean_data: If True, remove zero-variance features before compression
            reg: Regularization parameter for covariance matrix (C + reg*I) to avoid singularity
            seed: Random seed for reproducible train/test splits
        """
        super().__init__()
        self.train_frac = train_frac
        self.clean_data = clean_data
        self.reg = reg
        self.seed = seed
        self._cached_compression_matrix = None
        self._cached_valid_idx = None

    def __repr__(self):
        """String representation showing configuration."""
        return (f"MOPEDCompression(train_frac={self.train_frac}, "
                f"clean_data={self.clean_data}, reg={self.reg})")

    def split_data(self, summaries: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Split summaries into train/test sets using a local RNG.

        Important: theta_minus and theta_plus pairs use the same random seed during generation,
        so they must use the same permutation to maintain pairing.

        Args:
            summaries: List of tensors [fiducial, theta_minus_0, theta_plus_0, theta_minus_1, theta_plus_1, ...]

        Returns:
            (train_summaries, test_summaries)
        """

        def split_with_perm(tensor, perm, train_frac):
            """Split a tensor using a given permutation."""
            n = tensor.shape[0]
            shuffled = tensor[perm]
            n_train = int(n * train_frac)
            return shuffled[:n_train], shuffled[n_train:]

        # Create local generator for reproducible splits without affecting global state
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        # Fiducial gets its own permutation
        perm_fid = torch.randperm(summaries[0].shape[0], generator=generator)

        # Each parameter pair (theta_minus, theta_plus) shares a permutation
        n_params = (len(summaries) - 1) // 2
        perms_deriv = [torch.randperm(summaries[1 + 2*i].shape[0], generator=generator) for i in range(n_params)]

        # Apply permutations and split
        split_summaries_list = []

        # Fiducial
        split_summaries_list.append(split_with_perm(summaries[0], perm_fid, self.train_frac))

        # Derivatives (pairs share permutation to maintain seed pairing)
        for i in range(n_params):
            perm = perms_deriv[i]
            split_summaries_list.append(split_with_perm(summaries[1 + 2*i], perm, self.train_frac))
            split_summaries_list.append(split_with_perm(summaries[2 + 2*i], perm, self.train_frac))

        # Reorganize into separate lists
        train_summaries = [s[0] for s in split_summaries_list]
        test_summaries = [s[1] for s in split_summaries_list]

        return train_summaries, test_summaries

    def cache_compression_matrix(self, summaries: List[torch.Tensor]):
        """
        Pre-compute and cache the MOPED compression matrix from a large sample.

        When cached, forward() in training mode uses the cached matrix and
        applies it to ALL input summaries (no train/test split), giving the
        CNN a much more stable gradient signal.

        Call this periodically during training (e.g., every 50 epochs) with
        summaries computed from the full training set.

        Args:
            summaries: List of summary tensors [fid, minus_0, plus_0, ...]
                       from a large sample (ideally full training set).
        """
        # Clean and remember which features were kept
        if self.clean_data:
            summaries = self._clean_summaries(summaries)
            self._cached_valid_idx = self._last_valid_idx

        # Use the standard train/test split to compute the matrix
        train_summaries, _ = self.split_data(summaries)
        self._cached_compression_matrix = self._compute_moped_matrix(train_summaries).detach()

    def clear_compression_cache(self):
        """Clear the cached compression matrix (reverts to per-batch computation)."""
        self._cached_compression_matrix = None
        self._cached_valid_idx = None

    def forward(
        self,
        summaries: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute and apply MOPED compression to summaries.

        In training mode with a cached compression matrix, applies the cached
        matrix to ALL input summaries (no train/test split). This avoids the
        catastrophically ill-conditioned covariance estimation from small batches.

        In eval mode (or without cache), uses the standard approach: split into
        train/test, compute matrix from train, apply to test.

        Args:
            summaries: List of summary tensors [fid, minus_0, plus_0, minus_1, plus_1, ...]

        Returns:
            Compressed summaries with shape (n_samples, n_params)
        """
        # Training mode with cached matrix: apply cached valid_idx + matrix
        # (skip batch-level _clean_summaries which may remove different features)
        if self.training and self._cached_compression_matrix is not None:
            if self._cached_valid_idx is not None:
                summaries = [s[:, self._cached_valid_idx] for s in summaries]
            compressed_summaries = [
                s @ self._cached_compression_matrix for s in summaries
            ]
            return compressed_summaries

        # Clean data if requested (standard path)
        if self.clean_data:
            summaries = self._clean_summaries(summaries)

        # Standard path: split, compute, apply to test only
        train_summaries, test_summaries = self.split_data(summaries)

        # Compute compression matrix using ONLY train set
        compression_matrix = self._compute_moped_matrix(train_summaries)
        self.compression_matrix = compression_matrix  # store for external use

        # Apply compression to test summaries only
        compressed_summaries = [s @ compression_matrix for s in test_summaries]

        return compressed_summaries

    def _compute_moped_matrix(
        self,
        summaries: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute MOPED compression matrix: B = C^{-1} dμ

        When n_samples < n_features (common for learnable filtrations where
        d = N×N can be very large), we use the Woodbury/dual formulation to
        avoid building and inverting the full d×d covariance matrix.

        Dual trick: If X is the (n, d) centered data matrix, then
            C = X^T X / (n-1)  [d×d, rank ≤ n-1]
            C^{-1} b ≈ X^T (X X^T + λI)^{-1} X b / ... 

        More precisely, we use the SVD of X to compute C^{-1} b efficiently:
            X = U S V^T   (thin SVD: U is n×r, S is r×r, V is d×r, r = min(n,d))
            C = V (S²/(n-1)) V^T
            C^{-1} b = V ((n-1)/S²) V^T b   (with regularization on small singular values)

        This reduces the cost from O(d³) to O(n²d + n³), which for d=1024, n=50
        is ~400× faster.

        Args:
            summaries: List of summary tensors (already split, train set)

        Returns:
            Compression matrix of shape (n_features, n_params)
        """
        # Extract fiducial and finite difference summaries
        vecs_cov = summaries[0]  # (n_samples, d)
        perturbed_vecs = summaries[1:]

        # Compute finite differences
        finite_diffs = self._compute_finite_differences(perturbed_vecs)
        mean_derivatives = finite_diffs.mean(dim=1)  # (n_params, d)

        n_samples, n_features = vecs_cov.shape

        # Center the data
        vecs_centered = vecs_cov - vecs_cov.mean(dim=0, keepdim=True)  # (n, d)

        if n_samples < n_features:
            # ── Dual/SVD path: work in n-dimensional space ──────────────
            # Thin SVD of centered data: X = U S V^T
            # U: (n, r), S: (r,), V: (d, r)  where r = min(n, d)
            U, S, Vh = torch.linalg.svd(vecs_centered, full_matrices=False)
            V = Vh.T  # (d, r)

            # Singular values → eigenvalues of covariance: λ_i = s_i² / (n-1)
            eigvals = S ** 2 / (n_samples - 1)

            # Adaptive regularization based on spectrum
            trace_est = eigvals.sum()
            reg_val = max(self.reg, 1e-4 * trace_est / n_features)

            # Apply Hartlap correction if valid
            if n_samples > n_features + 2:
                hartlap_factor = (n_samples - n_features - 2.0) / (n_samples - 1.0)
                eigvals = eigvals / hartlap_factor

            # Regularized inverse eigenvalues: 1 / (λ_i + reg)
            inv_eigvals = 1.0 / (eigvals + reg_val)

            # C^{-1} dμ^T = V diag(1/λ) V^T dμ^T
            # Step 1: project dμ into eigenspace: V^T dμ^T  →  (r, n_params)
            proj = V.T @ mean_derivatives.T  # (r, n_params)
            # Step 2: scale by inverse eigenvalues
            scaled = inv_eigvals.unsqueeze(-1) * proj  # (r, n_params)
            # Step 3: project back: V @ scaled  →  (d, n_params)
            compression_matrix = V @ scaled

        else:
            # ── Standard path: d×d covariance is feasible ───────────────
            C = self._compute_covariance(vecs_cov)

            # Adaptive regularization
            reg_val = self.reg
            trace_C = torch.trace(C)
            if trace_C > 0:
                adaptive_reg = max(self.reg, 1e-4 * trace_C / C.shape[0])
                reg_val = adaptive_reg
            C = C + reg_val * torch.eye(n_features, device=C.device, dtype=C.dtype)

            # Solve C @ B = dμ^T
            compression_matrix = self._robust_solve(C, mean_derivatives.T)

        return compression_matrix
    
    def _robust_solve(self, C: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Robustly solve C @ x = b using eigenvalue decomposition with truncation.
        
        This handles ill-conditioned covariance matrices by:
        1. Trying direct solve
        2. Falling back to pinv
        3. Using eigenvalue decomposition with small eigenvalue truncation
        """
        # Try direct solve first
        try:
            return torch.linalg.solve(C, b)
        except:
            pass
        
        # Try pseudo-inverse
        try:
            C_pinv = torch.linalg.pinv(C, rcond=1e-6)
            return C_pinv @ b
        except:
            pass
        
        # Eigenvalue decomposition with truncation
        try:
            # Symmetric eigendecomposition: C = V @ diag(w) @ V^T
            w, V = torch.linalg.eigh(C)
            
            # Truncate small/negative eigenvalues
            max_w = w.abs().max()
            threshold = max(1e-6 * max_w, 1e-10)
            w_safe = torch.where(w > threshold, w, threshold * torch.ones_like(w))
            
            # Inverse: C^{-1} = V @ diag(1/w) @ V^T
            w_inv = 1.0 / w_safe
            C_inv = V @ torch.diag(w_inv) @ V.T
            
            return C_inv @ b
        except Exception as e:
            # Last resort: use diagonal approximation
            print(f"Warning: All inverse methods failed ({e}), using diagonal approximation")
            diag_C = torch.diag(C)
            diag_C_safe = torch.where(diag_C > 1e-10, diag_C, 1e-10 * torch.ones_like(diag_C))
            return b / diag_C_safe.unsqueeze(-1) if b.dim() > 1 else b / diag_C_safe

    def _clean_summaries(self, summaries: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Remove zero-variance features.

        Note: We intentionally do NOT remove correlated features here.
        Ill-conditioned covariance matrices are handled downstream by:
        1. Adaptive regularization (C + λI) in _compute_moped_matrix
        2. Eigenvalue truncation in _robust_solve
        This is both faster (O(d) vs O(d²)) and more numerically principled.

        Args:
            summaries: List of summary tensors

        Returns:
            Cleaned summaries with zero-variance features removed
        """
        vecs_cov = summaries[0]

        # Compute variance — O(d), cheap
        var = vecs_cov.var(dim=0)
        valid_idx = var > 1e-10

        if (~valid_idx).any():
            n_removed = (~valid_idx).sum().item()
            if not getattr(self, '_warned_zero_var', False):
                print(f"MOPEDCompression: Removing {n_removed} zero-variance features (suppressing further)")
                self._warned_zero_var = True

        # Store for cache_compression_matrix to pick up
        self._last_valid_idx = valid_idx

        # Filter all summaries
        return [s[:, valid_idx] for s in summaries]

    def _compute_covariance(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Compute covariance matrix with Hartlap correction when applicable.

        The Hartlap et al. (2007) correction factor (n-d-2)/(n-1) corrects
        the bias in the inverse of the sample covariance. It is only valid
        when n_samples > n_features + 2; otherwise the sample covariance is
        singular anyway and the correction is undefined. In that regime we
        return the plain sample covariance and rely on downstream
        regularization (adaptive reg + eigenvalue truncation in _robust_solve).

        Args:
            vecs: Tensor of shape (n_samples, n_features)

        Returns:
            Covariance matrix of shape (n_features, n_features)
        """
        n_samples, n_features = vecs.shape

        # Compute sample covariance
        vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)
        cov = (vecs_centered.T @ vecs_centered) / (n_samples - 1)

        # Hartlap correction — only valid when n > d + 2
        if n_samples > n_features + 2:
            hartlap_factor = (n_samples - n_features - 2.0) / (n_samples - 1.0)
            cov = cov / hartlap_factor

        return cov

    def _compute_finite_differences(
        self,
        perturbed_vecs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute finite differences between plus and minus perturbations.

        We simply compute (plus - minus) without any scaling factor.
        The actual scaling doesn't matter because the Fisher matrix is invariant
        to this choice - FisherAnalyzer will apply the correct scaling based on
        the actual parameter step sizes used in data generation.

        Args:
            perturbed_vecs: List of tensors ordered as [minus_0, plus_0, minus_1, plus_1, ...]

        Returns:
            Tensor of shape (n_params, n_d, n_features)
        """
        n_params = len(perturbed_vecs) // 2
        finite_diffs = []

        for i in range(n_params):
            vecs_minus = perturbed_vecs[2 * i]      # theta minus perturbation
            vecs_plus = perturbed_vecs[2 * i + 1]   # theta plus perturbation

            # Simple difference - no scaling needed
            # Fisher matrix invariant to this choice
            diff = vecs_plus - vecs_minus
            finite_diffs.append(diff)

        return torch.stack(finite_diffs)  # (n_params, n_d, n_features)
