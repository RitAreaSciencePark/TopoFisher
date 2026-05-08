"""Base class for learnable pipelines.

TODO: Wrap filtration, vectorization and compression into a single model.
Currently these components are separate, but they could be wrapped together
into a single nn.Module (as a PyTorch model) for cleaner learnable pipeline implementations.
This would enable end-to-end optimization and better gradient flow through
all components.
"""

import copy
import torch
import torch.nn as nn
from abc import abstractmethod
from typing import List, Tuple, Dict, Any
from ..base import BasePipeline, LazyStackedTensor, LAZY_TENSOR_TYPES
from ...config import AnalysisConfig, TrainingConfig, FisherResult
from ...vectorizations.differentiable_persistence_image import DifferentiablePersistenceImageLayer

class LearnablePipeline(BasePipeline):
    """
    Base class for pipelines with learnable components.

    Provides common infrastructure for training compression, vectorization,
    or filtration components to maximize Fisher information.
    """

    def _get_bandwidth_info(self) -> str:
        """Get learnable bandwidth info string for logging (empty if no learnable PI layers)."""
        parts = []
        for name, module in self.named_modules():
            if isinstance(module, DifferentiablePersistenceImageLayer) and module.learn_bandwidth:
                parts.append(f"σ={module.get_bandwidth():.4f}")
        if parts:
            return ", " + ", ".join(parts)
        return ""

    def _supports_topology_prewarm(self) -> bool:
        """Check whether filtration exposes topology cache APIs used for prewarm."""
        return (
            hasattr(self, 'filtration')
            and hasattr(self.filtration, 'get_topologies_cached_batch')
            and hasattr(self.filtration, 'clear_cache')
            and hasattr(self.filtration, 'get_cache_stats')
        )

    def _prewarm_topology_cache(self, data_list: List[torch.Tensor]) -> None:
        """Precompute topology cache entries for selected tensors in a dataset list.

        Processes in chunks of 500 to keep memory bounded.
        The filtration's get_topologies_cached_batch handles parallelism internally.
        """
        if not self._supports_topology_prewarm():
            return

        import time
        t0 = time.time()
        total_samples = 0
        for data in data_list:
            if isinstance(data, (torch.Tensor,) + LAZY_TENSOR_TYPES) and data.ndim >= 3:
                n = data.shape[0]
                total_samples += n
                # Process in chunks to keep memory bounded
                chunk_size = 500
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    _ = self.filtration.get_topologies_cached_batch(data[start:end])

        elapsed = time.time() - t0
        if total_samples > 0 and elapsed > 1.0:
            rate = total_samples / elapsed
            print(f"    Prewarm: {total_samples} samples in {elapsed:.1f}s ({rate:.0f} samples/s)")

    def _select_topology_prewarm_subset(
        self,
        data_list: List[torch.Tensor],
        max_samples: int,
    ) -> List[torch.Tensor]:
        """Select a bounded subset of samples for topology prewarm.

        Keeps per-dataset representation while capping total sample count.
        """
        tensor_sets = [d for d in data_list if isinstance(d, (torch.Tensor,) + LAZY_TENSOR_TYPES) and d.ndim >= 3]
        if len(tensor_sets) == 0:
            return []

        total = sum(int(d.shape[0]) for d in tensor_sets)
        if total <= max_samples:
            return tensor_sets

        per_set = max(1, max_samples // len(tensor_sets))
        subset = []
        for d in tensor_sets:
            take = min(int(d.shape[0]), per_set)
            subset.append(d[:take])

        # Fill remaining budget from larger sets if any budget remains.
        used = sum(int(d.shape[0]) for d in subset)
        remaining = max_samples - used
        if remaining > 0:
            for i, d in enumerate(tensor_sets):
                if remaining <= 0:
                    break
                already = int(subset[i].shape[0])
                cap = int(d.shape[0]) - already
                if cap <= 0:
                    continue
                extra = min(cap, remaining)
                subset[i] = d[:already + extra]
                remaining -= extra

        return subset

    def split_data(
        self,
        data_list: List[torch.Tensor],
        train_frac: float = 0.5,
        val_frac: float = 0.25,
        seed: int = 42
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Split data into train/val/test sets.

        IMPORTANT: For derivative data, theta_minus and theta_plus pairs
        must use the same permutation to maintain pairing from same seed.

        Args:
            data_list: [fid, minus_0, plus_0, minus_1, plus_1, ...]
            train_frac: Fraction for training
            val_frac: Fraction for validation
            seed: Random seed for reproducibility

        Returns:
            train_data, val_data, test_data (each maintaining list structure)
        """
        # Ensure every element is a Tensor (some callers pass plain lists
        # of tensors or lists of dicts).  LazyStackedTensor passes through
        # unchanged so that large multi-bin data stays mmap-backed.
        converted = []
        for d in data_list:
            if isinstance(d, (torch.Tensor,) + LAZY_TENSOR_TYPES):
                converted.append(d)
            elif isinstance(d, list):
                if len(d) > 0 and isinstance(d[0], torch.Tensor):
                    converted.append(torch.stack(d))
                elif len(d) > 0 and isinstance(d[0], dict):
                    # List of dicts with a 'data'/'pts' tensor — stack those
                    key = 'data' if 'data' in d[0] else 'pts' if 'pts' in d[0] else None
                    if key is not None:
                        converted.append(torch.stack([item[key] for item in d]))
                    else:
                        raise TypeError(
                            f"split_data: list-of-dicts has no 'data' or 'pts' key. "
                            f"Keys found: {list(d[0].keys())}"
                        )
                else:
                    converted.append(d)  # pass through, will fail downstream with clear error
            else:
                converted.append(d)
        data_list = converted

        def _subset(d, idx):
            """Index a dataset element, keeping lazy wrappers lazy."""
            if isinstance(d, LAZY_TENSOR_TYPES):
                return d.lazy_subset(idx)
            return d[idx]

        generator = torch.Generator()
        generator.manual_seed(seed)

        train_data = []
        val_data = []
        test_data = []

        # Split fiducial data
        n_fid = data_list[0].shape[0]
        n_train = int(n_fid * train_frac)
        n_val = int(n_fid * val_frac)

        perm_fid = torch.randperm(n_fid, generator=generator)
        train_idx = perm_fid[:n_train]
        val_idx = perm_fid[n_train:n_train + n_val]
        test_idx = perm_fid[n_train + n_val:]

        train_data.append(_subset(data_list[0], train_idx))
        val_data.append(_subset(data_list[0], val_idx))
        test_data.append(_subset(data_list[0], test_idx))

        # Split derivative data (pairs must share permutation)
        n_params = (len(data_list) - 1) // 2
        for param_idx in range(n_params):
            minus_data = data_list[1 + 2 * param_idx]
            plus_data = data_list[2 + 2 * param_idx]

            n_deriv = minus_data.shape[0]
            n_train_d = int(n_deriv * train_frac)
            n_val_d = int(n_deriv * val_frac)

            # Same permutation for both minus and plus
            perm_deriv = torch.randperm(n_deriv, generator=generator)
            train_idx = perm_deriv[:n_train_d]
            val_idx = perm_deriv[n_train_d:n_train_d + n_val_d]
            test_idx = perm_deriv[n_train_d + n_val_d:]

            train_data.append(_subset(minus_data, train_idx))
            train_data.append(_subset(plus_data, train_idx))
            val_data.append(_subset(minus_data, val_idx))
            val_data.append(_subset(plus_data, val_idx))
            test_data.append(_subset(minus_data, test_idx))
            test_data.append(_subset(plus_data, test_idx))

        return train_data, val_data, test_data

    @abstractmethod
    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from input data.

        Each pipeline type implements this differently:
        - Filtration: data → filtration → vectorization → compression
        - Compression: summaries → compression
        - Vectorization: diagrams → vectorization → compression

        Args:
            data: Input data in pipeline-specific format

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        pass

    def _compute_summaries_batched(
        self,
        data: List[torch.Tensor],
        eval_batch_size: int = 500
    ) -> List[torch.Tensor]:
        """
        Compute summaries in batches to avoid GPU OOM during evaluation.

        Default implementation: no batching, just calls _compute_summaries().

        Subclasses with expensive per-sample GPU work (e.g., filtration
        pipeline with CNN → cubical persistence) should override this with
        a proper batching strategy that preserves global consistency of:
        - Vectorization hyperparameters (TopK k auto-selection via fit())
        - Compression weights (MOPED covariance matrix)

        See LearnableFiltrationPipeline._compute_summaries_batched for the
        correct multi-phase batching approach.

        Args:
            data: Input data [fid, minus_0, plus_0, ...]
            eval_batch_size: Maximum samples per batch during eval

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        # Default: no batching. Compression and vectorization pipelines don't
        # have expensive per-sample GPU work, so batching is unnecessary.
        return self._compute_summaries(data)

    def compute_fisher_result(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor,
        check_gaussianity: bool = False,
        compute_moments: bool = False,
        eval_batch_size: int = 0
    ):
        """
        Compute Fisher result with optional Gaussianity check and moment penalties.

        This method eliminates redundancy by centralizing the Fisher computation
        and Gaussianity checking logic. Subclasses only need to implement
        _compute_summaries().

        Args:
            data: Input data
            delta_theta: Parameter step sizes
            check_gaussianity: If True, also check Gaussianity
            compute_moments: If True, compute skewness/kurtosis penalties
            eval_batch_size: If > 0, process summaries in batches to avoid OOM.
                             Only used during evaluation (not training).

        Returns:
            FisherResult with fisher_matrix, constraints, log_det_fisher, is_gaussian,
            and optionally skewness_penalty/kurtosis_penalty
        """
        # Get compressed summaries (subclass-specific implementation)
        if eval_batch_size > 0:
            compressed = self._compute_summaries_batched(data, eval_batch_size)
        else:
            compressed = self._compute_summaries(data)

        # Compute Fisher result (with optional Gaussianity check and moments)
        fisher_result = self.fisher_analyzer(
            compressed, delta_theta,
            check_gaussianity=check_gaussianity,
            compute_moments=compute_moments
        )

        return fisher_result

    def _refit_moped_cache(self, train_data, eval_batch_size, verbose=False,
                           n_samples=0):
        """
        Refit the MOPED compression matrix cache from training data.

        Computes pre-compression summaries (filtration → vectorization) then
        caches the MOPED matrix. n_samples > 0 limits the subset per set,
        which is critical for pipelines with expensive filtrations (e.g. raw
        512×512 CNN maps): n_samples=1000 costs ~4 min vs ~5 h for all data.

        Args:
            train_data: Full training data [fid, minus_0, plus_0, ...]
            eval_batch_size: Batch size for chunked forward passes
            verbose: Print a status message
            n_samples: Max samples per dataset (0 = use all)
        """
        from ...compressions.moped import MOPEDCompression
        if not isinstance(self.compression, MOPEDCompression):
            return

        was_training = self.training
        self.eval()

        # NOTE: We do NOT reset _vectorization_fitted here. The DiffCurves
        # grid must remain fixed throughout training to keep the feature space
        # stable. Re-fitting the grid would change all feature values
        # discontinuously, causing extreme gradient spikes and training crashes.
        # Only the MOPED matrix is updated to track CNN evolution.

        with torch.no_grad():
            if n_samples > 0:
                data_to_use = [
                    d[:n_samples] if hasattr(d, '__len__') and len(d) > n_samples else d
                    for d in train_data
                ]
            else:
                data_to_use = train_data
            summaries = self._compute_pre_compression_summaries(
                data_to_use, eval_batch_size
            )
            n_samp = summaries[0].shape[0]
            n_features = summaries[0].shape[1]
            self.compression.cache_compression_matrix(summaries)

        if verbose:
            grid_status = "fitted (first time)" if not getattr(self, '_pre_refit_grid_was_set', True) else "fixed (reusing epoch-0 grid)"
            subset_str = f" (subset {n_samples})" if n_samples > 0 else ""
            print(f"  MOPED cache refreshed: {n_samp} samples{subset_str}, "
                  f"{n_features} features (Hartlap df={n_samp//2 - n_features - 2}), "
                  f"grid {grid_status}")
            self._pre_refit_grid_was_set = True

        if was_training:
            self.train()

    def _renormalize_perslay(self, train_data, eval_batch_size, n_samples=0):
        """Refit PersLayLayer input_mean/input_std from the current filtration output.

        Uses fiducial training data only (train_data[0]) — the derivative sets
        have essentially the same diagram distribution at the fiducial cosmology.
        With n_samples=1000 and 512×512 maps costs ~4 min on H100 vs ~5 h for
        all training data. pers_threshold is left unchanged.
        """
        try:
            from ...vectorizations.perslay import PersLayLayer
        except ImportError:
            return

        vec = getattr(self, 'vectorization', None)
        if vec is None:
            return
        layers = getattr(vec, 'layers', [vec])
        perslay_layers = [(i, l) for i, l in enumerate(layers)
                         if isinstance(l, PersLayLayer)]
        if not perslay_layers:
            return

        was_training = self.training
        self.eval()

        with torch.no_grad():
            fid_data = train_data[0]
            if n_samples > 0 and hasattr(fid_data, '__len__') and len(fid_data) > n_samples:
                fid_data = fid_data[:n_samples]
            diagrams_list = self.compute_diagrams([fid_data], verbose=False)
            fid_diagrams = diagrams_list[0]  # list indexed by hom_dim
            for layer_idx, layer in perslay_layers:
                if layer_idx < len(fid_diagrams):
                    layer.renormalize(fid_diagrams[layer_idx])

        if was_training:
            self.train()

    def _compute_pre_compression_summaries(
        self, data, eval_batch_size
    ):
        """
        Compute summaries up to (but not including) MOPED compression.

        Returns the vectorized summaries that would normally be fed into
        self.compression(). Handles batching for memory efficiency.

        Two paths:
        - Streaming (default for DiffCurves etc.): filtration + vectorization
          are fused per chunk. Peak RAM = chunk_size × n_features × n_sets.
          Used when vectorization has no fit() or is already fitted.
        - Accumulate-then-fit (TopK, unfitted PI): all diagrams are held in
          RAM so vectorization.fit() can see the global distribution before
          transforming. Peak RAM ≈ n_samples × avg_pairs × n_sets.

        Args:
            data: Input data [fid, minus_0, plus_0, ...]
            eval_batch_size: Max samples per chunk

        Returns:
            List of summary tensors [fid, minus_0, plus_0, ...]
        """
        import gc
        n_samples = data[0].shape[0] if isinstance(data[0], (torch.Tensor,) + LAZY_TENSOR_TYPES) else len(data[0])

        if n_samples <= eval_batch_size:
            diagrams = self.compute_diagrams(data, verbose=False)
            return self.vectorize(diagrams)

        n_chunks = (n_samples + eval_batch_size - 1) // eval_batch_size

        # Decide path: only accumulate all diagrams when we need to fit.
        # If already fitted or vectorization has no fit(), stream instead.
        vec_needs_fitting = (
            hasattr(self.vectorization, 'fit')
            and not getattr(self, '_vectorization_fitted', False)
        )

        if vec_needs_fitting:
            # ── Accumulate path (TopK / unfitted PI) ──────────────────────
            # Phase 1: run filtration on all chunks, accumulate raw diagrams
            all_diagrams = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_data = [d[start:end] for d in data]
                chunk_diagrams = self.compute_diagrams(chunk_data, verbose=False)

                if all_diagrams is None:
                    all_diagrams = [
                        [[] for _ in range(len(chunk_diagrams[0]))]
                        for _ in range(len(chunk_diagrams))
                    ]
                for set_idx in range(len(chunk_diagrams)):
                    for hom_idx in range(len(chunk_diagrams[set_idx])):
                        all_diagrams[set_idx][hom_idx].extend(
                            chunk_diagrams[set_idx][hom_idx]
                        )
                del chunk_diagrams
                gc.collect()

            # Phase 2: fit vectorization once on all diagrams
            self.vectorization.fit(all_diagrams)
            self._vectorization_fitted = True

            # Phase 3: vectorize chunk by chunk
            all_summaries = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_diagrams = [
                    [hom_dgms[start:end] for hom_dgms in set_dgms]
                    for set_dgms in all_diagrams
                ]
                chunk_summaries = [self.vectorization(diags) for diags in chunk_diagrams]

                if all_summaries is None:
                    all_summaries = chunk_summaries
                else:
                    for i in range(len(all_summaries)):
                        all_summaries[i] = torch.cat([all_summaries[i], chunk_summaries[i]], dim=0)

            del all_diagrams
        else:
            # ── Streaming path (DiffCurves, already-fitted PI, etc.) ───────
            # Fuse filtration + vectorization per chunk.
            # Peak RAM: chunk_size × n_features × n_sets — avoids storing
            # O(n_samples × avg_pairs) raw diagrams in memory simultaneously.
            all_summaries = None
            for chunk_idx in range(n_chunks):
                start = chunk_idx * eval_batch_size
                end = min(start + eval_batch_size, n_samples)
                chunk_data = [d[start:end] for d in data]
                chunk_diagrams = self.compute_diagrams(chunk_data, verbose=False)
                chunk_summaries = [self.vectorization(diags) for diags in chunk_diagrams]
                del chunk_diagrams
                gc.collect()

                if all_summaries is None:
                    all_summaries = chunk_summaries
                else:
                    for i in range(len(all_summaries)):
                        all_summaries[i] = torch.cat([all_summaries[i], chunk_summaries[i]], dim=0)

        return all_summaries

    def train_model(
        self,
        train_data: List[torch.Tensor],
        val_data: List[torch.Tensor],
        delta_theta: torch.Tensor,
        training_config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Train all learnable components automatically.

        PyTorch automatically tracks all parameters in the pipeline,
        so we just use self.parameters() to train everything that's learnable.

        Args:
            train_data: Training data [fid, minus_0, plus_0, ...]
            val_data: Validation data with same structure
            delta_theta: Parameter step sizes for Fisher computation
            training_config: Training hyperparameters

        Returns:
            Training history with loss curves, best epoch, etc.
        """
        # Two-phase curriculum: freeze filtration for phase 1 so PersLay + MOPED
        # converge on the near-identity filtration before CNN starts changing diagrams.
        freeze_cnn_epochs = training_config.freeze_cnn_epochs
        if freeze_cnn_epochs > 0 and hasattr(self, 'filtration'):
            for p in self.filtration.parameters():
                p.requires_grad = False

        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found in pipeline. "
                "Ensure at least one component (filtration, vectorization, or compression) "
                "has trainable parameters."
            )

        if training_config.verbose:
            try:
                n_params = sum(p.numel() for p in trainable_params)
                if freeze_cnn_epochs > 0 and hasattr(self, 'filtration'):
                    n_frozen = sum(p.numel() for p in self.filtration.parameters())
                    n_total = n_params + n_frozen
                    print(f"  Phase 1 ({freeze_cnn_epochs} epochs): {n_params} trainable params "
                          f"| CNN frozen ({n_frozen} params)")
                    print(f"  Phase 2 ({training_config.n_epochs - freeze_cnn_epochs} epochs): "
                          f"CNN unfreezes at epoch {freeze_cnn_epochs} "
                          f"({n_total} total params, LR reset to {training_config.lr:.2e})")
                else:
                    print(f"  Training {n_params} parameters")
            except (ValueError, RuntimeError):
                print(f"  Training parameters (lazy initialization - will be determined on first forward pass)")

        # Build optimizer with only currently trainable params.
        # Frozen filtration params are added via add_param_group at phase transition.
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=training_config.lr,
            weight_decay=training_config.weight_decay
        )
        
        # Setup learning rate scheduler if configured
        scheduler = None
        scheduler_is_epoch_based = False  # True for cosine schedulers (step every epoch)
        if training_config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=training_config.lr_factor,
                patience=training_config.lr_patience,
                min_lr=training_config.lr_min
            )
            if training_config.verbose:
                print(f"  Using ReduceLROnPlateau scheduler (patience={training_config.lr_patience}, factor={training_config.lr_factor})")
        elif training_config.lr_scheduler == "cosine_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=training_config.lr_T0,
                T_mult=training_config.lr_T_mult,
                eta_min=training_config.lr_min
            )
            scheduler_is_epoch_based = True
            if training_config.verbose:
                print(f"  Using CosineAnnealingWarmRestarts scheduler "
                      f"(T_0={training_config.lr_T0}, T_mult={training_config.lr_T_mult}, "
                      f"eta_min={training_config.lr_min})")
        elif training_config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config.n_epochs,
                eta_min=training_config.lr_min
            )
            scheduler_is_epoch_based = True
            if training_config.verbose:
                print(f"  Using CosineAnnealingLR scheduler "
                      f"(T_max={training_config.n_epochs}, eta_min={training_config.lr_min})")

        # Training state
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')    # Track best loss among saved (Gaussian) models
        best_model_state = None
        best_epoch = 0
        epochs_no_improve = 0           # Validation checks since last meaningful improvement
        _min_delta = getattr(training_config, 'min_delta', 0.0) or 0.0
        _early_stop_threshold = float('inf')  # best_val_loss - min_delta

        # Batch size handling
        min_train_size = min(
            s.shape[0] if isinstance(s, (torch.Tensor,) + LAZY_TENSOR_TYPES) else len(s)
            for s in train_data
        )
        batch_size = min(training_config.batch_size, min_train_size)
        # No-grad paths (pre-fit / val / test) can handle much larger chunks than
        # training does — the backward pass activations dominate GPU memory, so
        # we decouple eval chunking from the (small) training batch_size.
        eval_chunk_size = max(batch_size, 500)
        use_moment_penalty = (training_config.lambda_s > 0) or (training_config.lambda_k > 0)

        # Optional prewarm for topology-heavy filtrations to avoid repeated
        # simplex+edge construction during sampled training batches.
        # Prewarm ALL training data so forward passes are pure cache hits.
        if self._supports_topology_prewarm():
            cache_size = getattr(self.filtration, 'topology_cache_size', 50000)
            total_train = sum(
                int(d.shape[0]) for d in train_data
                if isinstance(d, (torch.Tensor,) + LAZY_TENSOR_TYPES) and d.ndim >= 3
            )
            # Ensure cache is large enough for all training data
            if total_train > cache_size:
                self.filtration.topology_cache_size = total_train + 1000
                if training_config.verbose:
                    print(f"  Auto-expanded topology cache: {cache_size} → {self.filtration.topology_cache_size} "
                          f"(to fit {total_train} training samples)")

            # Prewarm with ALL training data (not just a subset)
            prewarm_data = [d for d in train_data
                           if isinstance(d, (torch.Tensor,) + LAZY_TENSOR_TYPES) and d.ndim >= 3]
            self._prewarm_topology_cache(prewarm_data)
            if training_config.verbose:
                stats = self.filtration.get_cache_stats()
                prewarm_count = sum(int(d.shape[0]) for d in prewarm_data)
                print(
                    f"  Topology cache prewarmed: size={stats['size']} "
                    f"(max={stats['max_size']}), prewarm_samples={prewarm_count}"
                )

        # Pre-fit vectorization grid from ALL training data.
        # This ensures the DiffCurves/TopK grid is representative (17,500 samples)
        # rather than fitted from the first small batch (200 samples).
        # The grid remains fixed for the entire training run.
        if (hasattr(self, 'vectorization') and hasattr(self.vectorization, 'fit')
                and not getattr(self, '_vectorization_fitted', False)):
            if training_config.verbose:
                print("  Pre-fitting vectorization grid from full training data...")
            self.eval()
            with torch.no_grad():
                self._compute_pre_compression_summaries(train_data, eval_chunk_size)
            self.train()
            if training_config.verbose:
                print(f"  Vectorization grid fitted and locked (vectorization_fitted={self._vectorization_fitted})")

        for epoch in range(training_config.n_epochs):
            # Phase 2 transition: unfreeze CNN and add it to the optimizer.
            if freeze_cnn_epochs > 0 and epoch == freeze_cnn_epochs and hasattr(self, 'filtration'):
                for p in self.filtration.parameters():
                    p.requires_grad = True
                optimizer.add_param_group({
                    'params': [p for p in self.filtration.parameters()],
                    'lr': training_config.lr,
                    'weight_decay': training_config.weight_decay,
                })
                # Reset all LRs so CNN starts at full LR alongside refreshed PersLay.
                for pg in optimizer.param_groups:
                    pg['lr'] = training_config.lr
                # Fresh scheduler so plateau patience resets for phase 2.
                if scheduler is not None and not scheduler_is_epoch_based:
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        factor=training_config.lr_factor,
                        patience=training_config.lr_patience,
                        min_lr=training_config.lr_min,
                    )
                # Reset early-stop counter: CNN unfreeze creates a new optimization
                # landscape — patience from phase 1 should not carry over.
                epochs_no_improve = 0
                _early_stop_threshold = best_val_loss - _min_delta
                if training_config.verbose:
                    n_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
                    print(f"\n  === Phase 2: CNN unfrozen at epoch {epoch} "
                          f"({n_total} params total, LR reset to {training_config.lr:.2e}) ===\n")

            # Periodic MOPED refit: recompute compression matrix from full training
            # data to avoid ill-conditioned covariance from small batches.
            # epoch==0 gives the initial cache; subsequent refits track CNN drift.
            if (training_config.moped_refit_every > 0
                    and epoch % training_config.moped_refit_every == 0):
                self._refit_moped_cache(
                    train_data, eval_chunk_size,
                    verbose=(training_config.verbose and epoch == 0),
                    n_samples=training_config.moped_refit_n_samples,
                )

            if (training_config.renormalize_every > 0
                    and epoch > 0
                    and epoch % training_config.renormalize_every == 0):
                self._renormalize_perslay(
                    train_data, eval_chunk_size,
                    n_samples=training_config.renormalize_n_samples,
                )

            # Linear warmup: override LR during warmup phase
            if training_config.lr_warmup > 0 and epoch < training_config.lr_warmup:
                warmup_lr = training_config.lr * (epoch + 1) / training_config.lr_warmup
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

            # Training step
            self.train()  # Set entire pipeline to train mode

            # Sample random batch
            idx = torch.randperm(min_train_size)[:batch_size]
            batch_data = self._extract_batch(train_data, idx)

            # Forward pass: compute Fisher result and extract loss
            fisher_result = self.compute_fisher_result(
                batch_data, delta_theta,
                check_gaussianity=False,
                compute_moments=use_moment_penalty
            )
            loss = -fisher_result.log_det_fisher

            # Clamp loss to prevent extreme slogdet gradient spikes when
            # Fisher matrix is near-singular (early training / topology changes).
            if training_config.loss_clamp is not None:
                loss = loss.clamp(min=-training_config.loss_clamp,
                                  max=training_config.loss_clamp)

            # Add Gaussianity regularization if enabled, with optional linear
            # annealing from the configured value to 0 over lambda_anneal_epochs.
            _anneal = training_config.lambda_anneal_epochs
            if _anneal > 0:
                lambda_scale = max(0.0, 1.0 - epoch / _anneal)
            else:
                lambda_scale = 1.0
            eff_lambda_s = training_config.lambda_s * lambda_scale
            eff_lambda_k = training_config.lambda_k * lambda_scale
            if eff_lambda_s > 0:
                loss = loss + eff_lambda_s * fisher_result.skewness_penalty
            if eff_lambda_k > 0:
                loss = loss + eff_lambda_k * fisher_result.kurtosis_penalty

            # Guard: skip backward if loss is NaN/Inf or has no gradient graph.
            # NaN loss WITH grad_fn is especially dangerous: backward() would
            # propagate NaN gradients into ALL parameters (including learnable
            # bandwidth σ), making them permanently NaN with no recovery.
            loss_is_bad = (
                torch.isnan(loss) or torch.isinf(loss) or loss.grad_fn is None
            )

            if loss_is_bad:
                train_losses.append(float('nan'))
                n_bad_epochs = getattr(self, '_n_bad_epochs', 0) + 1
                self._n_bad_epochs = n_bad_epochs

                # Recovery: after nan_recovery_patience consecutive bad epochs,
                # reload best model, clear optimizer momentum, and inject small
                # noise to escape the NaN-inducing topology region.
                _patience = training_config.nan_recovery_patience
                if n_bad_epochs >= _patience and n_bad_epochs % _patience == 0 and best_model_state is not None:
                    self.load_state_dict(best_model_state)
                    optimizer.state.clear()
                    # Inject small noise to escape NaN-inducing topology region.
                    # Without noise, the same weights produce the same topology
                    # → same NaN → permanent cascade.
                    _noise_std = training_config.nan_recovery_noise
                    if _noise_std > 0:
                        with torch.no_grad():
                            for p in self.parameters():
                                p.add_(torch.randn_like(p) * _noise_std)
                    # Halve LR to stabilise after recovery
                    for pg in optimizer.param_groups:
                        pg['lr'] = max(pg['lr'] * 0.5, training_config.lr_min)
                    import warnings
                    warnings.warn(
                        f"Epoch {epoch}: {n_bad_epochs} consecutive NaN/bad epochs. "
                        f"Reloading best model (epoch {best_epoch}) + noise(σ={_noise_std:.0e}), "
                        f"LR → {optimizer.param_groups[0]['lr']:.2e}",
                        RuntimeWarning,
                    )

                if n_bad_epochs >= training_config.nan_abort_threshold:
                    raise RuntimeError(
                        f"Training aborted: {n_bad_epochs} consecutive NaN/degenerate "
                        f"epochs. Pipeline cannot recover."
                    )
                continue
            else:
                self._n_bad_epochs = 0  # Reset counter on successful epoch

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm before clipping (for monitoring)
            grad_norm_pre = torch.nn.utils.clip_grad_norm_(
                self.parameters(), float('inf')
            ).item() if training_config.verbose else 0.0

            # Apply gradient clipping if configured
            if training_config.grad_clip is not None and training_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), training_config.grad_clip)

            # Extra safety: check for NaN/Inf in gradients before stepping
            has_nan_grad = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in self.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()  # Discard bad gradients
                train_losses.append(float('nan'))
                n_bad_epochs = getattr(self, '_n_bad_epochs', 0) + 1
                self._n_bad_epochs = n_bad_epochs
                continue

            # Log extreme gradient norms (>10× clip threshold)
            if (training_config.verbose and training_config.grad_clip
                    and grad_norm_pre > 10 * training_config.grad_clip
                    and epoch % 100 == 0):
                import warnings
                warnings.warn(
                    f"Epoch {epoch}: extreme grad norm {grad_norm_pre:.1f} "
                    f"(clipped to {training_config.grad_clip})",
                    RuntimeWarning,
                )

            optimizer.step()

            # Step epoch-based schedulers (cosine, cosine_restarts) after each optimizer step
            # Skip during warmup phase to avoid conflicting LR updates
            if scheduler_is_epoch_based and scheduler is not None:
                if training_config.lr_warmup == 0 or epoch >= training_config.lr_warmup:
                    scheduler.step()

            train_losses.append(loss.item())

            # Validation
            if (epoch + 1) % training_config.validate_every == 0:
                self.eval()  # Set entire pipeline to eval mode
                with torch.no_grad():
                    val_result = self.compute_fisher_result(
                        val_data, delta_theta,
                        check_gaussianity=True,
                        eval_batch_size=eval_chunk_size
                    )
                    # Note: val_loss is pure -log|F|, does NOT include regularization terms
                    # (lambda_k * kurtosis + lambda_s * skewness). This ensures we select
                    # models based on Fisher information, not regularization strength.
                    val_loss = -val_result.log_det_fisher

                val_losses.append(val_loss.item())
                
                # Step the LR scheduler based on validation loss (plateau only)
                # Cosine schedulers are stepped per-epoch above, not at validation time
                if scheduler is not None and not scheduler_is_epoch_based:
                    scheduler.step(val_loss)

                # Check validation conditions
                # Note: is_best_loss compares against best_val_loss, which is only
                # updated when Gaussianity passes. So if Gaussianity never passes,
                # best_val_loss stays at infinity and is_best_loss is always True.
                is_best_loss = val_loss.item() < best_val_loss
                is_gaussian = val_result.is_gaussian

                # Save best model (both conditions must pass)
                improved_meaningfully = val_loss.item() < _early_stop_threshold
                if is_best_loss and is_gaussian:
                    best_val_loss = val_loss.item()
                    _early_stop_threshold = best_val_loss - _min_delta
                    best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(self.state_dict())
                    if improved_meaningfully:
                        epochs_no_improve = 0

                    if training_config.verbose:
                        current_lr = optimizer.param_groups[0]['lr']
                        bw_info = self._get_bandwidth_info()
                        print(f"  Epoch {epoch+1}/{training_config.n_epochs}: "
                              f"Train Loss={loss.item():.3f}, "
                              f"Val Loss={val_loss.item():.3f}, "
                              f"LR={current_lr:.2e}{bw_info}, "
                              f"Best Loss: ✓, Gaussianity: ✓ → Model updated")
                else:
                    epochs_no_improve += 1
                    if training_config.verbose:
                        best_loss_mark = "✓" if is_best_loss else "✗"
                        gaussian_mark = "✓" if is_gaussian else "✗"
                        current_lr = optimizer.param_groups[0]['lr']
                        bw_info = self._get_bandwidth_info()
                        print(f"  Epoch {epoch+1}/{training_config.n_epochs}: "
                              f"Train Loss={loss.item():.3f}, "
                              f"Val Loss={val_loss.item():.3f}, "
                              f"LR={current_lr:.2e}{bw_info}, "
                              f"Best Loss: {best_loss_mark}, Gaussianity: {gaussian_mark}")

                if (training_config.patience is not None
                        and epochs_no_improve >= training_config.patience):
                    if training_config.verbose:
                        print(f"\n  Early stopping at epoch {epoch+1}: "
                              f"no improvement for {epochs_no_improve} validation checks "
                              f"(patience={training_config.patience})")
                    break

        # Load best model (entire pipeline)
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            if training_config.verbose:
                print(f"\n  Loaded best model from epoch {best_epoch} "
                      f"(val loss: {best_val_loss:.3f})")

        # Clear MOPED cache so evaluation uses the standard (split) path
        if training_config.moped_refit_every > 0:
            from ...compressions.moped import MOPEDCompression
            if isinstance(self.compression, MOPEDCompression):
                self.compression.clear_compression_cache()

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch
        }

    def _extract_batch(self, data_list: List, indices: torch.Tensor) -> List:
        """
        Extract batch from data list.

        Handles both tensor and list types.

        Args:
            data_list: List of data elements
            indices: Batch indices

        Returns:
            Batched data maintaining structure
        """
        batch = []
        for data in data_list:
            if isinstance(data, (torch.Tensor,) + LAZY_TENSOR_TYPES):
                batch.append(data[indices])
            elif isinstance(data, list):
                # Handle nested lists (e.g., persistence diagrams)
                batch.append([data[i] for i in indices])
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
        return batch

    def evaluate(
        self,
        data: List[torch.Tensor],
        delta_theta: torch.Tensor,
        eval_batch_size: int = 0
    ) -> FisherResult:
        """
        Evaluate the pipeline on a dataset.

        Always checks Gaussianity on test data for final verification.
        Processes all data at once to ensure consistent vectorization
        hyperparameters (TopK k) and compression weights (MOPED) across
        all samples.

        Args:
            data: Data to evaluate [fid, minus_0, plus_0, ...]
            delta_theta: Parameter step sizes
            eval_batch_size: If > 0, process summaries in batches to avoid OOM.

        Returns:
            Fisher result with matrix, constraints, log determinant, and is_gaussian
        """
        return self.compute_fisher_result(
            data, delta_theta,
            check_gaussianity=True,
            eval_batch_size=eval_batch_size
        )

    def run(
        self,
        config: AnalysisConfig,
        training_config: TrainingConfig,
        data: List[torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Run the complete learnable pipeline.

        Steps:
        1. Split data into train/val/test
        2. Train component (uses evaluate() on val during training)
        3. Evaluate on test set

        Args:
            config: Pipeline configuration
            training_config: Training configuration
            data: Pre-generated data to use [fid, minus_0, plus_0, ...]

        Returns:
            Dictionary with test result and training history
        """
        # 1. Split data
        if training_config.verbose:
            print("\n" + "="*80)
            print("Splitting Data")
            print("="*80)

        train_data, val_data, test_data = self.split_data(
            data,
            train_frac=training_config.train_frac,
            val_frac=training_config.val_frac,
            seed=42  # Fixed seed for reproducibility
        )

        # Free the original data list to release memory.
        # For mmap-backed tensors (CNN+GAP) this reclaims virtual mappings;
        # for plain tensors (GAP+TopK torch.cat output) this reclaims the
        # anonymous RAM that would otherwise stay pinned during training.
        import gc
        del data
        gc.collect()

        # Track split sizes (needed for return value)
        n_train = train_data[0].shape[0]
        n_val = val_data[0].shape[0]
        n_test = test_data[0].shape[0]

        if training_config.verbose:
            print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # 2. Train (internally uses evaluate() on validation)
        if training_config.verbose:
            print("\n" + "="*80)
            print("Training Component")
            print("="*80)

        history = self.train_model(
            train_data=train_data,
            val_data=val_data,
            delta_theta=config.delta_theta,
            training_config=training_config
        )

        # 3. Evaluate on test set
        if training_config.verbose:
            print("\n" + "="*80)
            print("Evaluating on Test Set")
            print("="*80)

        self.eval()
        with torch.no_grad():
            test_result = self.evaluate(test_data, config.delta_theta,
                                        eval_batch_size=max(training_config.batch_size, 500))

        if training_config.verbose:
            print(f"  Test log|F|: {test_result.log_det_fisher:.3f}")
            print(f"  Test constraints: {test_result.constraints.detach().cpu().numpy()}")

        return {
            'test_result': test_result,
            'training_history': history,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test
        }