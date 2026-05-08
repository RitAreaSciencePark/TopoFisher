"""Pipeline for training learnable compression."""

import os
from typing import List, Optional
import torch
import torch.nn as nn

from .base import LearnablePipeline


class LearnableCompressionPipeline(LearnablePipeline):
    """
    Pipeline for training compression from summaries.

    This pipeline takes vectorized summaries and trains a compression
    component to maximize the Fisher information determinant.
    PyTorch automatically tracks and trains any learnable parameters.
    """

    def _compute_summaries(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute compressed summaries from vectorized summaries.

        Pipeline: summaries → compression

        Args:
            data: Vectorized summaries [fid, minus_0, plus_0, ...]

        Returns:
            List of compressed summary tensors [fid, minus_0, plus_0, ...]
        """
        compressed = self.compression(data)
        return compressed

    def _compute_pre_compression_summaries(self, data, eval_batch_size):
        """
        Return data as-is: for LearnableCompressionPipeline, the data passed
        to train_model() is already vectorized summaries from generate_data().
        Re-running filtration + vectorization would fail (cubical on 2D features)
        and is unnecessary.
        """
        return data

    def _get_data_file_paths(self, config):
        """
        Build ordered list of data file paths matching generate_data order.

        Returns:
            List of absolute paths: [fiducial.pt, minus_p0.pt, plus_p0.pt, ...]
        """
        import os
        data_dir = config.data_dir
        n_params = len(config.delta_theta)
        fnames = ['fiducial.pt']
        for i in range(n_params):
            fnames.append(f'minus_p{i}.pt')
            fnames.append(f'plus_p{i}.pt')
        return [os.path.join(data_dir, f) for f in fnames]

    def _get_per_bin_file_paths(self, config, n_bins):
        """
        Build per-bin file paths: file_paths[set_idx][bin_idx] = path.

        Per-bin files use naming: fiducial_bin0.pt, fiducial_bin1.pt, etc.
        Returns None if per-bin files don't exist.
        """
        import os
        data_dir = config.data_dir
        n_params = len(config.delta_theta)
        base_names = ['fiducial']
        for i in range(n_params):
            base_names.append(f'minus_p{i}')
            base_names.append(f'plus_p{i}')

        paths = []
        for base in base_names:
            bin_paths = []
            for bin_idx in range(n_bins):
                p = os.path.join(data_dir, f'{base}_bin{bin_idx}.pt')
                if not os.path.isfile(p):
                    return None  # per-bin files not available
                bin_paths.append(p)
            paths.append(bin_paths)
        return paths

    def _load_bin_data(self, per_bin_paths, set_idx, bin_idx, mmap=True,
                       file_paths=None):
        """
        Load one bin's data for one dataset using memory-mapping.
        The returned tensor is mmap-backed; callers should .clone() slices
        to avoid keeping the mmap alive and to get contiguous memory.

        Args:
            per_bin_paths: paths[set_idx][bin_idx] from _get_per_bin_file_paths
            set_idx: dataset index
            bin_idx: tomographic bin index
            mmap: use memory-mapping (default True)
            file_paths: optional full-file paths for fallback loading

        Returns:
            Tensor of shape (n_samples, H, W) — mmap-backed, on CPU.
        """
        if per_bin_paths is not None:
            # Memory-mapped load of per-bin file (~5 GB virtual, ~0 RSS)
            return torch.load(per_bin_paths[set_idx][bin_idx],
                              map_location='cpu', weights_only=True,
                              mmap=True)
        elif file_paths is not None:
            # Fallback: mmap full file, extract bin slice
            tensor = torch.load(file_paths[set_idx], map_location='cpu',
                                weights_only=True, mmap=True)
            bin_data = tensor[:, bin_idx, :, :]
            del tensor
            return bin_data
        else:
            raise ValueError("Either per_bin_paths or file_paths must be provided")

    def _summaries_cache_paths(self, config, n_sets: int) -> List[str]:
        bin_idx_only = getattr(config, 'bin_idx', None)
        suffix = f'_bin{bin_idx_only}' if bin_idx_only is not None else ''
        return [os.path.join(config.summaries_dir, f'summaries_set{i}{suffix}.pt')
                for i in range(n_sets)]

    def _try_load_summaries_cache(self, config) -> Optional[List[torch.Tensor]]:
        if config.summaries_dir is None:
            return None
        n_sets = 1 + 2 * len(config.delta_theta)
        paths = self._summaries_cache_paths(config, n_sets)
        if not all(os.path.isfile(p) for p in paths):
            return None
        print(f"  Loading pre-computed summaries from {config.summaries_dir}", flush=True)
        summaries = [torch.load(p, map_location=self.device, weights_only=True)
                     for p in paths]
        self._vectorization_fitted = True
        return summaries

    def _save_summaries_cache(self, summaries: List[torch.Tensor], config) -> None:
        if config.summaries_dir is None:
            return
        os.makedirs(config.summaries_dir, exist_ok=True)
        for i, s in enumerate(summaries):
            final = os.path.join(config.summaries_dir, f'summaries_set{i}.pt')
            tmp = final + '.tmp'
            torch.save(s.cpu(), tmp)
            os.replace(tmp, final)
        print(f"  Saved {len(summaries)} summary tensors to {config.summaries_dir}",
              flush=True)

    def generate_data(self, config):
        """
        Generate vectorized summaries for compression training.

        Overrides parent to return vectorized summaries instead of raw data,
        ensuring compression trains on topological features, not raw fields.

        Memory-efficient: uses lazy data loading + per-bin + per-chunk processing.
        For multi-bin input (n_samples, n_bins, H, W), each dataset is loaded
        from disk one bin at a time. Each bin is further sub-chunked (default
        500 samples) so only ~500 maps are in memory during persistence.

        Peak memory: ~500 maps (0.5 GB) + GPU intermediates (~1 GB) + diagrams
        for 500 samples (~0.6 GB) ≈ 2 GB.

        Args:
            config: AnalysisConfig with parameters

        Returns:
            List of vectorized summaries [fid, minus_0, plus_0, ...]
        """
        import time
        import gc

        # Fast path: load pre-computed summaries if available
        cached = self._try_load_summaries_cache(config)
        if cached is not None:
            return cached

        CHUNK_SIZE = 500  # samples per GPU persistence call

        is_tda = self._is_tda_filtration()
        n_hom = len(self.filtration.dimensions) if hasattr(self.filtration, 'dimensions') else 2

        # Check if we have pre-generated multi-bin data on disk
        if config.data_dir is not None:
            file_paths = self._get_data_file_paths(config)
            has_pregenerated = all(os.path.isfile(p) for p in file_paths)
        else:
            file_paths = []
            has_pregenerated = False

        # Also check for per-bin-only files (no monolithic .pt files)
        # This happens when data was generated with generate_and_save_chunked()
        per_bin_only = False
        if not has_pregenerated and config.data_dir is not None:
            # Try to detect per-bin files without knowing n_bins yet
            # Check if fiducial_bin0.pt exists
            test_path = os.path.join(config.data_dir, 'fiducial_bin0.pt')
            if os.path.isfile(test_path):
                # Count bins
                _n_bins = 0
                while os.path.isfile(os.path.join(config.data_dir, f'fiducial_bin{_n_bins}.pt')):
                    _n_bins += 1
                # Verify all datasets have the same bins
                _per_bin_paths = self._get_per_bin_file_paths(config, _n_bins)
                if _per_bin_paths is not None:
                    per_bin_only = True
                    has_pregenerated = True
                    n_bins = _n_bins
                    has_bins = True
                    # Read shape from first per-bin file
                    first_tensor = torch.load(_per_bin_paths[0][0], map_location='cpu',
                                              weights_only=True, mmap=True)
                    n_samples = first_tensor.shape[0]
                    shape_str = f'{n_bins}x' + 'x'.join(str(s) for s in first_tensor.shape[1:])
                    del first_tensor

        if has_pregenerated and not per_bin_only:
            # Peek at first file to detect shape without loading all data
            # mmap=True: memory-maps the file, only metadata is read (no RAM used)
            first_tensor = torch.load(file_paths[0], map_location='cpu',
                                      weights_only=True, mmap=True)
            has_bins = first_tensor.ndim == 4
            n_bins = first_tensor.shape[1] if has_bins else 1
            n_samples = first_tensor.shape[0]
            shape_str = 'x'.join(str(s) for s in first_tensor.shape[1:])
            del first_tensor
        elif not per_bin_only:
            has_bins = False
            n_bins = 1

        # Check for per-bin files (preferred: direct load, ~5 GB each)
        per_bin_paths = None
        if has_bins and n_bins > 1 and has_pregenerated:
            if per_bin_only:
                per_bin_paths = _per_bin_paths
                load_mode = "per-bin files (no monolithic)"
            else:
                per_bin_paths = self._get_per_bin_file_paths(config, n_bins)
                if per_bin_paths is not None:
                    load_mode = "per-bin files"
                else:
                    load_mode = "mmap (full files)"

        n_sets = len(file_paths)

        # When config.bin_idx is set, process only that single bin.
        # When not set, process all bins and concatenate features.
        bin_idx_only = getattr(config, 'bin_idx', None)

        if has_bins and n_bins > 1 and has_pregenerated:
            # === Two-pass per-bin processing ===
            # Pass 1: compute persistence per dataset, gather min/max for PI fit.
            # Pass 2: recompute persistence per dataset, vectorize with fitted PI.
            # Only 1 bin's data (~5 GB) + 1 dataset's diagrams (~6 GB) in RAM.
            # Trades 2× persistence compute for minimal memory.

            bins_to_process = [bin_idx_only] if bin_idx_only is not None else list(range(n_bins))

            t0_total = time.time()

            if is_tda:
                print(f"  Computing persistence for {n_sets} datasets "
                      f"({n_sets * n_samples} total samples, {shape_str})",
                      flush=True)
                print(f"  Processing {len(bins_to_process)} bins {bins_to_process} "
                      f"({n_hom} hom dims each)",
                      flush=True)
                print(f"  Loading: {load_mode}, chunk_size={CHUNK_SIZE}", flush=True)

            per_set_features = [[] for _ in range(n_sets)]

            # Create temp directory for caching persistence diagrams between passes.
            # This avoids recomputing persistence in pass 2 (2× speedup).
            import tempfile
            cache_dir = tempfile.mkdtemp(prefix='topofisher_dgm_cache_')
            if is_tda:
                print(f"  Diagram cache: {cache_dir}", flush=True)

            # Determine how to fit the vectorization (if at all).
            # Layers in a CombinedVectorization are SHARED across bins — there is
            # no per-bin layer duplication. The old layer_offset=bin*n_hom logic
            # was wrong for any pipeline that doesn't replicate layers per bin.
            #
            # Three fitting modes for layers inside a CombinedVectorization:
            #   streaming  — layer has partial_fit (DifferentiablePersistenceImageLayer)
            #   batch      — layer has fit() but not partial_fit (PersistenceImageLayer)
            #   none       — layer has neither (DifferentiablePersistenceCurves, etc.)
            #
            # For a bare (non-combined) vectorization the same three modes apply.
            #
            # Fitting must happen BEFORE Pass 2.  For batch-fit layers we must see
            # ALL bins first, so Pass 1 runs for all bins, then we fit, then Pass 2
            # runs for all bins.  For streaming layers finalize_fit is called once
            # after all Pass-1 bins.

            # Collect per-layer mode info for combined vectorizations
            if hasattr(self.vectorization, 'layers'):
                vec_layers = list(self.vectorization.layers)
                layer_modes = []
                for layer in vec_layers:
                    if hasattr(layer, 'partial_fit'):
                        layer_modes.append('streaming')
                    elif hasattr(layer, 'fit'):
                        layer_modes.append('batch')
                    else:
                        layer_modes.append('none')
                any_batch_layer = 'batch' in layer_modes
                # Buffer per-layer: list of diagrams for each layer_idx
                layer_fit_buffers = [[] for _ in vec_layers]
            else:
                vec_layers = None
                layer_modes = None
                any_batch_layer = False
                # For bare vectorization
                if hasattr(self.vectorization, 'partial_fit'):
                    bare_mode = 'streaming'
                elif hasattr(self.vectorization, 'fit'):
                    bare_mode = 'batch'
                else:
                    bare_mode = 'none'
                bare_fit_buffer = []  # used only for batch mode

            # ── Pass 1: compute persistence, stream-fit layers, cache ──
            t0_pass1 = time.time()
            for bin_idx in bins_to_process:
                if is_tda:
                    print(f"  Bin {bin_idx}/{n_bins-1} pass 1:", flush=True)

                for set_idx in range(n_sets):
                    bin_data = self._load_bin_data(
                        per_bin_paths, set_idx, bin_idx,
                        file_paths=file_paths)
                    n_total = bin_data.shape[0]
                    chunk_idx = 0
                    for chunk_start in range(0, n_total, CHUNK_SIZE):
                        chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
                        chunk = bin_data[chunk_start:chunk_end].clone().to(self.device)
                        with torch.no_grad():
                            diagrams = self.filtration(chunk)
                        del chunk

                        if vec_layers is not None:
                            # Combined vectorization: fit each layer with its hom-dim slice.
                            # layer i corresponds to diagrams[i] (by CombinedVectorization
                            # convention); skip if i >= n_hom (more layers than hom dims).
                            for layer_idx, (layer, mode) in enumerate(
                                    zip(vec_layers, layer_modes)):
                                if mode == 'streaming' and layer_idx < len(diagrams):
                                    layer.partial_fit(diagrams[layer_idx])
                                elif mode == 'batch' and set_idx == 0:
                                    if layer_idx < len(diagrams):
                                        layer_fit_buffers[layer_idx].extend(
                                            diagrams[layer_idx])
                        else:
                            # Bare vectorization
                            if bare_mode == 'streaming':
                                self.vectorization.partial_fit(diagrams)
                            elif bare_mode == 'batch' and set_idx == 0:
                                # Accumulate fiducial diagrams for all hom dims
                                bare_fit_buffer.append(diagrams)

                        cache_path = os.path.join(
                            cache_dir,
                            f'dgm_bin{bin_idx}_set{set_idx}_chunk{chunk_idx}.pt')
                        torch.save(diagrams, cache_path)

                        del diagrams
                        gc.collect()
                        chunk_idx += 1
                    del bin_data
                    gc.collect()

            if is_tda:
                print(f"  pass 1 total: {time.time()-t0_pass1:.1f}s", flush=True)

            # ── Finalize / batch fit (once, after all bins) ──
            if vec_layers is not None:
                for layer, mode, buf in zip(vec_layers, layer_modes, layer_fit_buffers):
                    if mode == 'streaming' and hasattr(layer, 'finalize_fit'):
                        layer.finalize_fit()
                    elif mode == 'batch' and buf:
                        layer.fit(buf)
                del layer_fit_buffers
            else:
                if bare_mode == 'streaming' and hasattr(self.vectorization, 'finalize_fit'):
                    self.vectorization.finalize_fit()
                elif bare_mode == 'batch' and bare_fit_buffer:
                    # bare_fit_buffer is list of diagram tuples [diagrams_H0, diagrams_H1, ...]
                    all_diagrams_by_hom = [[] for _ in range(n_hom)]
                    for dgm_tuple in bare_fit_buffer:
                        for hom_idx in range(n_hom):
                            all_diagrams_by_hom[hom_idx].extend(dgm_tuple[hom_idx])
                    self.vectorization.fit([all_diagrams_by_hom])
                    del bare_fit_buffer, all_diagrams_by_hom
            gc.collect()

            # ── Pass 2: load cached diagrams, vectorize ──
            # Always delegate to self.vectorization(diagrams) — the combined
            # container (or bare layer) handles routing to its sub-layers.
            t0_pass2 = time.time()
            for bin_idx in bins_to_process:
                if is_tda:
                    print(f"  Bin {bin_idx}/{n_bins-1} pass 2:", flush=True)

                t0_vec = time.time()
                for set_idx in range(n_sets):
                    n_total = n_samples
                    set_bin_features_chunks = []
                    chunk_idx = 0
                    for chunk_start in range(0, n_total, CHUNK_SIZE):
                        cache_path = os.path.join(
                            cache_dir,
                            f'dgm_bin{bin_idx}_set{set_idx}_chunk{chunk_idx}.pt')
                        diagrams = torch.load(cache_path, map_location='cpu',
                                              weights_only=False)

                        chunk_features = self.vectorization(diagrams)
                        set_bin_features_chunks.append(chunk_features)
                        del diagrams
                        gc.collect()
                        chunk_idx += 1

                    bin_features = torch.cat(set_bin_features_chunks, dim=0)
                    per_set_features[set_idx].append(bin_features)
                    del set_bin_features_chunks
                    gc.collect()

                # Clean up cache files for this bin
                for set_idx_cleanup in range(n_sets):
                    ci = 0
                    while True:
                        cp = os.path.join(
                            cache_dir,
                            f'dgm_bin{bin_idx}_set{set_idx_cleanup}_chunk{ci}.pt')
                        if os.path.isfile(cp):
                            os.remove(cp)
                            ci += 1
                        else:
                            break

                if is_tda:
                    print(f"    vectorize: {time.time()-t0_vec:.1f}s", flush=True)

            if is_tda:
                print(f"  pass 2 total: {time.time()-t0_pass2:.1f}s", flush=True)

            # Clean up cache directory
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

            # Concatenate per-bin features into final summaries
            all_summaries = []
            for set_idx in range(n_sets):
                summary = torch.cat(per_set_features[set_idx], dim=-1)
                all_summaries.append(summary.to(self.device))
            del per_set_features

            # Mark vectorization as fitted so train_model() doesn't try to
            # re-fit via _compute_pre_compression_summaries (which would
            # incorrectly run CubicalLayer on already-vectorized features).
            self._vectorization_fitted = True

            elapsed_total = time.time() - t0_total
            if is_tda:
                feat_dim = all_summaries[0].shape[-1]
                print(f"  Done: {feat_dim}-dim features, {elapsed_total:.1f}s total",
                      flush=True)

            self._save_summaries_cache(all_summaries, config)
            return all_summaries

        else:
            # === Fallback: load all data, then process ===
            # Used for single-bin data, non-TDA, or simulator-generated data.
            all_data = super().generate_data(config)

            if not is_tda:
                return all_data

            n_sets = len(all_data)

            if is_tda:
                total_samples = sum(d.shape[0] for d in all_data)
                shape_str = 'x'.join(str(s) for s in all_data[0].shape[1:])
                print(f"  Single-bin mode: processing {n_sets} datasets "
                      f"({total_samples} total samples, {shape_str})",
                      flush=True)

            t0 = time.time()
            all_diagrams = []
            for i in range(n_sets):
                data_i = all_data[i].to(self.device)
                all_data[i] = None
                diagrams = self.filtration(data_i)
                all_diagrams.append(diagrams)
                del data_i
            del all_data
            elapsed = time.time() - t0

            if is_tda:
                n_hom_out = len(all_diagrams[0])
                print(f"    → {n_hom_out} hom dims, {elapsed:.1f}s total",
                      flush=True)

            all_summaries = self.vectorize(all_diagrams)

            self._save_summaries_cache(all_summaries, config)
            return all_summaries