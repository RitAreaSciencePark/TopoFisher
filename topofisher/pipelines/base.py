"""
Base pipeline for Fisher information analysis.

Forward-only pipeline with no training capabilities.
"""
import os
import json
import time
from typing import List
import numpy as np
import torch
import torch.nn as nn

from ..config import AnalysisConfig, FisherResult


class LazyStackedTensor:
    """Memory-efficient lazy stacking of mmap'd per-bin tensors.

    Wraps a list of per-bin tensors [(n_samples, H, W), ...] and presents
    them as a single (n_samples, n_bins, H, W) tensor. Only materializes
    the stacked data when indexed (e.g., during batch extraction).

    Supports chained lazy indexing via ``_indices``: when a subset is
    selected (e.g., during train/val/test splitting), the original mmap'd
    tensors are kept and only the index mapping is updated.  Actual data
    is materialised only at batch-extraction time (typically 50 samples).

    This avoids the ~500 GB memory spike from torch.stack on 5 bins of
    20,000 × 512 × 512 float32 tensors across 5 datasets.
    """

    def __init__(self, bin_tensors: list, indices=None):
        self._bins = bin_tensors          # list of (n_total, H, W) mmap tensors
        self._indices = indices           # optional: subset indices into _bins
        n_samples = len(indices) if indices is not None else bin_tensors[0].shape[0]
        self.shape = torch.Size([
            n_samples,
            len(bin_tensors),             # n_bins
            *bin_tensors[0].shape[1:],    # H, W
        ])
        self.dtype = bin_tensors[0].dtype
        self.device = bin_tensors[0].device

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        """Index along sample dimension, stack bins on-the-fly.

        Resolves through ``_indices`` first so that chained lazy subsets
        hit the original mmap tensors directly.
        """
        if self._indices is not None:
            actual_idx = self._indices[idx]
        else:
            actual_idx = idx
        stacked = torch.stack([b[actual_idx] for b in self._bins], dim=-3)
        return stacked

    def lazy_subset(self, idx):
        """Return a new LazyStackedTensor viewing the same mmap bins
        but restricted to ``idx`` samples (no data copy)."""
        if self._indices is not None:
            actual_idx = self._indices[idx]
        else:
            actual_idx = idx
        return LazyStackedTensor(self._bins, indices=actual_idx)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def size(self, *args):
        if args:
            return self.shape[args[0]]
        return self.shape


class FieldWithTopK:
    """Thin wrapper: mmap field (n, H, W) + tiny TopK rows (n, R, W).

    Presents as a (n, H+R, W) tensor.  Critically, __getitem__ with fancy
    indices produces a NEW FieldWithTopK whose field part is a contiguous
    RAM tensor (fancy-indexed from mmap — same as CNN+GAP split_data path)
    and whose TopK part is also contiguous.

    This means split_data works identically to CNN+GAP:
    - After split, train/val/test hold contiguous RAM (evictable mmap pages
      freed) + tiny TopK.
    - No torch.cat on the full 20k dataset at load time.
    - torch.cat only happens per-batch in .to(device), where 25×513×512 = 26MB.

    Memory profile: identical to CNN+GAP + negligible TopK overhead (~3MB).
    """

    def __init__(self, field, topk_rows):
        self._field = field          # (n, H, W) — mmap or contiguous
        self._topk_rows = topk_rows  # (n, R, W) — tiny
        n = field.shape[0]
        H = field.shape[1]
        R = topk_rows.shape[1]
        W = field.shape[2]
        self.shape = torch.Size([n, H + R, W])
        self.dtype = field.dtype
        self.device = field.device

    def __len__(self):
        return self.shape[0]

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def size(self, *args):
        if args:
            return self.shape[args[0]]
        return self.shape

    def __getitem__(self, idx):
        """Index both field and topk, return new FieldWithTopK.

        For fancy indexing (e.g., split_data permutation), this materializes
        the mmap field into contiguous RAM — same as plain tensor[perm] in
        CNN+GAP.  For slice/int indexing (batch extraction), same behavior.
        """
        f = self._field[idx]
        t = self._topk_rows[idx]
        # Single-sample case: expand back to 3D
        if f.ndim == 2:
            f = f.unsqueeze(0)
            t = t.unsqueeze(0)
        return FieldWithTopK(f, t)

    def lazy_subset(self, idx):
        """For split_data compat. Materializes field (same as __getitem__)."""
        return self[idx]

    def to(self, *args, **kwargs):
        """Move to device by packing field + topk → plain tensor.

        This is only called at batch time (25 samples = 26MB), so torch.cat
        is cheap.  Returns a plain tensor so downstream code (filtration
        forward, .float(), etc.) works unchanged.
        """
        f = self._field.to(*args, **kwargs)
        t = self._topk_rows.to(*args, **kwargs)
        return torch.cat([f, t], dim=1)  # plain tensor (B, H+R, W)

    def float(self):
        """Cast to float32 by packing → plain tensor."""
        f = self._field.float()
        t = self._topk_rows.float()
        return torch.cat([f, t], dim=1)


# Keep old name as alias for backward compat
LazyFieldTopK = FieldWithTopK

# Tuple for isinstance checks — all lazy/wrapper tensor types
class LazyMmapTensor:
    """Lazy view over a single mmap'd tensor with optional sample-axis indices.

    Lets train/val/test split keep a permutation of indices instead of
    materializing a (n_train, ...) contiguous copy in RAM. Only the per-batch
    `__getitem__` materializes data via fancy-indexing into the underlying
    mmap.

    Used for precomputed feature tensors where the source file already has
    the desired layout (e.g. (N, K, H, W) float16 dict-feature caches).
    Without this wrapper, train/val/test split fancy-indices the mmap into
    a 70 GB-per-dataset contiguous tensor → OOM.
    """

    def __init__(self, tensor, indices=None):
        self._tensor = tensor                 # (N_total, ...) mmap tensor
        self._indices = indices               # optional subset indices
        n = len(indices) if indices is not None else tensor.shape[0]
        self.shape = torch.Size([n, *tensor.shape[1:]])
        self.dtype = tensor.dtype
        self.device = tensor.device

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if self._indices is not None:
            actual_idx = self._indices[idx]
        else:
            actual_idx = idx
        return self._tensor[actual_idx]

    def lazy_subset(self, idx):
        """Return a new LazyMmapTensor restricted to ``idx`` (no data copy)."""
        if self._indices is not None:
            actual_idx = self._indices[idx]
        else:
            actual_idx = idx
        return LazyMmapTensor(self._tensor, indices=actual_idx)

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return len(self.shape)

    def size(self, *args):
        if args:
            return self.shape[args[0]]
        return self.shape


LAZY_TENSOR_TYPES = (LazyStackedTensor, FieldWithTopK, LazyMmapTensor)


def _robust_torch_load(fpath, max_retries=5, base_delay=5.0, **kwargs):
    """Load a torch file with retries to handle transient CephFS errors."""
    for attempt in range(max_retries):
        try:
            return torch.load(fpath, **kwargs)
        except (RuntimeError, EOFError, OSError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  [WARN] torch.load failed on {fpath} (attempt {attempt+1}/{max_retries}): "
                  f"{type(e).__name__}: {e}")
            print(f"         Retrying in {delay:.0f}s...")
            time.sleep(delay)
    # Should never reach here
    return torch.load(fpath, **kwargs)

class BasePipeline(nn.Module):
    """
    Base pipeline for Fisher information analysis.

    This pipeline orchestrates the full workflow:
        simulator → filtration → vectorization → compression → fisher_analyzer

    For non-learnable components only (e.g., MOPED compression).
    Use learnable pipelines for training neural network components.
    """

    def __init__(
        self,
        simulator,
        filtration: nn.Module,
        vectorization: nn.Module,
        compression: nn.Module,
        fisher_analyzer
    ):
        """
        Initialize pipeline.

        Args:
            simulator: Data simulator (must have generate() method)
            filtration: Persistence diagram computation (nn.Module with forward())
            vectorization: Diagram vectorization (nn.Module with forward())
            compression: Summary compression (nn.Module with forward())
            fisher_analyzer: Fisher matrix computation
        """
        super().__init__()

        # Auto-detect device (single source of truth)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store components
        self.simulator = simulator
        self.filtration = filtration
        self.vectorization = vectorization
        self.compression = compression
        self.fisher_analyzer = fisher_analyzer

        # Move all nn.Module components to device
        self.to(self.device)

        # Move simulator to device if it has a device attribute
        if hasattr(self.simulator, 'device'):
            self.simulator.device = self.device

    def generate_data(self, config: AnalysisConfig, verbose: bool = True) -> List[torch.Tensor]:
        """
        Generate raw data at fiducial and perturbed parameter values.

        If config.data_dir is set:
          - Loads pre-generated data from disk if available and metadata matches.
          - Otherwise generates the data and saves it to data_dir for reuse.

        Args:
            config: Pipeline configuration
            verbose: Whether to show progress bars

        Returns:
            List of data tensors: [fid, minus_0, plus_0, minus_1, plus_1, ...]
        """
        # Try loading from disk if data_dir is set
        if config.data_dir is not None:
            loaded = self._try_load_data(config, verbose=verbose)
            if loaded is not None:
                return loaded

        # No data loaded from disk — need simulator to generate
        if self.simulator is None:
            raise RuntimeError(
                f"Cannot generate data: no simulator configured and "
                f"pre-generated data could not be loaded from '{config.data_dir}'. "
                f"Check that data_dir exists and metadata matches."
            )

        # Generate data from simulator
        all_data = self._generate_from_simulator(config, verbose=verbose)

        # Save to disk if data_dir is set (for future reuse)
        if config.data_dir is not None:
            self._save_data(all_data, config, verbose=verbose)

        return all_data

    def _build_data_metadata(self, config: AnalysisConfig) -> dict:
        """Build metadata dict that uniquely identifies a simulation dataset."""
        meta = {
            'theta_fid': config.theta_fid.cpu().tolist(),
            'delta_theta': config.delta_theta.cpu().tolist(),
            'n_s': config.n_s,
            'n_d': config.n_d,
            'seed_cov': config.seed_cov,
            'seed_ders': list(config.seed_ders),
        }
        if self.simulator is not None:
            meta['simulator'] = {
                'type': self.simulator.__class__.__name__,
                'params': getattr(self.simulator, 'get_config', lambda: {})(),
            }
        return meta

    def _try_load_data(
        self, config: AnalysisConfig, verbose: bool = True
    ) -> List[torch.Tensor]:
        """
        Try to load pre-generated data from config.data_dir.

        Supports three data layouts:
        1. Monolithic files: fiducial.pt, minus_p0.pt, etc. (with metadata.json)
        2. Per-bin files with bin_idx: loads only fiducial_bin{idx}.pt → (n_samples, H, W)
        3. Per-bin files without bin_idx: stacks all bins → (n_samples, n_bins, H, W)

        Returns:
            List of data tensors if loaded successfully, None otherwise.
        """
        data_dir = config.data_dir

        # --- Try monolithic files first ---
        meta_path = os.path.join(data_dir, 'metadata.json')
        has_metadata = os.path.isfile(meta_path)

        # Check monolithic files
        n_params = len(config.delta_theta)
        mono_fnames = ['fiducial.pt']
        for i in range(n_params):
            mono_fnames.append(f'minus_p{i}.pt')
            mono_fnames.append(f'plus_p{i}.pt')

        has_monolithic = all(
            os.path.isfile(os.path.join(data_dir, f)) for f in mono_fnames
        )

        if has_monolithic and has_metadata:
            # Validate metadata
            with open(meta_path, 'r') as f:
                saved_meta = json.load(f)
            expected_meta = self._build_data_metadata(config)

            float_keys = {'theta_fid', 'delta_theta'}
            meta_ok = True
            for key in ['theta_fid', 'delta_theta', 'n_s', 'n_d', 'seed_cov', 'seed_ders']:
                saved_val = saved_meta.get(key)
                expected_val = expected_meta.get(key)
                if key in float_keys and saved_val is not None and expected_val is not None:
                    if not np.allclose(saved_val, expected_val, rtol=1e-5, atol=1e-8):
                        if verbose:
                            print(f"  Metadata mismatch on '{key}': "
                                  f"saved={saved_val}, expected={expected_val}")
                        meta_ok = False
                        break
                elif saved_val != expected_val:
                    if verbose:
                        print(f"  Metadata mismatch on '{key}': "
                              f"saved={saved_val}, expected={expected_val}")
                    meta_ok = False
                    break

            if meta_ok:
                if 'simulator' in expected_meta and 'simulator' in saved_meta:
                    if saved_meta['simulator'] != expected_meta['simulator']:
                        if verbose:
                            print(f"  Simulator config mismatch, will regenerate data.")
                        meta_ok = False

            if meta_ok:
                t0 = time.time()
                all_data = []
                for fname in mono_fnames:
                    fpath = os.path.join(data_dir, fname)
                    tensor = _robust_torch_load(fpath, map_location='cpu', weights_only=True)
                    all_data.append(tensor)
                elapsed = time.time() - t0
                if verbose:
                    shape_str = 'x'.join(str(s) for s in all_data[0].shape)
                    print(f"  Loaded pre-generated data from {data_dir}")
                    print(f"    {len(all_data)} datasets, shape {shape_str}, "
                          f"{elapsed:.1f}s load time")
                return all_data

        # --- Try per-bin files ---
        n_bins, per_bin_paths = self._detect_per_bin_data(config)
        if n_bins > 0:
            bin_idx = getattr(config, 'bin_idx', None)

            if bin_idx is not None:
                # Load single bin: returns (n_samples, H, W) [or (n_samples, K, H, W)
                # for precomputed dict feature caches]. Each tensor is wrapped in
                # LazyMmapTensor so train/val/test split keeps a permutation of
                # indices instead of materializing a (n_train, ...) RAM copy
                # — critical for 4D feature caches where one split would be
                # ~70 GB × 5 datasets and OOM the node.
                if bin_idx >= n_bins:
                    raise ValueError(f"bin_idx={bin_idx} but only {n_bins} bins found")

                t0 = time.time()
                all_data = []
                base_names = self._get_data_file_names(config)
                for set_idx in range(len(base_names)):
                    tensor = self._load_bin_data(per_bin_paths, set_idx, bin_idx)
                    #all_data.append(LazyMmapTensor(tensor))
                    all_data.append(tensor)
                elapsed = time.time() - t0

                if verbose:
                    shape_str = 'x'.join(str(s) for s in all_data[0].shape)
                    print(f"  Loaded per-bin data (bin {bin_idx}/{n_bins-1}) from {data_dir}")
                    print(f"    {len(all_data)} datasets, shape {shape_str}, "
                          f"{elapsed:.1f}s load time")
                return all_data

            else:
                # Stack all bins: returns lazy (n_samples, n_bins, H, W)
                # Uses LazyStackedTensor to avoid materializing the full
                # stacked tensor (~100 GB for 5 bins × 20K × 512×512).
                # Only stacks when indexed (at batch extraction time).
                t0 = time.time()
                all_data = []
                base_names = self._get_data_file_names(config)
                for set_idx in range(len(base_names)):
                    bin_tensors = []
                    for b in range(n_bins):
                        t = self._load_bin_data(per_bin_paths, set_idx, b)
                        bin_tensors.append(t)
                    lazy = LazyStackedTensor(bin_tensors)
                    all_data.append(lazy)
                elapsed = time.time() - t0

                if verbose:
                    shape_str = 'x'.join(str(s) for s in all_data[0].shape)
                    print(f"  Loaded per-bin data ({n_bins} bins, lazy-stacked) from {data_dir}")
                    print(f"    {len(all_data)} datasets, shape {shape_str}, "
                          f"{elapsed:.1f}s load time")
                return all_data

        if verbose:
            print(f"  No cached data found at {data_dir}, will generate.")
        return None

    # ──────────────────────────────────────────────────────────
    # Per-bin data loading helpers
    # ──────────────────────────────────────────────────────────

    def _get_data_file_names(self, config: AnalysisConfig):
        """Build ordered list of base file names matching generate_data order."""
        n_params = len(config.delta_theta)
        fnames = ['fiducial']
        for i in range(n_params):
            fnames.append(f'minus_p{i}')
            fnames.append(f'plus_p{i}')
        return fnames

    def _get_per_bin_file_paths(self, config: AnalysisConfig, n_bins: int):
        """
        Build per-bin file paths: paths[set_idx][bin_idx] = path.

        Per-bin files use naming: fiducial_bin0.pt, fiducial_bin1.pt, etc.
        Returns None if any per-bin file is missing.
        """
        data_dir = config.data_dir
        base_names = self._get_data_file_names(config)

        paths = []
        for base in base_names:
            bin_paths = []
            for bin_idx in range(n_bins):
                p = os.path.join(data_dir, f'{base}_bin{bin_idx}.pt')
                if not os.path.isfile(p):
                    return None
                bin_paths.append(p)
            paths.append(bin_paths)
        return paths

    def _detect_per_bin_data(self, config: AnalysisConfig):
        """
        Detect per-bin data files in data_dir.

        Returns:
            (n_bins, per_bin_paths) if per-bin files found, (0, None) otherwise.
        """
        if config.data_dir is None:
            return 0, None

        # Check if fiducial_bin0.pt exists
        test_path = os.path.join(config.data_dir, 'fiducial_bin0.pt')
        if not os.path.isfile(test_path):
            return 0, None

        # Count bins
        n_bins = 0
        while os.path.isfile(os.path.join(config.data_dir, f'fiducial_bin{n_bins}.pt')):
            n_bins += 1

        # Verify all datasets have the same bins
        per_bin_paths = self._get_per_bin_file_paths(config, n_bins)
        if per_bin_paths is None:
            return 0, None

        return n_bins, per_bin_paths

    def _load_bin_data(self, per_bin_paths, set_idx, bin_idx, mmap=True):
        """
        Load one bin's data for one dataset.

        Args:
            per_bin_paths: paths[set_idx][bin_idx] from _get_per_bin_file_paths
            set_idx: dataset index (0=fid, 1=minus_p0, 2=plus_p0, ...)
            bin_idx: tomographic bin index
            mmap: use memory-mapping (default True)

        Returns:
            Tensor of shape (n_samples, H, W) on CPU.
        """
        return _robust_torch_load(
            per_bin_paths[set_idx][bin_idx],
            map_location='cpu', weights_only=True,
            mmap=mmap
        )

    def _forward_per_bin(self, config: AnalysisConfig) -> FisherResult:
        """
        Process per-bin data for non-TDA filtrations.

        Loads one bin at a time, applies filtration + vectorization per bin,
        concatenates features across bins, then applies compression + Fisher.

        This avoids loading all bins simultaneously (which would exceed memory
        for large datasets like 20K × 512×512 lensing maps).

        Args:
            config: Pipeline configuration

        Returns:
            Fisher information results
        """
        import gc

        n_bins, per_bin_paths = self._detect_per_bin_data(config)
        if n_bins == 0:
            raise RuntimeError("_forward_per_bin called but no per-bin data found")

        # When config.bin_idx is set, process only that single bin.
        # When not set, process all bins and concatenate features.
        bin_idx_only = getattr(config, 'bin_idx', None)
        bins_to_process = [bin_idx_only] if bin_idx_only is not None else list(range(n_bins))

        base_names = self._get_data_file_names(config)
        n_sets = len(base_names)

        # Read shape from first per-bin file
        first_tensor = torch.load(
            per_bin_paths[0][bins_to_process[0]], map_location='cpu', weights_only=True, mmap=True
        )
        n_samples = first_tensor.shape[0]
        shape_str = 'x'.join(str(s) for s in first_tensor.shape[1:])
        del first_tensor

        print(f"  Per-bin processing: {n_sets} datasets × {len(bins_to_process)} bins "
              f"({bins_to_process}), {n_samples} samples, field shape {shape_str}", flush=True)

        # Determine device and move pipeline components there
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.filtration.to(device)
        self.vectorization.to(device)

        # Collect features per set, across bins
        per_set_features = [[] for _ in range(n_sets)]
        BATCH_SIZE = 200  # process maps in batches to limit GPU memory

        for bin_idx in bins_to_process:
            t0_bin = time.time()
            print(f"  Bin {bin_idx}/{n_bins-1}:", flush=True)

            for set_idx in range(n_sets):
                bin_data = self._load_bin_data(per_bin_paths, set_idx, bin_idx)
                n_total = bin_data.shape[0]

                batch_features = []
                for chunk_start in range(0, n_total, BATCH_SIZE):
                    chunk_end = min(chunk_start + BATCH_SIZE, n_total)
                    chunk = bin_data[chunk_start:chunk_end].clone()
                    chunk = chunk.to(device).float()

                    with torch.no_grad():
                        diagrams = self.filtration(chunk)
                        summaries = self.vectorization(diagrams)
                    batch_features.append(summaries.cpu())
                    del chunk, diagrams, summaries
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                set_features = torch.cat(batch_features, dim=0)
                per_set_features[set_idx].append(set_features)
                del bin_data, batch_features
                gc.collect()

            elapsed = time.time() - t0_bin
            print(f"    done ({elapsed:.1f}s)", flush=True)

        # Concatenate across bins: (n_samples, feat_bin0 | feat_bin1 | ...)
        all_summaries = []
        for set_idx in range(n_sets):
            combined = torch.cat(per_set_features[set_idx], dim=-1)
            all_summaries.append(combined)
        del per_set_features

        feat_dim = all_summaries[0].shape[-1]
        print(f"  Combined: {feat_dim}-dim features per sample", flush=True)

        # Compress (MOPED)
        compressed = self.compress(all_summaries, config.delta_theta)

        # Compute Fisher
        result = self.compute_fisher(
            compressed, config.delta_theta, check_gaussianity=True
        )

        return result

    def _save_data(
        self,
        all_data: List[torch.Tensor],
        config: AnalysisConfig,
        verbose: bool = True,
    ) -> None:
        """Save generated data to config.data_dir for future reuse."""
        data_dir = config.data_dir
        os.makedirs(data_dir, exist_ok=True)

        t0 = time.time()

        # Save metadata
        meta = self._build_data_metadata(config)
        with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        # Save tensors: fiducial + derivative pairs
        n_params = len(config.delta_theta)
        fnames = ['fiducial.pt']
        for i in range(n_params):
            fnames.append(f'minus_p{i}.pt')
            fnames.append(f'plus_p{i}.pt')

        total_bytes = 0
        for fname, tensor in zip(fnames, all_data):
            fpath = os.path.join(data_dir, fname)
            torch.save(tensor.cpu(), fpath)
            total_bytes += tensor.nelement() * tensor.element_size()

        elapsed = time.time() - t0

        if verbose:
            size_gb = total_bytes / (1024**3)
            print(f"  Saved simulation data to {data_dir}")
            print(f"    {len(all_data)} datasets, {size_gb:.2f} GB, "
                  f"{elapsed:.1f}s save time")

    def _generate_from_simulator(
        self, config: AnalysisConfig, verbose: bool = True
    ) -> List[torch.Tensor]:
        """
        Generate raw data from the simulator (original logic).

        Args:
            config: Pipeline configuration
            verbose: Whether to show progress bars

        Returns:
            List of data tensors: [fid, minus_0, plus_0, minus_1, plus_1, ...]
        """
        n_params = len(config.delta_theta)
        all_data = []

        # Fiducial samples (for covariance)
        fid_data = self.simulator.generate(
            theta=config.theta_fid,
            n_samples=config.n_s,
            seed=config.seed_cov,
            desc=f"Generating fiducial samples (n={config.n_s})" if verbose else None
        )
        all_data.append(fid_data)

        # Derivative samples (theta ± delta_theta/2 for each parameter)
        for i in range(n_params):
            # Get seed for this parameter's derivatives
            seed_der = config.seed_ders[i] if config.seed_ders is not None else None

            # theta - delta_theta/2
            theta_minus = config.theta_fid.clone()
            theta_minus[i] -= config.delta_theta[i] / 2
            data_minus = self.simulator.generate(
                theta=theta_minus,
                n_samples=config.n_d,
                seed=seed_der,
                desc=f"Generating θ_{i}⁻ samples (n={config.n_d})" if verbose else None
            )
            all_data.append(data_minus)

            # theta + delta_theta/2
            theta_plus = config.theta_fid.clone()
            theta_plus[i] += config.delta_theta[i] / 2
            data_plus = self.simulator.generate(
                theta=theta_plus,
                n_samples=config.n_d,
                seed=seed_der,  # Same seed for theta+ and theta- to reduce variance
                desc=f"Generating θ_{i}⁺ samples (n={config.n_d})" if verbose else None
            )
            all_data.append(data_plus)

        return all_data

    def _is_tda_filtration(self) -> bool:
        """Check whether the filtration computes persistence diagrams (TDA).

        CNN+persistence filtrations return False here so that forward() routes
        them through _forward_per_bin, which processes filtration+vectorization
        in small batches (200 maps) and frees GPU memory between batches.
        The alternative (standard TDA path) computes all 100k diagrams first,
        which leaves them on GPU (~38 GB) before vectorization can start.
        """
        from ..filtrations.learnable_cnn import LearnableCNNFiltration
        from ..filtrations.learnable_mlp import LearnableMLPFiltration
        from ..filtrations.cnn_gap import CNNGAPFiltration
        from ..filtrations.scattering import ScatteringFiltration
        from ..filtrations.power_spectrum import PowerSpectrumFiltration
        from ..filtrations.peak_counts import PeakCountsFiltration
        from ..filtrations.identity import IdentityFiltration
        from ..filtrations.cnn_fullres_persistence_v2 import CNNFullResPersistenceV2Filtration
        from ..filtrations.cnn_strided_persistence import CNNStridedPersistenceFiltration
        non_tda_types = (
            LearnableCNNFiltration, LearnableMLPFiltration,
            CNNGAPFiltration, ScatteringFiltration, PowerSpectrumFiltration,
            PeakCountsFiltration, IdentityFiltration,
            CNNFullResPersistenceV2Filtration, CNNStridedPersistenceFiltration,
        )
        return not isinstance(self.filtration, non_tda_types)

    def compute_diagrams(self, all_data: List[torch.Tensor], verbose: bool = True) -> List[List[List[torch.Tensor]]]:
        """
        Compute persistence diagrams (or feature vectors) from raw data.

        Args:
            all_data: List of data tensors
            verbose: Whether to print progress (default True, but suppressed
                     during training where it would flood the log)

        Returns:
            List of diagram sets: [fid_diagrams, minus_diagrams_0, plus_diagrams_0, ...]
            Each set has structure: List[hom_dim][sample] -> diagram
        """
        import time
        is_tda = self._is_tda_filtration()
        n_sets = len(all_data)
        total_samples = sum(d.shape[0] for d in all_data)

        if verbose and is_tda:
            shape_str = 'x'.join(str(s) for s in all_data[0].shape[1:])
            print(f"  Computing persistence for {n_sets} sets "
                  f"({total_samples} total samples, {shape_str})...", flush=True)

        # Move data to model device when the filtration has GPU parameters.
        # This covers non-TDA filtrations (CNN, Scattering, PS) AND learnable
        # TDA filtrations (e.g., LearnableDownsampleFiltration with a CNN
        # followed by GPU cubical persistence).
        if hasattr(self, 'device'):
            target_device = self.device
        else:
            # Try parameters first, then buffers
            param = next(self.filtration.parameters(), None)
            if param is not None:
                target_device = param.device
            else:
                buf = next(self.filtration.buffers(), None)
                target_device = buf.device if buf is not None else torch.device('cpu')
        if not is_tda:
            self.filtration.to(target_device)
        # For CPU-only filtrations target_device is 'cpu' → .to() is a no-op
        if target_device == torch.device('cpu'):
            target_device = None  # skip unnecessary .to() call

        # For TDA filtrations with GPU, chunk the data transfer to avoid OOM.
        # 20k×512×512 + CNN intermediates (8ch) = ~180 GB — exceeds any GPU.
        # Use outer chunks of 500 (matching eval_batch_size used during training);
        # the persistence layer handles its own sub_batch_size internally.
        tda_chunk_size = None
        if is_tda and target_device is not None:
            tda_chunk_size = 500

        t0 = time.time()
        all_diagrams = []
        for data in all_data:
            if tda_chunk_size is not None and isinstance(data, (torch.Tensor,) + LAZY_TENSOR_TYPES) and data.shape[0] > tda_chunk_size:
                # Chunk the GPU transfer and filtration to avoid OOM
                n = data.shape[0]
                chunk_diagrams = None
                for start in range(0, n, tda_chunk_size):
                    end = min(start + tda_chunk_size, n)
                    chunk = data[start:end].to(target_device).float()
                    with torch.no_grad():
                        cdgms = self.filtration(chunk)
                    del chunk
                    if target_device.type == 'cuda':
                        torch.cuda.empty_cache()
                    if chunk_diagrams is None:
                        chunk_diagrams = [[] for _ in range(len(cdgms))]
                    for h, hom_dgms in enumerate(cdgms):
                        chunk_diagrams[h].extend([d.cpu() if isinstance(d, torch.Tensor) else d for d in hom_dgms])
                all_diagrams.append(chunk_diagrams)
            else:
                if target_device is not None:
                    data = data.to(target_device).float()
                diagrams = self.filtration(data)
                all_diagrams.append(diagrams)
        elapsed = time.time() - t0

        if verbose and is_tda:
            n_hom = len(all_diagrams[0])
            print(f"    → {n_hom} hom dims, {elapsed:.1f}s total", flush=True)

        return all_diagrams

    def vectorize(self, all_diagrams: List[List[List[torch.Tensor]]]) -> List[torch.Tensor]:
        """
        Vectorize persistence diagrams to feature summaries.

        IMPORTANT: Vectorization hyperparameters (bounds, ranges, etc.) must be
        CONSISTENT across all sets. Use fit() on ALL diagrams before transform.

        Args:
            all_diagrams: List of diagram sets

        Returns:
            List of summary tensors: [fid_summaries, minus_summaries_0, plus_summaries_0, ...]
        """
        # Fit vectorization on ALL diagrams to ensure consistent hyperparameters
        # Only fit ONCE; subsequent calls reuse the fitted parameters
        if hasattr(self.vectorization, 'fit') and not getattr(self, '_vectorization_fitted', False):
            self.vectorization.fit(all_diagrams)
            self._vectorization_fitted = True

        # Transform each set using the SAME fitted parameters.
        # Only skip grad tracking in eval mode — in training mode, gradients must
        # flow through the vectorization output back to filtration/vectorization params.
        # (no_grad was previously applied unconditionally here, which silently broke
        # all trainable filtration pipelines by detaching CNN/PersLay from autograd.)
        all_summaries = []
        import contextlib
        ctx = contextlib.nullcontext() if self.training else torch.no_grad()
        with ctx:
            for diagrams in all_diagrams:
                summaries = self.vectorization(diagrams)
                all_summaries.append(summaries)

        return all_summaries

    def compress(
        self,
        all_summaries: List[torch.Tensor],
        delta_theta: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Apply compression to summaries.

        Note: delta_theta is kept in signature for backward compatibility but not
        passed to compression. The Fisher matrix is invariant to the delta_theta
        values used internally by compression methods like MOPED.

        Args:
            all_summaries: List of summary tensors
            delta_theta: Parameter step sizes (kept for interface compatibility)

        Returns:
            Compressed summaries (may be smaller if compression splits data)
        """
        return self.compression(all_summaries)

    def compute_fisher(
        self,
        compressed_summaries: List[torch.Tensor],
        delta_theta: torch.Tensor,
        check_gaussianity: bool = False
    ) -> FisherResult:
        """
        Compute Fisher information matrix.

        Args:
            compressed_summaries: Compressed summary tensors
            delta_theta: Parameter step sizes
            check_gaussianity: If True, run Gaussianity test

        Returns:
            Fisher analysis results
        """
        return self.fisher_analyzer(compressed_summaries, delta_theta, check_gaussianity=check_gaussianity)

    def state_dict(self):
        """
        Return state dict of all nn.Module components (PyTorch style).

        Returns nested dict: {component_name: component_state_dict}
        Only includes filtration, vectorization, compression if they are nn.Module.
        """
        state = {}
        for name in ['filtration', 'vectorization', 'compression']:
            component = getattr(self, name, None)
            if component is not None and isinstance(component, nn.Module):
                state[name] = component.state_dict()
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict into nn.Module components (PyTorch style).

        Args:
            state_dict: Dict mapping component name to state_dict
            strict: If True, raise error for missing/unexpected keys
        """
        for name, state in state_dict.items():
            if state is not None:
                component = getattr(self, name, None)
                if component is not None and isinstance(component, nn.Module):
                    component.load_state_dict(state, strict=strict)

    def forward(self, config: AnalysisConfig) -> FisherResult:
        """
        Run full pipeline: simulator → filtration → vectorization → compression → fisher.

        For non-TDA filtrations with per-bin data (e.g., lensing), automatically
        dispatches to memory-efficient per-bin processing that loads one
        tomographic bin at a time.

        Args:
            config: Pipeline configuration

        Returns:
            Fisher information results
        """
        # Use memory-efficient per-bin path for non-TDA and CNN+persistence filtrations.
        # _forward_per_bin processes filtration+vectorization in 200-map batches,
        # freeing GPU memory between batches. This avoids the standard-path issue
        # where 100k diagrams accumulate on GPU (~38 GB) before vectorization starts.
        if not self._is_tda_filtration():
            n_bins, per_bin_paths = self._detect_per_bin_data(config)
            if n_bins > 0:
                return self._forward_per_bin(config)

        # Standard path: load all data, then process sequentially
        # 1. Generate data
        all_data = self.generate_data(config)

        # 2. Compute persistence diagrams
        all_diagrams = self.compute_diagrams(all_data)

        # 3. Vectorize diagrams
        all_summaries = self.vectorize(all_diagrams)

        # 4. Compress summaries
        compressed_summaries = self.compress(all_summaries, config.delta_theta)

        # 5. Compute Fisher matrix (with Gaussianity check)
        result = self.compute_fisher(compressed_summaries, config.delta_theta, check_gaussianity=True)

        return result