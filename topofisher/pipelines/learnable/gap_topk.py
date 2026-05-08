"""
Pipeline for GAP + precomputed TopK persistence training.

Extends LearnableFiltrationPipeline by overriding data loading to wrap
field tensors with precomputed TopK sidecar files in a FieldWithTopK wrapper.

The field stays mmap-backed (identical to CNN+GAP) and TopK features are
stored as a tiny separate tensor (~3MB).  The wrapper packs them together
only at batch time on GPU (25×513×512 = 26MB), avoiding the 100GB
torch.cat that would otherwise convert evictable mmap pages to anonymous RAM.

Memory profile: identical to CNN+GAP (~120G).
"""
import os

import torch
import torch.nn.functional as F

from ..base import FieldWithTopK
from .filtration import LearnableFiltrationPipeline


class GAPTopKFiltrationPipeline(LearnableFiltrationPipeline):
    """
    Learnable filtration pipeline that loads TopK sidecar data alongside fields.

    For each per-bin data file (e.g., fiducial_bin0.pt), also loads the
    corresponding TopK file (fiducial_bin0_topk.pt) and wraps them in a
    FieldWithTopK:

        field: (n_samples, H, W)       — mmap-backed, evictable
        topk:  (n_samples, topk_dim)   — tiny (~3MB)
        wrapper shape: (n_samples, H + n_extra_rows, W)

    The wrapper packs field + topk into a single tensor only at batch time
    (in .to(device)), so the full-dataset torch.cat never happens.
    """

    def _load_bin_data(self, per_bin_paths, set_idx, bin_idx, mmap=True):
        """Load field data + TopK sidecar, return FieldWithTopK wrapper."""
        # Load original field data (mmap-backed — identical to CNN+GAP)
        field = super()._load_bin_data(per_bin_paths, set_idx, bin_idx, mmap=mmap)
        n_samples, H, W = field.shape

        # Load TopK sidecar
        field_path = per_bin_paths[set_idx][bin_idx]
        topk_path = field_path.replace('.pt', '_topk.pt')

        if not os.path.isfile(topk_path):
            raise FileNotFoundError(
                f"TopK sidecar not found: {topk_path}. "
                f"Run scripts/precompute_topk_persistence.py first."
            )

        topk_data = torch.load(topk_path, map_location='cpu', weights_only=False)

        # Extract and flatten TopK features
        k = self.filtration.topk_k
        h0 = topk_data['h0_topk'][:, :k, :]  # (n, k, 2)
        h1 = topk_data['h1_topk'][:, :k, :]  # (n, k, 2)
        topk_flat = torch.cat([h0.flatten(1), h1.flatten(1)], dim=1)  # (n, 4k)

        # Pad to full row width and reshape into extra rows
        n_extra_rows = (topk_flat.shape[1] + W - 1) // W
        pad_len = n_extra_rows * W - topk_flat.shape[1]
        if pad_len > 0:
            topk_flat = F.pad(topk_flat, (0, pad_len))
        topk_rows = topk_flat.reshape(n_samples, n_extra_rows, W)

        # Return wrapper: field stays mmap, topk is tiny (~3MB)
        # torch.cat only happens per-batch in FieldWithTopK.to(device)
        return FieldWithTopK(field, topk_rows)
