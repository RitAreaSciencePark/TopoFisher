"""
Full-resolution CNN + adaptive avg pool — NN baseline for Phase 6.

Same backbone architecture as cnn_fullres_persistence_v2 (same param count),
but NO skip connection and replaces persistence with adaptive avg pool → flatten.

The persistence path needs skip connection so the field starts near the raw input
(meaningful topology from epoch 1). The flat path does NOT — the CNN should
directly learn informative features, not small perturbations of the raw field.

The comparison is:
  - Persistence: skip CNN → topology-driven compression (512² → ~120 features)
  - Avg pool: direct CNN → spatial averaging compression (512² → pool_size² features)

Both have exactly the same 673 trainable parameters.

Architecture:
  Input (B, H, W)
  │
  Conv2d(1→C, k, s=1, circular) + ReLU   × n_layers
  Conv2d(C→1, k=1)  [no activation]
  │
  field = cnn(input)    # (B, 1, H, W)  — no skip connection
  │
  → per-sample standardize
  → AdaptiveAvgPool2d(pool_size)
  → flatten  →  pool_size² features
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFullResFlatFiltration(nn.Module):
    """
    Full-res CNN + adaptive avg pool + flatten.  No skip connection.

    Args:
        hidden_channels: Backbone intermediate channels (default 8).
        n_layers: Backbone conv layers before the 1×1 projection (default 2).
        kernel_size: Backbone kernel size (default 3).
        pool_size: AdaptiveAvgPool2d target size (default 16 → 256-dim output).
        standardize: Per-sample standardize before pool (default True).
        init_scale: Unused (kept for config compatibility). Always Xavier.
    """

    def __init__(
        self,
        hidden_channels: int = 8,
        n_layers: int = 2,
        kernel_size: int = 3,
        pool_size: int = 16,
        standardize: bool = True,
        init_scale: Optional[float] = None,
    ):
        super().__init__()

        self.standardize = standardize
        self._hidden_channels = hidden_channels
        self._n_layers = n_layers
        self._kernel_size = kernel_size
        self._init_scale = init_scale
        self._pool_size = pool_size

        # === Backbone: stride-1 CNN with circular padding ===
        pad = kernel_size // 2
        layers = []
        in_ch = 1
        for _ in range(n_layers):
            layers.append(nn.Conv2d(
                in_ch, hidden_channels, kernel_size,
                stride=1, padding=pad, padding_mode='circular',
            ))
            layers.append(nn.ReLU(inplace=True))
            in_ch = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, 1, 1))
        self.backbone = nn.Sequential(*layers)

        # === Non-learnable compression: adaptive avg pool ===
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.output_dim = pool_size * pool_size

        self._init_weights(init_scale)

    def _init_weights(self, scale):
        total = sum(p.numel() for p in self.backbone.parameters())
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(f"  Xavier backbone ({total} params), "
              f"pool={self._pool_size}×{self._pool_size}, "
              f"output_dim={self.output_dim}", flush=True)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x_4d = x.unsqueeze(1)  # (B, 1, H, W)

        # Direct CNN transform — no skip connection
        field = self.backbone(x_4d)  # (B, 1, H, W)

        # Per-sample standardization
        if self.standardize:
            mean = field.mean(dim=(2, 3), keepdim=True)
            std = field.std(dim=(2, 3), keepdim=True)
            field = (field - mean) / (std + 1e-8)

        # Pool → flatten
        pooled = self.pool(field)         # (B, 1, pool_size, pool_size)
        flat = pooled.flatten(1)          # (B, pool_size²)

        vectors = [flat[i] for i in range(flat.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        total = sum(p.numel() for p in self.backbone.parameters())
        return (f"CNNFullResFlatFiltration(\n"
                f"  cnn: {self._n_layers}×Conv2d({self._hidden_channels}ch, "
                f"k={self._kernel_size}, stride=1, circular) + 1×1, "
                f"params={total}\n"
                f"  field = cnn(input)  [no skip connection]\n"
                f"  pool: AdaptiveAvgPool2d({self._pool_size}) → "
                f"{self.output_dim}-dim\n"
                f"  standardize={self.standardize}\n"
                f")")
