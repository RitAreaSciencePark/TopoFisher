"""
Full-resolution CNN with strided downsampling + periodic cubical persistence.

The CNN runs feature extraction at full input resolution so every fine-scale
topological structure is visible to the gradient. A cascade of stride-2 conv
layers then brings the filtration field down to output_size×output_size before
persistence, making persistence (and its backward) cheap.

A residual skip — avg_pool(input, stride=H/output_size) — is added to the
strided output. With near-zero init this gives output ≈ avg_pool(input) at
epoch 1, so PersLay normalization is fitted on diagrams of the avg-pooled
field (a valid, coarse-but-real topological starting point). As training
proceeds the CNN learns a better filtration function at the coarser resolution.

output_size=64 → 3 stride-2 layers (512→256→128→64), CPU periodic ~160 ms/32
output_size=32 → 4 stride-2 layers (512→256→128→64→32), GPU periodic ~3 ms/32
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStridedPersistenceFiltration(nn.Module):
    """
    Args:
        output_size:        Persistence grid side length after striding (32 or 64).
        hidden_channels:    Channels in every conv layer.
        n_full_res_layers:  Conv layers at full input resolution before striding.
        kernel_size:        Kernel for all conv layers (stride-2 layers included).
        homology_dimensions: Hom dims for persistence (default [0, 1]).
        construction:       'T' (top cells) or 'V' (vertices).
        sub_batch_size:     Samples per GPU persistence call.
        standardize:        Per-sample standardize the filtration field before
                            persistence (applied to the strided output).
        init_scale:         Std of weight init. Near-zero → output ≈ avg_pool(input).
    """

    def __init__(
        self,
        output_size: int = 64,
        hidden_channels: int = 8,
        n_full_res_layers: int = 1,
        kernel_size: int = 3,
        homology_dimensions: Optional[List[int]] = None,
        construction: str = 'T',
        sub_batch_size: int = 100,
        standardize: bool = True,
        init_scale: float = 0.01,
    ):
        super().__init__()

        if homology_dimensions is None:
            homology_dimensions = [0, 1]

        self.output_size = output_size
        self.standardize = standardize
        self._kernel_size = kernel_size
        self._pad = kernel_size // 2

        # ── Full-resolution feature extraction ────────────────────────────────
        full_res = []
        in_ch = 1
        for _ in range(n_full_res_layers):
            full_res.append(nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=0))
            full_res.append(nn.ReLU(inplace=True))
            in_ch = hidden_channels
        self.full_res = nn.Sequential(*full_res)

        # ── Strided downsampling cascade ──────────────────────────────────────
        # Stride-2 convs reduce spatial dims by 2 each step.
        # n_strides = log2(input_size / output_size); assumes input is 512.
        import math
        n_strides = int(round(math.log2(512 / output_size)))
        stride_layers = []
        for _ in range(n_strides):
            stride_layers.append(nn.Conv2d(hidden_channels, hidden_channels,
                                           kernel_size, stride=2, padding=0))
            stride_layers.append(nn.ReLU(inplace=True))
        self.stride_layers = nn.ModuleList(stride_layers[i]
                                           for i in range(0, len(stride_layers), 2))
        self.stride_acts = nn.ModuleList(stride_layers[i]
                                         for i in range(1, len(stride_layers), 2))

        # ── 1×1 projection to single channel ─────────────────────────────────
        self.proj = nn.Conv2d(hidden_channels, 1, 1)

        # ── Persistence ───────────────────────────────────────────────────────
        from .gudhi_gpu_cubical import GUDHIGPUCubicalLayer
        self.persistence = GUDHIGPUCubicalLayer(
            homology_dimensions=homology_dimensions,
            min_persistence=[0.0] * len(homology_dimensions),
            superlevel=False,
            construction=construction,
            sub_batch_size=sub_batch_size,
            periodic=True,          # periodic BC; GPU path for output_size=32,
        )                           # CPU fallback for 64 (still fast).

        self._init_weights(init_scale)
        self._n_strides = n_strides

    def _init_weights(self, scale: float):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Identity-like init (σ={scale}): strided CNN has {total} params, "
              f"output ≈ avg_pool(input, {512 // self.output_size}) at epoch 1",
              flush=True)

    def _circular_pad(self, x: torch.Tensor) -> torch.Tensor:
        p = self._pad
        return F.pad(x, (p, p, p, p), mode='circular')

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x_4d = x.unsqueeze(1)          # (B, 1, H, W)

        # Full-resolution feature extraction (circular padding manually).
        h = x_4d
        for layer in self.full_res:
            if isinstance(layer, nn.Conv2d):
                h = layer(self._circular_pad(h))
            else:
                h = layer(h)

        # Strided downsampling with circular padding at each step.
        for conv, act in zip(self.stride_layers, self.stride_acts):
            h = act(conv(self._circular_pad(h)))

        # 1×1 projection to single channel.
        proj = self.proj(h)             # (B, 1, out, out)

        # Residual skip: avg_pool(input) gives identity-like init.
        stride = 512 // self.output_size
        residual = F.avg_pool2d(x_4d, kernel_size=stride, stride=stride)

        field = (proj + residual).squeeze(1)    # (B, out, out)

        if self.standardize:
            mean = field.mean(dim=(1, 2), keepdim=True)
            std = field.std(dim=(1, 2), keepdim=True)
            field = (field - mean) / (std + 1e-8)

        return self.persistence(field)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
