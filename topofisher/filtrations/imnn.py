"""
IMNN — Information Maximizing Neural Network.

Reproduces the architecture of Charnock, Lavaux & Wandelt (2018,
arXiv:1802.03537): a CNN encoder followed by a fully-connected head that
outputs a small set of summaries (typically n_summaries = n_params).

The summaries are learned end-to-end to maximize the Fisher information
on the inference parameters. No analytic compression layer is used
downstream — the dense head IS the compression. Pair this filtration with
IdentityVectorization and IdentityCompression so the Fisher is computed
directly on the network's output.

Architecture (default for 512x512 lensing):
  Input (1, H, W)
    -> 5 strided conv layers (LeakyReLU, no BatchNorm)
    -> Flatten
    -> Dense (hidden_dim) + LeakyReLU
    -> Dense (n_summaries)

  No BatchNorm: it removes the parameter-dependent mean structure that
  carries the Fisher signal (same constraint as cnn_gap).

Output format mirrors cnn_gap (identity-filtration shape):
  forward(x) -> [[s_0, s_1, ..., s_{B-1}]]  with each s_i of shape (n_summaries,)
"""
from typing import List, Optional
import torch
import torch.nn as nn


class IMNNFiltration(nn.Module):
    """
    Information Maximizing Neural Network (Charnock et al., 2018).

    A learnable end-to-end summary extractor: convolutional trunk + dense head
    producing `n_summaries` outputs per sample, optimized to maximize Fisher
    information on the inference parameters via the standard TopoFisher
    -log|F| training loss.

    Args:
        n_summaries:        Output dimension (default 2 — one per parameter for
                            (Omega_m, sigma_8); the original IMNN paper uses
                            n_summaries = n_params).
        encoder_channels:   Output channels of each conv layer.
        encoder_kernels:    Kernel sizes per layer.
        encoder_strides:    Strides per layer.
        hidden_dim:         Width of the single hidden dense layer.
        circular_padding:   Use 'circular' padding mode for periodic fields.
        input_size:         Spatial size of the (square) input map. Required to
                            size the Flatten -> Dense transition.
        leaky_slope:        Negative slope of LeakyReLU (default 0.01).
    """

    def __init__(
        self,
        n_summaries: int = 2,
        encoder_channels: Optional[List[int]] = None,
        encoder_kernels: Optional[List[int]] = None,
        encoder_strides: Optional[List[int]] = None,
        hidden_dim: int = 128,
        circular_padding: bool = False,
        input_size: int = 512,
        leaky_slope: float = 0.01,
    ):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [16, 32, 32, 32, 16]
        if encoder_kernels is None:
            encoder_kernels = [7, 5, 3, 3, 3]
        if encoder_strides is None:
            encoder_strides = [4, 2, 2, 2, 2]

        assert len(encoder_channels) == len(encoder_kernels) == len(encoder_strides), \
            "encoder_channels, encoder_kernels, encoder_strides must have same length"

        pad_mode = 'circular' if circular_padding else 'zeros'
        layers = []
        in_ch = 1
        spatial = int(input_size)
        for out_ch, k, s in zip(encoder_channels, encoder_kernels, encoder_strides):
            pad = k // 2
            layers.append(nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad,
                                    padding_mode=pad_mode))
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope))
            in_ch = out_ch
            spatial = (spatial + 2 * pad - k) // s + 1

        self.encoder = nn.Sequential(*layers)
        self.flat_dim = in_ch * spatial * spatial
        self.input_size = int(input_size)
        self.n_summaries = int(n_summaries)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = self.n_summaries

        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, self.hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_slope),
            nn.Linear(self.hidden_dim, self.n_summaries),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        feat = self.encoder(x)
        flat = feat.flatten(start_dim=1)
        s = self.head(flat)

        vectors = [s[i] for i in range(s.shape[0])]
        return [vectors]

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        channels = [m.out_channels for m in self.encoder if isinstance(m, nn.Conv2d)]
        return (f"IMNNFiltration(channels={channels}, hidden={self.hidden_dim}, "
                f"n_summaries={self.n_summaries}, "
                f"num_params={self.get_num_parameters()})")
