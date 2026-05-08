# TopoFisher

Code for the paper:

> **TopoFisher: Learning Topological Summary Statistics by Maximizing Fisher Information**  
> Matteo Biagetti, Mathieu Carrière, Francesco Conti, Enrico Maria Ferrari, Sven C. Heydenreich, Karthik Viswanathan  
> Preprint, 2025

TopoFisher is a differentiable persistent-homology pipeline that learns topological summary statistics for simulation-based inference. It optimizes filtrations, diagram vectorizations, and compressors by maximizing local Gaussian Fisher information — no posterior samples or supervised regression targets required.

## Installation

```bash
git clone https://github.com/RitAreaSciencePark/TopoFisher.git
cd TopoFisher
pip install -e .
```

Core dependencies (`gudhi`, `torch`, `numpy`, `scipy`, `pyyaml`, `tqdm`) are installed automatically. Additional dependencies for specific experiments:

```bash
# GNN-based spiral filtration (TF-TDA-GNN)
pip install torch-geometric

# Wavelet scattering baseline
pip install kymatio

# Weak lensing simulator (lognormal and LPT maps via sbi_lens)
pip install jax jax-cosmo
pip install git+https://github.com/DifferentiableUniverseInitiative/sbi_lens.git
```

### GUDHI GPU backend

The paper's large-scale lensing experiments use a GPU-accelerated cubical persistence backend (`backend='gudhi_gpu'`). This backend is based on a custom GUDHI fork with CUDA extensions that is **still under active development and not yet publicly released**. Anyone interested in early access should contact [matteo.biagetti@areasciencepark.it](mailto:matteo.biagetti@areasciencepark.it).

All experiments fall back to the standard CPU GUDHI backend automatically when the GPU extension is unavailable — results are identical, but runtime is longer for large ($512^2$) maps.

## Quick start

Run any experiment from a YAML configuration file:

```bash
# Fixed-filtration inference (no training needed)
python run_pipeline.py examples/configs/grf_cubical_pi.yaml

# Train a learnable pipeline (TF-Cubical-PersLay)
python run_pipeline.py examples/configs/grf_cubical_perslay.yaml --train

# Useful command-line overrides
python run_pipeline.py examples/configs/grf_cubical_perslay.yaml --train \
    --output-dir experiments/my_run \
    --n-epochs 500 \
    --seed 1
```

Results (`config.yaml`, `results.json`, `fisher_matrix.npy`, `pipeline.pt`) are saved in the output directory.

## Pipeline structure

Every configuration follows the same four-stage pipeline from the paper (Eq. 2):

```
Simulator → Filtration → Vectorization → Compression → Fisher Analyzer
```

| Stage | Role | Examples |
|---|---|---|
| **Simulator** | Generate data near a fiducial parameter value | GRF, noisy spiral, lensing |
| **Filtration** | Compute persistence diagrams (or a raw summary vector for non-TDA baselines) | cubical, alpha+DTM, CNN+persistence, IMNN |
| **Vectorization** | Map diagrams to Euclidean features | PersLay, persistence images, silhouettes, curves |
| **Compression** | Reduce to $d$ summaries (one per parameter) | MOPED (analytical), MLP (learned) |

## Example configurations

The `examples/configs/` directory contains ready-to-run configs for the paper's main experiments:

| Config | Experiment | Table |
|---|---|---|
| `grf_cubical_perslay.yaml` | TF-Cubical-PersLay on GRFs | Table 2 |
| `grf_cubical_pi.yaml` | Cubical-PI baseline on GRFs | Table 2 |
| `spiral_tf_tda_mlp.yaml` | TF-TDA-MLP on noisy spirals | Table 1 |
| `spiral_dtm.yaml` | DTM baseline on noisy spirals | Table 1 |
| `lensing_cubical_perslay.yaml` | TF-Cubical-PersLay on lensing | Table 3 |
| `lensing_imnn.yaml` | IMNN baseline on lensing | Table 3 |

Example scripts for reproducing paper tables are in `examples/scripts/`.

## Writing a config

A YAML config has four sections: `experiment`, `analysis`, `simulator`/`filtration`/`vectorization`/`compression`, and optionally `training` for learnable components.

### Minimal non-learnable example (GRF + fixed cubical + MOPED)

```yaml
experiment:
  name: my_grf_experiment
  output_dir: experiments/my_grf_experiment

analysis:
  theta_fid: [1.0, 0.0]      # fiducial (A_s, B)
  delta_theta: [0.1, 0.1]    # finite-difference step sizes
  n_s: 5000                  # samples for covariance
  n_d: 5000                  # samples for derivatives
  seed_cov: 42

simulator:
  type: grf
  params:
    N: 64
    dim: 2

filtration:
  type: cubical
  trainable: false
  params:
    homology_dimensions: [0, 1]
    periodic: true

vectorization:
  type: persistence_image
  trainable: false
  params:
    grid_size: 8
    bandwidth: 1.0
    weight: persistence

compression:
  type: moped
  trainable: false
  params:
    reg: 1.0e-8
```

### Learnable example (TF-Cubical-PersLay)

Add `training` to any config that has `trainable: true` components, then pass `--train`:

```yaml
vectorization:
  type: perslay
  trainable: true             # only this stage is learned
  params:
    point_dim: 16
    hidden_dim: 32
    spectral_norm: true       # Lipschitz control (recommended)

training:
  n_epochs: 2000
  lr: 1.0e-3
  batch_size: 500
  patience: 100
  seed: 0
  lambda_s: 0.05              # penalise skewness of compressed summaries
  lambda_k: 0.20              # penalise excess kurtosis
  lr_scheduler: plateau
  moped_refit_interval: 50    # refit MOPED every N epochs
```

### Available component types

**Simulators:** `grf`, `grf_fourier`, `gaussian_vector`, `noisy_ring`, `swiss_roll`, `lensing_lognormal`

**Filtrations (non-learnable):** `cubical`, `alpha`, `alpha_dtm`, `power_spectrum`, `peak_counts`, `scattering`, `identity`

**Filtrations (learnable):** `learnable_dense_point` (MLP on kNN distances → alpha complex), `gnn_point`, `cnn_fullres_persistence_v2` (CNN + cubical, paper's TF-CNN-PersLay), `imnn` (end-to-end IMNN baseline)

**Vectorizations (non-learnable):** `persistence_image`, `persistence_silhouette`, `differentiable_persistence_curves`, `persistence_landscape`, `topk`, `identity`

**Vectorizations (learnable):** `perslay`

**Compressions:** `moped` (analytical, lossless under Gaussianity), `mlp` (learned, requires `--train`), `identity`

## Reproducing paper results

### GRF benchmark (Table 2)

```bash
bash examples/scripts/run_grf_experiment.sh
```

Trains TF-Cubical-PersLay for all 5 spectral indices $B_0 \in \{-2,-1,0,1,2\}$ and 5 seeds. Runtime: ~6 min per job on 32 CPU cores.

### Noisy spiral benchmark (Table 1)

```bash
bash examples/scripts/run_spiral_experiment.sh
```

Trains TF-TDA-MLP for all 5 fiducial configurations. Runtime: ~30 min per configuration on CPU.

### Weak lensing benchmark (Table 3)

```bash
bash examples/scripts/run_lensing_experiment.sh
```

Trains per tomographic bin. The survey-level Fisher matrix is obtained by summing the 5 per-bin matrices. Runtime: ~63 min per bin per method on 32 CPU cores (TF-Cubical-PersLay); GPU required for TF-CNN-PersLay and IMNN (see Table 12 in the paper appendix).

Note: large simulation datasets are not included in the repository. They are regenerated automatically the first time each script is run.

## Citation

If you use this code, please cite:

```bibtex
@article{biagetti2025topofisher,
  title   = {{TopoFisher}: Learning Topological Summary Statistics by Maximizing {Fisher} Information},
  author  = {Biagetti, Matteo and Carri{\`e}re, Mathieu and Conti, Francesco and Ferrari, Enrico Maria and Heydenreich, Sven C. and Viswanathan, Karthik},
  year    = {2025},
  note    = {Preprint}
}
```
