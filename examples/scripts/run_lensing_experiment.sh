#!/usr/bin/env bash
# Run the weak lensing experiment from Table 3 of the paper.
#
# Trains TF-Cubical-PersLay independently for each of the 5 tomographic bins
# and 5 seeds, then adds their Fisher matrices for the survey-level constraint.
# Also runs the IMNN baseline for comparison.
#
# Requires: JAX, jax-cosmo, and sbi_lens (see README for install instructions).
#
# Usage:
#   bash examples/scripts/run_lensing_experiment.sh

set -euo pipefail

BINS=(0 1 2 3 4)
SEEDS=(0 1 2 3 4)

for BIN in "${BINS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=== TF-Cubical-PersLay  bin=${BIN}  seed=${SEED} ==="
        python run_pipeline.py \
            examples/configs/lensing_cubical_perslay.yaml \
            --train \
            --output-dir "experiments/lensing/cubical_perslay/bin${BIN}_seed${SEED}"

        echo "=== IMNN baseline  bin=${BIN}  seed=${SEED} ==="
        python run_pipeline.py \
            examples/configs/lensing_imnn.yaml \
            --train \
            --output-dir "experiments/lensing/imnn/bin${BIN}_seed${SEED}"
    done
done

echo "Done. Results in experiments/lensing/"
echo "Sum Fisher matrices across bins to obtain survey-level constraints."
