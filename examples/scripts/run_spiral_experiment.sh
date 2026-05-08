#!/usr/bin/env bash
# Run the noisy spiral benchmark from Table 1 of the paper.
#
# Trains TF-TDA-MLP for all 5 fiducial configurations and 5 seeds,
# then runs the fixed DTM baseline for comparison.
#
# Usage:
#   bash examples/scripts/run_spiral_experiment.sh

set -euo pipefail

# Five fiducial configurations from Table 4
MU_VALUES=(0.6 0.8 0.7 0.5 0.7)
W_VALUES=(0.1  0.1 0.2 0.2 0.3)
SEEDS=(0 1 2 3 4)

for i in "${!MU_VALUES[@]}"; do
    MU="${MU_VALUES[$i]}"
    W="${W_VALUES[$i]}"
    for SEED in "${SEEDS[@]}"; do
        TAG="mu${MU}_w${W}_seed${SEED}"
        echo "=== TF-TDA-MLP  mu=${MU}  w=${W}  seed=${SEED} ==="
        python run_pipeline.py \
            examples/configs/spiral_tf_tda_mlp.yaml \
            --train \
            --output-dir "experiments/spiral/tf_tda_mlp_${TAG}" \
            --seed "${SEED}"

        echo "=== DTM baseline  mu=${MU}  w=${W}  seed=${SEED} ==="
        python run_pipeline.py \
            examples/configs/spiral_dtm.yaml \
            --output-dir "experiments/spiral/dtm_${TAG}"
    done
done

echo "Done. Results in experiments/spiral/"
