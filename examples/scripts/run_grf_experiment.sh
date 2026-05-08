#!/usr/bin/env bash
# Run the GRF benchmark from Table 2 of the paper.
#
# Trains TF-Cubical-PersLay for all 5 spectral indices (B0) and 5 seeds,
# then runs the Cubical-PI baseline (no training needed) for comparison.
#
# Usage:
#   bash examples/scripts/run_grf_experiment.sh
#
# Results land in experiments/grf/.

set -euo pipefail

B_VALUES=(-2 -1 0 1 2)
SEEDS=(0 1 2 3 4)

for B in "${B_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "=== TF-Cubical-PersLay  B=${B}  seed=${SEED} ==="
        python run_pipeline.py \
            examples/configs/grf_cubical_perslay.yaml \
            --train \
            --output-dir "experiments/grf/cubical_perslay_B${B}_seed${SEED}" \
            --seed "${SEED}" \
            --seed-cov "$((SEED * 100 + 42))" \
            --seed-ders "[${SEED}00143, ${SEED}00144]"
    done
done

echo ""
echo "=== Cubical-PI baseline (no training) ==="
for B in "${B_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python run_pipeline.py \
            examples/configs/grf_cubical_pi.yaml \
            --output-dir "experiments/grf/cubical_pi_B${B}_seed${SEED}" \
            --seed-cov "$((SEED * 100 + 42))" \
            --seed-ders "[${SEED}00143, ${SEED}00144]"
    done
done

echo "Done. Results in experiments/grf/"
