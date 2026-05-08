#!/bin/bash

# Default time is 1 hour if no argument is provided
HOURS=${1:-1}

srun --partition=small-g \
     --nodes=1 \
     --gpus-per-node=1 \
     --time=$(printf "%02d:00:00" "$HOURS") \
     --account=project_465002390 \
     --pty bash -c "
        # Source the environment activation script
        source /pfs/lustrep1/projappl/project_465001340/fair_stuff/TopoFisher/.claude/activate_env.sh
        exec bash
     "