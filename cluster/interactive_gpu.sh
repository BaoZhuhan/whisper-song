#!/usr/bin/env bash
set -euo pipefail
partition="${SLURM_PARTITION:-scavenger-gpu}"
gres="${SLURM_GRES:-gpu:6000_ada:1}"
exec srun --account="${SLURM_ACCOUNT:-honglab}" --partition="$partition" \
  --gres="$gres" --cpus-per-task=8 --mem=48G --time=01:00:00 \
  --qos=interactive --pty bash

