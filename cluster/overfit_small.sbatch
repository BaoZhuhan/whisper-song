#!/usr/bin/env bash
#SBATCH --job-name=lyrics-overfit-small
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
[[ -f "$OUTPUT_ROOT/gates/single_batch.PASS" ]] || { echo 'Single-batch gate missing' >&2; exit 4; }
python scripts/train.py --config configs/overfit_small.yaml --output_dir "$OUTPUT_ROOT/smoke/overfit-small" --resume_from_checkpoint auto
