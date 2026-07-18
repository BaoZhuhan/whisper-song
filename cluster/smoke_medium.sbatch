#!/usr/bin/env bash
#SBATCH --job-name=lyrics-smoke-medium
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
[[ -f "$OUTPUT_ROOT/gates/small_overfit.PASS" ]] || { echo 'Small-overfit gate missing' >&2; exit 4; }
python scripts/train.py --config configs/smoke_medium.yaml --output_dir "$OUTPUT_ROOT/smoke/medium" --resume_from_checkpoint auto
