#!/usr/bin/env bash
#SBATCH --job-name=lyrics-medium
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-23:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
[[ -f "$OUTPUT_ROOT/gates/medium_smoke.PASS" ]] || { echo 'Medium smoke gate missing' >&2; exit 4; }
python scripts/train.py --config configs/train_medium_full.yaml --output_dir "$OUTPUT_ROOT/checkpoints/medium-full" --resume_from_checkpoint auto

