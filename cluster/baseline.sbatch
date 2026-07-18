#!/usr/bin/env bash
#SBATCH --job-name=lyrics-baseline
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
[[ -f "$OUTPUT_ROOT/gates/data_validation.PASS" ]] || { echo 'Data validation gate missing' >&2; exit 4; }
python scripts/evaluate.py --config configs/baseline.yaml "$@"
