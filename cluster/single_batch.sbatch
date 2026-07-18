#!/usr/bin/env bash
#SBATCH --job-name=lyrics-single-batch
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
[[ -f "$OUTPUT_ROOT/gates/data_validation.PASS" ]] || { echo 'Data validation gate missing' >&2; exit 4; }
python scripts/train.py --config configs/single_batch_small.yaml --output_dir "$OUTPUT_ROOT/smoke/single-batch"
