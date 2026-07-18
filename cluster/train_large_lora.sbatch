#!/usr/bin/env bash
#SBATCH --job-name=lyrics-large-lora
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
[[ -f "$OUTPUT_ROOT/gates/large_lora_smoke.PASS" ]] || { echo 'Large LoRA smoke gate missing' >&2; exit 4; }
python scripts/train.py --config configs/train_large_v3_lora.yaml --output_dir "$OUTPUT_ROOT/checkpoints/large-v3-lora" --resume_from_checkpoint auto

