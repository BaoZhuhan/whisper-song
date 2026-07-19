#!/usr/bin/env bash
#SBATCH --job-name=lyrics-medium-audit-eval
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
source cluster/job_common.sh
MODEL="$OUTPUT_ROOT/checkpoints/medium-full"
python scripts/audit_model.py --model "$MODEL" --output "$OUTPUT_ROOT/evaluation/medium-full/model_audit.json"
python scripts/evaluate.py --config configs/eval_medium_final.yaml --model-path "$MODEL" --model-label medium-finetuned --output-root "$OUTPUT_ROOT/evaluation"
