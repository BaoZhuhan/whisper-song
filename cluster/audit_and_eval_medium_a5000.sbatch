#!/usr/bin/env bash
#SBATCH --job-name=lyrics-medium-audit-eval-a5000
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

# Keep the evaluation implementation and locking logic identical across GPU variants.
exec bash cluster/audit_and_eval_medium.sbatch
