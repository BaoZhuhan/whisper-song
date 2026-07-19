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

# Multiple compatible-GPU submissions may race for the same locked evaluation.
# Serialize them and let a later starter exit once the complete artifacts exist.
mkdir -p "$OUTPUT_ROOT/evaluation/medium-full"
exec 9>"$OUTPUT_ROOT/evaluation/medium-full/.audit_eval.lock"
flock -x 9
if [[ -s "$OUTPUT_ROOT/evaluation/medium-full/model_audit.json" && \
      -s "$OUTPUT_ROOT/evaluation/medium-finetuned/test/metrics.json" && \
      -s "$OUTPUT_ROOT/evaluation/medium-finetuned/test/predictions.jsonl" ]]; then
  echo "Locked medium evaluation already complete; exiting without overwriting artifacts."
  exit 0
fi

python scripts/audit_model.py --model "$MODEL" --output "$OUTPUT_ROOT/evaluation/medium-full/model_audit.json"
python scripts/evaluate.py --config configs/eval_medium_final.yaml --model-path "$MODEL" --model-label medium-finetuned --output-root "$OUTPUT_ROOT/evaluation"
