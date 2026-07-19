#!/usr/bin/env bash
#SBATCH --job-name=lyrics-medium-analysis
#SBATCH --partition=common
#SBATCH --account=honglab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
set -euo pipefail
cd /hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper
source cluster/env.sh
source "$VIRTUAL_ENV_PATH/bin/activate"
python scripts/analyze_errors.py \
 --baseline "$OUTPUT_ROOT/baseline/whisper-medium/test/predictions.jsonl" \
 --finetuned "$OUTPUT_ROOT/evaluation/medium-finetuned/test/predictions.jsonl" \
 --output-dir "$OUTPUT_ROOT/evaluation/medium-comparison" --bootstrap 10000 --seed 42
