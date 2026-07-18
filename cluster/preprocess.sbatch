#!/usr/bin/env bash
#SBATCH --job-name=lyrics-preprocess
#SBATCH --partition=common
#SBATCH --account=honglab
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper}"
source cluster/env.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || exit 2
[[ -f "$VIRTUAL_ENV_PATH/bin/activate" ]] || { echo "Missing environment: $VIRTUAL_ENV_PATH" >&2; exit 2; }
source "$VIRTUAL_ENV_PATH/bin/activate"
mkdir -p "$DATA_ROOT/manifests" "$DATA_ROOT/reports"
python scripts/prepare_m4singer.py --raw-root "$DATA_ROOT/raw/m4singer" \
  --output "$DATA_ROOT/manifests/m4singer_all.jsonl" --rejected "$DATA_ROOT/reports/m4singer_rejected.jsonl"
python scripts/create_splits.py --input "$DATA_ROOT/manifests/m4singer_all.jsonl" --output-dir "$DATA_ROOT/manifests" --seed 42
python scripts/validate_dataset.py --manifest-dir "$DATA_ROOT/manifests" --report-dir "$DATA_ROOT/reports" \
  --gate "$OUTPUT_ROOT/gates/data_validation.PASS" --seed 42
