#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT:-/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper}"
source cluster/env.sh
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Refusing heavy work outside Slurm' >&2; exit 2; }
[[ -f "$VIRTUAL_ENV_PATH/bin/activate" ]] || { echo "Missing environment: $VIRTUAL_ENV_PATH" >&2; exit 2; }
source "$VIRTUAL_ENV_PATH/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "job=$SLURM_JOB_ID node=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi
