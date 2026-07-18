#!/usr/bin/env bash
#SBATCH --job-name=lyrics-probe
#SBATCH --partition=scavenger-gpu
#SBATCH --account=honglab
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
set -euo pipefail
cd "${PROJECT_ROOT:-/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper}"
source cluster/env.sh
[[ -f "$VIRTUAL_ENV_PATH/bin/activate" ]] && source "$VIRTUAL_ENV_PATH/bin/activate"
echo "job=${SLURM_JOB_ID:-unknown} host=$(hostname) start=$(date --iso-8601=seconds)"
nvidia-smi
python --version
python - <<'PY'
try:
    import torch
except ImportError:
    print("torch: not installed yet (GPU probe via nvidia-smi still valid)")
else:
    print("torch:", torch.__version__, "torch CUDA:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device:", p.name, "capability:", torch.cuda.get_device_capability(0))
        print("memory GiB:", p.total_memory / 1024**3)
PY
