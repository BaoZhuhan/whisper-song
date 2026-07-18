#!/usr/bin/env bash
#SBATCH --job-name=lyrics-model-download
#SBATCH --partition=common
#SBATCH --account=honglab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
set -euo pipefail
cd /hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper
source cluster/env.sh
[[ -f "$OUTPUT_ROOT/gates/environment.PASS" ]] || { echo 'Environment gate missing' >&2; exit 4; }
source "$VIRTUAL_ENV_PATH/bin/activate"
python scripts/download_models.py --model-root /hpc/group/honglab/zb78/model/whisper-song/base \
  --models openai/whisper-small openai/whisper-medium openai/whisper-large-v3
