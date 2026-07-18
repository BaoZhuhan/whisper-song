#!/usr/bin/env bash
#SBATCH --job-name=lyrics-env
#SBATCH --partition=common
#SBATCH --account=honglab
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
set -euo pipefail
cd /hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper
source cluster/env.sh
python3 -m venv "$VIRTUAL_ENV_PATH"
source "$VIRTUAL_ENV_PATH/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchaudio==2.11.0
python -m pip install -r requirements.txt
python -m pip check
python - <<'PY'
import torch, transformers, datasets, peft, soundfile, jiwer
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__, 'peft', peft.__version__)
PY
python -m pip freeze > requirements-lock.txt
touch "$OUTPUT_ROOT/gates/environment.PASS"
