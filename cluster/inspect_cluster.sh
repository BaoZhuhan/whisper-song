#!/usr/bin/env bash
set -u
hostname
whoami
pwd
date
for cmd in sbatch srun squeue sinfo qsub bsub nvidia-smi; do
  command -v "$cmd" || true
done
sinfo -o '%P %a %l %D %G'
scontrol show partition scavenger-gpu
squeue -u "$USER"
df -h "${PROJECT_ROOT:-/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper}" /tmp

