# Mandarin Lyrics Whisper

Reproducible Mandarin singing ASR experiments with Whisper. This repository is
currently at **phase 1**: cluster discovery and scheduler scaffolding only. No
dataset has been downloaded and no model result is claimed.

## Confirmed cluster facts (2026-07-18)

- Scheduler: Slurm.
- Current host `dcc-core-ferc-u-ab39-6-6` is a CPU node in partition `common`;
  `nvidia-smi` is absent. Treat it as a login/development node.
- Preferred accessible GPU resource: partition `scavenger-gpu`, GRES
  `gpu:6000_ada:1`, maximum wall time 7 days. This partition is preemptible and
  jobs must be checkpoint-resumable.
- Account association: `honglab`; normal QoS.
- Project storage is NFS with about 985 GiB free at inspection time. `/tmp` is
  node-local and non-persistent.

## Paths and configuration

The project defaults to this repository's persistent NFS directories. Override
any value at submission time; do not put checkpoints in home:

```bash
export PROJECT_ROOT=/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper
export STORAGE_ROOT=/hpc/group/honglab/zb78
export DATA_ROOT="$STORAGE_ROOT/dataset/whisper-song"
export OUTPUT_ROOT="$STORAGE_ROOT/model/whisper-song"
export HF_HOME="$STORAGE_ROOT/cache/huggingface"
export VIRTUAL_ENV_PATH="$STORAGE_ROOT/env/whisper-song"
export SLURM_PARTITION=scavenger-gpu
export SLURM_GRES=gpu:6000_ada:1
```

Inspect without submitting:

```bash
bash cluster/inspect_cluster.sh
bash cluster/submit_pipeline.sh --dry-run
```

Probe the actual GPU only after the Python environment is ready:

```bash
sbatch cluster/probe_gpu.sbatch
```

Formal training is deliberately gate-protected. `train_medium.sbatch` requires
`outputs/gates/medium_smoke.PASS`; LoRA requires
`outputs/gates/large_lora_smoke.PASS`. These files must only be created by the
future validation workflow after its reports have been reviewed.

## Completed provenance

- Environment job: `50201384`
- Base-model download job: `50201386`
- M4Singer download job: `50201385`
- M4Singer preprocessing/validation job: `50203074`
- Accepted samples: 20,896; rejected: 0; duration: 29.696 hours.
- Song-disjoint split: 16,834 train / 1,988 validation / 2,074 test.
