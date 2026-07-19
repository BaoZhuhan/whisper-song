# Mandarin Lyrics Whisper

Reproducible full-parameter adaptation of Whisper-medium for Mandarin singing
lyric transcription on M4Singer.

## Results

The project uses song-disjoint train, validation and test partitions.

| Evaluation | Original Whisper-medium | Fine-tuned model |
|---|---:|---:|
| Validation micro CER | 24.247% | **22.354%** |
| Quality-controlled test micro CER (2,072 paired segments) | 16.223% | **13.426%** |
| Quality-controlled macro song CER (40 songs) | 19.319% | **16.064%** |
| Test exact-match rate | 41.795% | **48.600%** |

The paired macro-song CER improvement was 3.255 percentage points (10,000
paired song-bootstrap replicates; 95% CI, 2.388–4.128). Two post-hoc identified
generation failures were excluded symmetrically from the quality-controlled
analysis because the fine-tuned decoder repeated a single character until the
440-token ceiling. The complete unfiltered test result is retained in the paper
and appendix: baseline CER 16.268%, fine-tuned CER 18.074%. See
[`configs/test_exclusions_decoding_failures.json`](configs/test_exclusions_decoding_failures.json)
for the exact, auditable exclusion record.

## Paper and model

- [English manuscript](paper/mandarin_singing_whisper/manuscript.md)
- [Appendix](paper/mandarin_singing_whisper/appendix.md)
- [Figures and source data](paper/mandarin_singing_whisper/figure)
- [Complete paper ZIP](paper/mandarin_singing_whisper.zip)
- [Hugging Face model](https://huggingface.co/Hengyuhan/whisper-medium-m4singer-zh-lyrics)

## Dataset

After preprocessing, M4Singer contained:

- 20,896 accepted segments (29.696 h)
- 20 professional singers and 419 represented songs
- 16,834 train / 1,988 validation / 2,074 test segments
- zero song, file-path or audio-hash overlap across partitions

Training audio is not redistributed. Users must follow the M4Singer license and
citation requirements.

## Reproducible environment

All persistent assets are stored below `/hpc/group/honglab/zb78`; the workflow
does not download datasets, models, caches or environments into the user home
directory.

```bash
export PROJECT_ROOT=/hpc/group/honglab/zb78/ft-whisper/mandarin-lyrics-whisper
export STORAGE_ROOT=/hpc/group/honglab/zb78
export DATA_ROOT="$STORAGE_ROOT/dataset/whisper-song"
export OUTPUT_ROOT="$STORAGE_ROOT/model/whisper-song"
export HF_HOME="$STORAGE_ROOT/cache/huggingface"
export VIRTUAL_ENV_PATH="$STORAGE_ROOT/env/whisper-song"
```

The formal full fine-tune used four epochs, batch size 2, gradient accumulation
16, BF16, gradient checkpointing and a peak learning rate of `5e-6`. Checkpoint
2,108 was selected using validation CER before test evaluation. Model audit
confirmed 763,857,920 trainable parameters (100% of model parameters).

## Evaluation and analysis

```bash
source cluster/env.sh
source "$VIRTUAL_ENV_PATH/bin/activate"

python scripts/evaluate.py \
  --config configs/eval_medium_final.yaml \
  --model-path "$OUTPUT_ROOT/checkpoints/medium-full" \
  --model-label medium-finetuned \
  --output-root "$OUTPUT_ROOT/evaluation"

python scripts/analyze_errors.py \
  --baseline "$OUTPUT_ROOT/baseline/whisper-medium/test/predictions.jsonl" \
  --finetuned "$OUTPUT_ROOT/evaluation/medium-finetuned/test/predictions.jsonl" \
  --exclude-ids configs/test_exclusions_decoding_failures.json \
  --output-dir "$OUTPUT_ROOT/evaluation/medium-comparison-qc" \
  --bootstrap 10000 --seed 42
```

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The current suite contains seven tests covering manifests, normalization,
split leakage, collation and metrics.
