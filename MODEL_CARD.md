---
license: cc-by-nc-sa-4.0
language:
- zh
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-medium
tags:
- whisper
- singing-voice-recognition
- mandarin
- lyrics-transcription
- m4singer
datasets:
- M4Singer
metrics:
- cer
---

# Whisper Medium for Mandarin Singing Lyrics

`whisper-medium-m4singer-zh-lyrics` is a full-parameter fine-tune of
[`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) for
Mandarin singing-voice transcription. It was trained on the M4Singer corpus
with song-disjoint train, validation, and test splits.

## Intended use

The model transcribes short Mandarin singing phrases into Chinese lyrics. It is
intended for non-commercial research and evaluation. It is not designed for
speech translation, speaker identification, or unrestricted commercial use.

## Data and license

Training used M4Singer under its CC BY-NC-SA 4.0 non-commercial terms:

- 20,896 accepted segments; 0 rejected
- 29.70 hours
- 20 singers and 419 songs
- 16,834 train / 1,988 validation / 2,074 test segments
- song, file-path, and audio-hash overlap across splits: zero

Users must comply with the M4Singer license and citation requirements. The
training audio is not redistributed in this repository.

## Training

- Base model: `openai/whisper-medium`
- Method: full-parameter fine-tuning
- Language/task: `zh` / `transcribe`
- Hardware: one NVIDIA RTX 6000 Ada Generation (48 GB)
- Precision: BF16; TF32 enabled
- Gradient checkpointing: enabled
- Per-device batch size: 2
- Gradient accumulation: 16
- Learning rate: 5e-6, linear schedule
- Epochs/steps: 4 / 2,108
- Runtime: 3 h 48 min
- Seed: 42
- Best checkpoint selected on validation: step 2,108
- PyTorch peak allocated memory: 6.23 GiB (not an `nvidia-smi` peak)

## Results

All CER values use the same configurable Chinese normalization and micro
character aggregation.

| Evaluation | Original Whisper-medium | This model |
|---|---:|---:|
| Validation micro CER | 24.247% | **22.354%** |
| Quality-controlled test micro CER (2,072 paired segments) | 16.223% | **13.426%** |
| Quality-controlled macro song CER (40 songs) | 19.319% | **16.064%** |
| Quality-controlled exact-match rate | 41.795% | **48.600%** |

The paired macro-song CER improvement was 3.255 percentage points (10,000
paired song-bootstrap replicates; 95% CI, 2.388–4.128). The test analysis
excluded two paired samples after inspection because the fine-tuned decoder
entered a single-character repetition loop and reached the 440-token ceiling.
This post-hoc quality-control decision is fully documented and the raw results
are retained. On the complete 2,074-segment test set, CER was 16.268% for the
baseline and 18.074% for this model; the two failures contributed 869 of the
fine-tuned model's 933 insertions.

Full methods, figures and the exclusion record are available in the
[GitHub paper](https://github.com/BaoZhuhan/whisper-song/blob/main/paper/mandarin_singing_whisper/manuscript.md)
and its [appendix](https://github.com/BaoZhuhan/whisper-song/blob/main/paper/mandarin_singing_whisper/appendix.md).
The complete paper bundle can be downloaded as a
[ZIP archive](https://github.com/BaoZhuhan/whisper-song/raw/main/paper/mandarin_singing_whisper.zip).

## Usage

```python
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "Hengyuhan/whisper-medium-m4singer-zh-lyrics"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=torch.float16, attn_implementation="sdpa"
).to("cuda")

audio, sample_rate = sf.read("singing.wav", dtype="float32")
inputs = processor(
    audio, sampling_rate=sample_rate, return_tensors="pt",
    return_attention_mask=True,
)
features = inputs.input_features.to("cuda", dtype=torch.float16)
mask = inputs.attention_mask.to("cuda")
model.generation_config.language = "zh"
model.generation_config.task = "transcribe"

with torch.inference_mode():
    generated = model.generate(features, attention_mask=mask, max_new_tokens=128)
print(processor.batch_decode(generated, skip_special_tokens=True)[0])
```

Audio should be mono and no longer than Whisper's approximately 30-second
input window. Resample to 16 kHz when necessary.

## Limitations

- M4Singer is primarily clean, professionally recorded singing; robustness to
  accompaniment, live recordings, reverberation, and consumer microphones is
  not established.
- Unseen-singer, external-dataset, and hallucination evaluations are not yet
  complete.
- The model can insert or repeat lyrics during silence, breaths, or instrumental
  passages. Two test samples entered single-character loops under a 440-token
  ceiling. Use a task-appropriate output bound and do not treat generated text
  as verified ground truth.
- Performance is not evidence of broad Mandarin conversational ASR quality.

## Citation

Project repository and paper:

- <https://github.com/BaoZhuhan/whisper-song>
- <https://github.com/BaoZhuhan/whisper-song/tree/main/paper/mandarin_singing_whisper>

Please cite both Whisper and M4Singer when using this model. M4Singer:

```bibtex
@inproceedings{zhang2022msinger,
  title={M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus},
  author={Zhang, Lichao and Li, Ruiqi and Wang, Shoutong and Deng, Liqun and Liu, Jinglin and Ren, Yi and He, Jinzheng and Huang, Rongjie and Zhu, Jieming and Chen, Xiao and Zhao, Zhou},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2022}
}
```
