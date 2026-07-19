# Appendix

## A. Terminology ledger

| Canonical term | First-use definition | Usage decision |
|---|---|---|
| automatic speech recognition (ASR) | Mapping audio to text | Spell out once, then use ASR |
| character error rate (CER) | Character-level Levenshtein error rate | Report as a proportion in calculations and percentage in prose/tables |
| normalized micro CER | Corpus-level CER after the project text normalization | Primary development metric |
| macro song CER | CER computed per song and averaged across songs | Pending test statistic |
| Whisper-medium | `openai/whisper-medium` | Baseline model name |
| fine-tuned checkpoint 2,108 | Validation-selected final checkpoint | Never call it test-selected or test-optimal |
| M4Singer | Mandarin multi-style, multi-singer singing corpus | Preserve capitalization |

## B. Dataset audit

| Field | Value |
|---|---:|
| Accepted segments | 20,896 |
| Rejected segments | 0 |
| Total duration | 29.6955 h |
| Singers | 20 |
| Songs represented after preprocessing | 419 |
| Minimum segment duration | 0.830 s |
| Mean segment duration | 5.116 s |
| Maximum segment duration | 11.980 s |
| Train segments | 16,834 |
| Validation segments | 1,988 |
| Test segments | 2,074 |
| Song/path/audio-hash leakage errors | 0 |

Duplicate lyric strings were retained because repeated text is expected in songs. Partitioning was performed by song rather than by transcript string.

## C. Training configuration

```yaml
model_name_or_path: openai/whisper-medium
training_method: full
language: zh
task: transcribe
seed: 42
bf16: true
tf32: true
gradient_checkpointing: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 5.0e-6
num_train_epochs: 4
weight_decay: 0.01
warmup_ratio: 0.05
max_grad_norm: 1.0
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 250
save_steps: 250
logging_steps: 10
save_total_limit: 3
load_best_model_at_end: true
predict_with_generate: true
generation_max_length: 440
dataloader_num_workers: 8
```

The observed main-run training summary was: 2,108 steps, four epochs, 13,011.795 s trainer runtime, final training loss 12.6801, final validation loss 0.272593, final normalized validation CER 0.223541 and peak PyTorch allocated memory 6.228 GiB.

## D. Validation checkpoint trajectory

| Step | Epoch | Validation loss | Normalized micro CER (%) |
|---:|---:|---:|---:|
| 250 | 0.475 | 1.709650 | 31.970 |
| 500 | 0.950 | 0.288100 | 45.158 |
| 750 | 1.424 | 0.278852 | 27.133 |
| 1,000 | 1.899 | 0.275157 | 24.749 |
| 1,250 | 2.373 | 0.273577 | 22.464 |
| 1,500 | 2.848 | 0.272805 | 22.438 |
| 1,750 | 3.321 | 0.272611 | 22.453 |
| 2,000 | 3.796 | 0.272500 | 22.401 |
| 2,108 | 4.000 | 0.272593 | 22.354 |

The duplicate terminal evaluation record at step 2,108 in the trainer log was counted once in this table and in Fig. 1.

## E. Pipeline gates and job provenance

| Stage | Scheduler job | Outcome |
|---|---:|---|
| Dataset download | 50201385 | Completed |
| Dataset preprocessing | 50203074 | Completed |
| GPU probe | 50203313 | RTX 6000 Ada Generation; 47.37 GiB device memory |
| Baseline evaluation | 50203746 | Completed |
| Single-batch gate | 50203747 | Loss 5.5239; peak allocated memory 1.884 GiB |
| 32-segment overfit gate | 50205011 | 1.0256% training-subset CER after 300 steps |
| 200-step smoke run | 50205012 | 15.5009% CER on a 64-segment validation subset |
| Formal medium training | 50205013 | Completed; selected step 2,108 |
| Locked audit and final test, A5000 replica | 50237767 | Completed; 2,074 predictions |
| Original 6000 Ada duplicate | 50236424 | Cancelled after replica completed |
| Song-level bootstrap/error analysis | 50236425 | Completed; 10,000 replicates |

## F. Statistical analysis and quality-control exclusion

Micro CER aggregated all character-level edit counts before division, whereas macro song CER first calculated one CER per song and then averaged across 40 songs. To preserve dependence among segments from the same song, the 95% confidence interval for the baseline-minus-fine-tuned macro CER difference was estimated from 10,000 paired bootstrap resamples of songs using the NumPy default random generator with seed 42 and percentile limits at 2.5% and 97.5%.

Two paired observations were removed from the quality-controlled analysis under `configs/test_exclusions_decoding_failures.json`. Both fine-tuned predictions entered a single-character loop and reached the hard 440-new-token ceiling. This rule was created after test inspection and is therefore post-hoc. The corresponding baseline observations were also removed to retain pairing. Original predictions and complete-data statistics were not overwritten.

| Analysis | Baseline micro CER | Fine-tuned micro CER | Baseline macro song CER | Fine-tuned macro song CER |
|---|---:|---:|---:|---:|
| Complete, 2,074 segments | 16.268% | 18.074% | 19.361% | 26.652% |
| Quality-controlled, 2,072 segments | 16.223% | 13.426% | 19.319% | 16.064% |

The two removed fine-tuned predictions contributed 869 of 933 insertions (93.1%). In the remaining paired data, the fine-tuned and baseline models produced 64 and 66 insertions, respectively.

No inferential *P* value is currently reported. If one is later added, the null hypothesis, resampling procedure and two-sided/one-sided convention must be specified before inspecting the result. Singer-, duration- and lyric-length subgroup analyses are descriptive unless a multiplicity-controlled inferential plan is defined.

## G. Completed-result and remaining-work checklist

- [x] Confirm audit: 763,857,920 total and trainable parameters; trainable ratio 1.0; fused AdamW.
- [x] Confirm baseline and fine-tuned predictions use the same test manifest and evaluation implementation.
- [x] Insert complete and quality-controlled test CER.
- [x] Insert macro song CER and paired song-bootstrap interval.
- [x] Insert substitution, deletion, insertion and exact-match rates.
- [x] Preserve raw predictions and per-sample metrics.
- [x] Produce singer, duration and lyric-length subgroup summaries in the analysis JSON.
- [ ] Insert silence/interlude/breath hallucination results.
- [x] Re-run manuscript and figure consistency checks after inserting values.

## H. Figure contract and reproducibility

**Core conclusion:** Full fine-tuning lowers complete-validation CER relative to the unadapted Whisper-medium baseline, and the selected checkpoint lies at the lowest observed formal validation CER.

**Archetype:** Quantitative grid.

**Panel map:** Panel a shows the primary baseline comparison; panel b shows the complete formal checkpoint trajectory and the selection point.

**Statistics:** Figure 1 is descriptive. Figure 2 reports a 10,000-replicate paired song-bootstrap interval and shows all 40 paired song estimates after the documented two-sample exclusion.

**Source data:** `figure/source_data_figure1.csv`, `figure/source_data_figure2_summary.csv` and `figure/source_data_figure2_per_song.csv`.

**Rendering:** Run both Python scripts in `figure/` from this paper directory. Each writes editable SVG and PDF, 600-dpi TIFF and a PNG preview.

## I. Claim–evidence map

| Claim | Evidence | Status |
|---|---|---|
| Full fine-tuning improved complete-validation CER | 24.247% baseline versus 22.354% checkpoint 2,108 | Supported |
| The core training pipeline is functional | Unit tests, single-batch gate and 32-segment overfit CER | Supported |
| Checkpoint 2,108 was selected without test feedback | Validation checkpoint ranking and scheduler provenance | Supported |
| Fine-tuning improves quality-controlled test CER | 16.223% versus 13.426% after the disclosed paired exclusion | Supported under post-hoc QC boundary |
| The quality-controlled improvement is stable across songs | Paired bootstrap difference 3.255 pp, 95% CI 2.388–4.128 | Supported under post-hoc QC boundary |
| Fine-tuning improves unfiltered test CER | 16.268% versus 18.074% | Not supported; rare generation failures reverse the aggregate |
| The model generalizes to unseen singers or accompanied music | No suitable completed evaluation | **Needs evidence** |
