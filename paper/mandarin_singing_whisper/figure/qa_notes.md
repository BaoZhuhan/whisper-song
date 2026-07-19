# Figure 1 QA notes

- Core conclusion: full fine-tuning reduced complete-validation CER, and checkpoint 2,108 had the lowest observed formal validation CER.
- Archetype: quantitative grid; panel a is the hero comparison and panel b is checkpoint-selection evidence.
- Backend: Python only (`matplotlib` and `pandas`).
- Final size: 183 × 78 mm before tight bounding-box adjustment.
- Source-data rows: 11 total; no exclusions. The duplicate terminal evaluation entry at step 2,108 in the trainer log was removed because it is the same evaluation event recorded twice.
- Split: song-disjoint train/validation/test; panel a and panel b use the 1,988-segment validation partition.
- Metric: normalized micro CER; one deterministic evaluation per checkpoint.
- Variability/statistics: no error bars or inferential tests because only one full-training seed and one evaluation per checkpoint are available.
- Baseline definition: unadapted `openai/whisper-medium` evaluated with the complete validation manifest.
- Export: SVG and PDF retain editable text; TIFF is 600 dpi; PNG is a 300-dpi preview.
- Color: gray identifies the unadapted baseline, blue the fine-tuned series and red only the selected checkpoint; meaning is also encoded by position, label and marker.
- Reviewer risk: the curve is a model-selection trace, not an independent generalization result. Independent test evidence is reported separately in Fig. 2.
