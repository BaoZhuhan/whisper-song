# Figure 2 QA notes

- Core conclusion: after transparent removal of two generation-ceiling repetition failures, the fine-tuned model reduces micro CER, macro song CER and every edit-error component on the paired test set.
- Archetype: quantitative grid. Panel a is the hero aggregate comparison, panel b decomposes the error change, and panel c shows all 40 paired song observations.
- Backend: Python only (`matplotlib`, `pandas`, `numpy`).
- Final size: 183 mm double-column width.
- Source data: 2,072 paired segments from 40 songs; two paired samples excluded under `configs/test_exclusions_decoding_failures.json`.
- Exclusion timing: post-hoc, after inspection of locked test predictions; this is stated in the manuscript and appendix.
- Independent unit for uncertainty: song. The interval uses 10,000 paired song-level bootstrap replicates with seed 42.
- No observations are omitted from the 40-song paired plot after applying the documented two-sample quality-control rule.
- Color: blue indicates improvement/fine-tuned results, gray indicates baseline, and red identifies songs that worsened; direction is also encoded geometrically.
- Export: editable SVG/PDF, 600-dpi TIFF and 300-dpi PNG preview.
