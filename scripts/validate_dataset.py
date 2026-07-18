#!/usr/bin/env python3
import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path
from lyrics_whisper.manifest import read_jsonl, validate_rows
from lyrics_whisper.splits import leakage_errors

p=argparse.ArgumentParser(); p.add_argument('--manifest-dir',required=True); p.add_argument('--report-dir',required=True); p.add_argument('--gate',required=True); p.add_argument('--seed',type=int,default=42)
a=p.parse_args(); report=Path(a.report_dir); report.mkdir(parents=True,exist_ok=True)
rows=[]
for split in ('train','validation','test'): rows.extend(read_jsonl(Path(a.manifest_dir)/f'{split}.jsonl'))
errors=validate_rows(rows)+leakage_errors(rows); splits=Counter(r['split'] for r in rows)
speakers=Counter(r['speaker_id'] for r in rows); songs=Counter(r['song_id'] for r in rows)
durations=[r['duration'] for r in rows]; texts=Counter(r['text_normalized'] for r in rows)
summary={"samples":len(rows),"hours":sum(durations)/3600,"speakers":len(speakers),"songs":len(songs),"splits":dict(splits),
 "duration_min":min(durations,default=0),"duration_mean":sum(durations)/len(durations) if durations else 0,"duration_max":max(durations,default=0),
 "duplicate_text_groups":sum(v>1 for v in texts.values()),"errors":errors}
(report/'data_quality_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
md=['# M4Singer data quality report','',f"- Samples: {summary['samples']}",f"- Hours: {summary['hours']:.3f}",f"- Singers: {summary['speakers']}",f"- Songs: {summary['songs']}",f"- Splits: {summary['splits']}",f"- Duration seconds (min/mean/max): {summary['duration_min']:.2f}/{summary['duration_mean']:.2f}/{summary['duration_max']:.2f}",f"- Duplicate normalized-text groups: {summary['duplicate_text_groups']}",f"- Validation errors: {len(errors)}"]
(report/'data_quality_report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
sample=random.Random(a.seed).sample(rows,min(100,len(rows))); parts=['<!doctype html><meta charset="utf-8"><title>M4Singer samples</title>']
for r in sample: parts.append(f"<section><p>{html.escape(r['id'])}: {html.escape(r['text_raw'])}</p><audio controls preload='none' src='file://{html.escape(r['audio_path'])}'></audio></section>")
(report/'sample_browser.html').write_text('\n'.join(parts),encoding='utf-8')
if errors or not rows: raise SystemExit(f'Dataset validation failed with {len(errors)} errors and {len(rows)} rows')
gate=Path(a.gate); gate.parent.mkdir(parents=True,exist_ok=True); gate.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
