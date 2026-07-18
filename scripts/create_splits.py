#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from lyrics_whisper.manifest import read_jsonl, write_jsonl
from lyrics_whisper.splits import grouped_song_split, leakage_errors

p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output-dir', required=True); p.add_argument('--seed', type=int, default=42)
a = p.parse_args(); rows = grouped_song_split(read_jsonl(a.input), seed=a.seed)
errors = leakage_errors(rows)
if errors: raise SystemExit("; ".join(errors))
for split in ('train','validation','test'):
    write_jsonl([r for r in rows if r['split'] == split], f"{a.output_dir}/{split}.jsonl")
print(dict(Counter(r['split'] for r in rows)))
